import torch
import torch.nn as nn
import torch.nn.functional as F


class FSMVRPModel(nn.Module):
    def __init__(self, env_params, **model_params):
        super().__init__()
        self.model_params = model_params
        self.env_params = env_params

        # 1. BỘ MÃ HÓA (Dùng chung) - Đọc bản đồ
        self.encoder = FSMVRP_Encoder(**model_params)

        # 2. HAI BỘ GIẢI MÃ CHUYÊN BIỆT
        self.fleet_decoder = Fleet_Decoder(self.env_params, **model_params)
        self.route_decoder = Route_Decoder(self.env_params, **model_params)

        self.encoded_nodes = None
        self.reset_state = None

    def pre_forward(self, reset_state):
        """Chạy 1 lần đầu mỗi ván để lưu thông tin gốc và mã hóa bản đồ."""
        self.reset_state = reset_state
        depot_xy = reset_state.depot_xy
        node_xy = reset_state.node_xy
        node_demand = reset_state.node_demand

        # Gộp tọa độ và demand: (batch, problem, 3)
        node_xy_demand = torch.cat((node_xy, node_demand[:, :, None]), dim=2)

        # Mã hóa toàn bộ các Node
        self.encoded_nodes = self.encoder(depot_xy, node_xy_demand)

        # Nạp dữ liệu bản đồ vào Route Decoder
        self.route_decoder.set_kv(self.encoded_nodes)

    def forward_fleet(self, step_state):
        """HÀNH ĐỘNG 1: CHỌN XE (High-level)"""
        probs = self.fleet_decoder(self.encoded_nodes, step_state, self.reset_state)
        return self._select_action(probs, step_state.fleet_mask)

    def forward_route(self, step_state):
        """HÀNH ĐỘNG 2: CHỌN ĐIỂM ĐẾN (Low-level)"""
        probs = self.route_decoder(self.encoded_nodes, step_state, self.reset_state)
        return self._select_action(probs, step_state.ninf_mask)

    def _select_action(self, probs, mask):
        """Tiện ích chọn action chung dựa trên chế độ (train/eval)."""
        batch_size = probs.size(0)
        pomo_size = probs.size(1)

        if self.training or self.model_params.get('eval_type', 'softmax') == 'softmax':
            # [FIX #7] Thêm max_retry để tránh infinite loop
            # BẢN CŨ: while True → nếu tất cả prob=0 (mask sai), loop mãi mãi
            # BẢN MỚI: giới hạn 1000 lần, fallback sang argmax nếu thất bại
            max_retry = 1000
            retry_count = 0

            # [FIX #7b] Khởi tạo mặc định → tránh warning "referenced before assignment"
            selected = probs.argmax(dim=2)
            prob = probs.gather(2, selected.unsqueeze(2)).squeeze(2)

            while retry_count < max_retry:
                with torch.no_grad():
                    selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1) \
                        .squeeze(1).reshape(batch_size, pomo_size)

                # Lấy xác suất của node vừa chọn
                prob = probs.gather(2, selected.unsqueeze(2)).squeeze(2)

                # NẾU không có POMO nào bốc dính xác suất 0 (node cấm) -> bốc thành công, thoát!
                if (prob != 0).all():
                    break

                retry_count += 1

            # [FIX #7] Fallback: nếu hết retry, dùng argmax (chọn node xác suất cao nhất)
            if retry_count >= max_retry:
                selected = probs.argmax(dim=2)
                prob = probs.gather(2, selected.unsqueeze(2)).squeeze(2)

                # tránh log(0) cho bước tính Loss
            prob = prob.clamp(min=1e-12)

        else:
            selected = probs.argmax(dim=2)
            prob = probs.gather(2, selected.unsqueeze(2)).squeeze(2)

        return selected, prob

########################################
# encoder
########################################
class VehicleEncoder(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.agent_embedder = nn.Linear(3, embedding_dim)

        self.agent_attn_q = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.agent_attn_k = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.agent_attn_v = nn.Linear(embedding_dim, embedding_dim, bias=False)

        self.agent_attn_combine = nn.Linear(embedding_dim, embedding_dim)

        # [FIX #12] Thêm LayerNorm để ổn định embedding xe
        # BẢN CŨ: Không có normalization → scale embedding bất thường
        # BẢN MỚI: LayerNorm sau attention + residual connection
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, vehicle_features):
        agent_emb = self.agent_embedder(vehicle_features)

        q = self.agent_attn_q(agent_emb)
        k = self.agent_attn_k(agent_emb)
        v = self.agent_attn_v(agent_emb)

        attn_score = torch.matmul(q, k.transpose(-2, -1)) / (agent_emb.size(-1) ** 0.5)
        attn_prob = F.softmax(attn_score, dim=-1)

        agent_context = torch.matmul(attn_prob, v)

        # [FIX #12] Residual connection + LayerNorm
        # BẢN CŨ: agent_emb = self.agent_attn_combine(agent_context) → ghi đè embedding
        # BẢN MỚI: output = LayerNorm(original_emb + attention_output) → giữ lại thông tin gốc
        agent_attn_out = self.agent_attn_combine(agent_context)
        agent_emb = self.layer_norm(agent_emb + agent_attn_out)

        return agent_emb
########################################
# CÁC BỘ GIẢI MÃ (DECODERS)
########################################

class Fleet_Decoder(nn.Module):
    def __init__(self, env_params, **model_params):
        super().__init__()
        self.embedding_dim = model_params['embedding_dim']

        # 1. Gọi trực tiếp module VehicleEncoder (Thằng lính chuyên xử lý thông tin xe)
        self.vehicle_encoder = VehicleEncoder(self.embedding_dim)

        # 2. Khai báo Wq và Wk
        # Wq nhận input là [depot(Emb) + global_context(Emb) + demand_ratio(1)] = Emb*2 + 4
        self.SPATIAL_FEATURES = 3
        self.Wq = nn.Linear(self.embedding_dim * 2 + 1 + self.SPATIAL_FEATURES, self.embedding_dim)
        self.Wk = nn.Linear(self.embedding_dim, self.embedding_dim)

        # ===== Future Cost Estimator =====
        self.future_cost_mlp = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, 1)
        )
        self.beta = model_params.get("future_beta", 1.0)

    def _compute_spatial_features(self, reset_state, step_state, B, P):
        node_xy = reset_state.node_xy  # (B, N, 2)
        N = node_xy.size(1)

        customer_unvisited = ~step_state.visited_mask[:, :, 1:].bool()  # (B, P, N)
        unvisited_float = customer_unvisited.float()
        num_unvisited = unvisited_float.sum(dim=2, keepdim=True).clamp(min=1.0)

        node_xy_exp = node_xy.unsqueeze(1).expand(B, P, N, 2)

        unvisited_xy = node_xy_exp * unvisited_float.unsqueeze(-1)
        mean_xy = unvisited_xy.sum(dim=2) / num_unvisited  # (B, P, 2)

        mean_xy_exp = mean_xy.unsqueeze(2).expand(B, P, N, 2)
        sq_diff = (node_xy_exp - mean_xy_exp) ** 2
        sq_diff_masked = sq_diff * unvisited_float.unsqueeze(-1)

        coord_var = sq_diff_masked.sum(dim=2) / num_unvisited
        coord_std = coord_var.clamp(min=1e-8).sqrt()  # (B, P, 2)

        node_demand = reset_state.node_demand  # (B, N)
        node_demand_exp = node_demand.unsqueeze(1).expand(B, P, N)
        unvisited_demand = node_demand_exp * unvisited_float
        total_unvisited_demand = unvisited_demand.sum(dim=2, keepdim=True).clamp(min=1e-8)

        dist_sq_to_mean = sq_diff.sum(dim=-1)  # (B, P, N)
        weighted_var = (unvisited_demand * dist_sq_to_mean).sum(dim=2) / total_unvisited_demand.squeeze(-1)
        demand_weighted_spread = weighted_var.clamp(min=1e-8).sqrt()  # (B, P)

        return torch.cat([coord_std, demand_weighted_spread.unsqueeze(-1)], dim=-1)  # (B, P, 3)
    def forward(self, encoded_nodes, step_state, reset_state):
        B, P = step_state.BATCH_IDX.size(0), step_state.BATCH_IDX.size(1)

        # ===== 1. Agent embeddings =====
        cap = reset_state.agent_capacity.unsqueeze(-1)
        fix = reset_state.agent_fixed_cost.unsqueeze(-1)
        var = reset_state.agent_variable_cost.unsqueeze(-1)

        agent_feats = torch.cat([cap, fix, var], dim=-1)  # (B, K, 3)

        agent_emb = self.vehicle_encoder(agent_feats)  # (B, K, Emb)
        agent_emb = agent_emb.unsqueeze(1).expand(B, P, -1, -1)  # (B, P, K, Emb)
        # ----------------------------

        # Tạo Key cho các loại xe
        k = self.Wk(agent_emb)  # (B, P, K, Emb)

        # ===== 2. Global context từ node chưa thăm =====
        visited_mask = step_state.visited_mask.float()
        node_emb = encoded_nodes.expand(B, P, -1, -1)
        unvisited_mask = 1 - visited_mask
        unvisited_emb = node_emb * unvisited_mask.unsqueeze(-1)

        num_unvisited = unvisited_mask.sum(dim=2, keepdim=True) + 1e-8
        global_context = unvisited_emb.sum(dim=2) / num_unvisited  # (B, P, Emb)

        total_demand = reset_state.node_demand.sum(dim=1, keepdim=True)
        remaining_demand = (reset_state.node_demand.unsqueeze(1) *
                            (1 - visited_mask[:, :, 1:])).sum(dim=2, keepdim=True)
        demand_ratio = remaining_demand / (total_demand.unsqueeze(1) + 1e-8)  # (B, P, 1)

        depot_emb = encoded_nodes[:, :, 0, :].expand(B, P, -1)  # (B, P, Emb)

        # Tạo Query để chọn xe
        spatial_features = self._compute_spatial_features(reset_state, step_state, B, P)
        fleet_query = torch.cat([depot_emb, global_context, demand_ratio, spatial_features], dim=-1)
        q = self.Wq(fleet_query).unsqueeze(2)  # (B, P, 1, Emb)

        # ===== 3. Attention score =====
        score = torch.matmul(q, k.transpose(2, 3)).squeeze(2)  # (B, P, K)
        score_scaled = score / (self.embedding_dim ** 0.5)

        # ===== Future Cost Regularization =====
        future_cost = self.future_cost_mlp(agent_emb).squeeze(-1)

        mean = future_cost.mean(dim=-1, keepdim=True)
        std = future_cost.std(dim=-1, keepdim=True) + 1e-6
        future_cost_norm = (future_cost - mean) / std

        score_adjusted = score_scaled - self.beta * future_cost_norm

        score_masked = score_adjusted + step_state.fleet_mask

        probs = F.softmax(score_masked, dim=2)
        return probs


class Route_Decoder(nn.Module):
    """Bộ não chịu trách nhiệm Định tuyến."""

    def __init__(self, env_params, **model_params):
        super().__init__()
        self.model_params = model_params
        self.embedding_dim = model_params['embedding_dim']
        head_num = model_params.get('head_num', 8)
        qkv_dim = model_params.get('qkv_dim', 16)

        # Query Context: Node hiện tại(Emb) + Dung tích còn lại(1) + Phí xe(1) = Emb + 2
        self.Wq_context = nn.Linear(self.embedding_dim*2 + 2, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(self.embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(self.embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, self.embedding_dim)

        self.k = None
        self.v = None
        self.single_head_key = None

    def set_kv(self, encoded_nodes):
        head_num = self.model_params.get('head_num', 8)
        self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num)
        self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num)
        self.single_head_key = encoded_nodes.transpose(2, 3)

    def forward(self, encoded_nodes, step_state, reset_state):
        B, P = step_state.BATCH_IDX.size(0), step_state.BATCH_IDX.size(1)
        # Left graph embedding
        visited_mask = step_state.visited_mask.float()
        node_emb = encoded_nodes.expand(B, P, -1, -1)
        unvisited_mask = 1 - visited_mask
        unvisited_emb = node_emb * unvisited_mask.unsqueeze(-1)

        num_unvisited = unvisited_mask.sum(dim=2, keepdim=True) + 1e-8
        global_context = unvisited_emb.sum(dim=2) / num_unvisited  # (B, P, Emb)

        # 1. Thu thập thông tin hoàn cảnh hiện tại (Context)
        # 1.a. Node đang đứng
        curr_node_idx = step_state.current_node[:, :, None, None].expand(B, P, 1, self.embedding_dim)
        curr_node_emb = encoded_nodes.expand(B, P, -1, -1).gather(2, curr_node_idx).squeeze(2)

        # 1.b. Tải trọng còn lại trên xe
        load = step_state.current_load.unsqueeze(2)

        # 1.c. Phí di chuyển của chiếc xe đang lái
        curr_veh = step_state.current_vehicle_type.unsqueeze(2)

        # Nếu env dùng -1 khi chưa chọn xe
        invalid_mask = (curr_veh < 0)
        curr_veh = curr_veh.masked_fill(invalid_mask, 0)
        var_cost = reset_state.agent_variable_cost.unsqueeze(1).expand(B, P, -1).gather(2, curr_veh)

        # Gộp tất cả lại làm Context
        context = torch.cat([global_context, curr_node_emb, load, var_cost], dim=-1)  # (B, P, Emb + 2)

        # 2. Multi-Head Attention
        head_num = self.model_params.get('head_num', 8)
        q = reshape_by_heads(self.Wq_context(context).unsqueeze(2), head_num)

        out_concat = multi_head_attention(q, self.k, self.v)
        mh_atten_out = self.multi_head_combine(out_concat).squeeze(2)

        # 3. Tính điểm cuối cùng (Single-Head Attention)
        score = torch.matmul(mh_atten_out.unsqueeze(2), self.single_head_key.expand(B, P, -1, -1)).squeeze(2)

        score_scaled = score / (self.embedding_dim ** 0.5)
        logit_clipping = self.model_params.get('logit_clipping', 10.0)
        score_clipped = logit_clipping * torch.tanh(score_scaled)

        # 4. Áp dụng Mặt nạ (ninf_mask: Cấm đi vào node đã thăm hoặc quá nặng)
        score_masked = score_clipped.masked_fill(
            step_state.ninf_mask == float('-inf'),
            float('-inf')
        )

        # Guard: if all actions are masked (-inf), replace with 0 to avoid nan from softmax
        all_masked = (score_masked == float('-inf')).all(dim=2, keepdim=True)
        score_masked = score_masked.masked_fill(all_masked, 0.0)
        probs = F.softmax(score_masked, dim=2)
        return probs


########################################
# CÁC HÀM TIỆN ÍCH (HELPER FUNCTIONS)
########################################

def reshape_by_heads(qkv, head_num):
    B, M, N, D = qkv.size()
    q_reshaped = qkv.reshape(B, M, N, head_num, -1)
    return q_reshaped.transpose(2, 3)


def multi_head_attention(q, k, v):
    # Thiết kế gọn nhẹ cho SMDP
    B, P, H, _, D = q.size()
    N = k.size(3)

    k_exp = k.expand(B, P, H, N, D)
    v_exp = v.expand(B, P, H, N, D)

    score = torch.matmul(q, k_exp.transpose(3, 4))
    score_scaled = score / (D ** 0.5)
    weights = F.softmax(score_scaled, dim=-1)

    out = torch.matmul(weights, v_exp)
    out_transposed = out.transpose(2, 3).reshape(B, P,_, H * D)
    return out_transposed


########################################
# ENCODER (GIỮ NGUYÊN TỪ BẢN GỐC VÌ CHẠY RẤT TỐT)
########################################

class FSMVRP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        encoder_layer_num = self.model_params['encoder_layer_num']

        self.embedding_depot = nn.Linear(2, embedding_dim)
        self.embedding_node = nn.Linear(3, embedding_dim)
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])

    def forward(self, depot_xy, node_xy_demand):
        embedded_depot = self.embedding_depot(depot_xy)
        embedded_node = self.embedding_node(node_xy_demand)
        out = torch.cat((embedded_depot, embedded_node), dim=1)

        for layer in self.layers:
            out = layer(out)
        # [FIX #11] Sửa comment shape:
        # Input ban đầu là (Batch, Problem+1, Emb) [3D]
        # EncoderLayer.forward() sẽ unsqueeze thành (Batch, 1, Problem+1, Emb) [4D]
        # → Output cuối cùng là (Batch, 1, Problem+1, Emb) [4D]
        return out


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.add_n_normalization_1 = AddAndInstanceNormalization(**model_params)
        self.feed_forward = FeedForward(**model_params)
        self.add_n_normalization_2 = AddAndInstanceNormalization(**model_params)

    def forward(self, input1):
        head_num = self.model_params['head_num']
        if len(input1.size()) == 3:
            input1 = input1[:, None, :, :]

        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)

        out_concat = multi_head_attention(q, k, v)
        multi_head_out = self.multi_head_combine(out_concat)

        out1 = self.add_n_normalization_1(input1, multi_head_out)
        out2 = self.feed_forward(out1)
        out3 = self.add_n_normalization_2(out1, out2)

        return out3


class AddAndInstanceNormalization(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)

    def forward(self, input1, input2):
        added = input1 + input2
        transposed = added.transpose(2, 3)
        normalized = self.norm(transposed.squeeze(1)).unsqueeze(1)
        return normalized.transpose(2, 3)


class FeedForward(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        return self.W2(F.relu(self.W1(input1)))
