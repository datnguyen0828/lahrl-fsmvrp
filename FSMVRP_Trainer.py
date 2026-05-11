"""
FSMVRPTrainer_PPO.py
====================
Trainer nâng cao cho FSMVRP áp dụng các phương pháp từ tài liệu:

1. Phân tách Hàm lợi thế (Decoupled Advantage Functions)
   - A_fleet: Dùng mạng Critic V_phi(s) + TD-style advantage
   - A_route: Dùng POMO-style (so sánh với trung bình lô)

2. PPO với Clipping (thay thế REINFORCE đơn giản)
   - L_fleet_PPO: clip ratio r_fleet với epsilon
   - L_route_PPO: clip ratio r_route với epsilon

3. Future Cost Estimator Loss (MSE)
   - Huấn luyện mạng dự đoán chi phí tương lai

4. Entropy Bonus (khuyến khích khám phá)
   - S_entropy từ phân phối fleet và route

5. Critic Network Loss (L_VF)
   - MSE giữa V_phi(s) và phần thưởng tích lũy thực tế

Hàm tổng thể:
    L_total = L_fleet_PPO + L_route_PPO + λ*L_future - α*S_entropy + c*L_VF
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from logging import getLogger
from torch.utils.checkpoint import checkpoint as _grad_checkpoint
import torch.distributed as dist

from FSMVRP_Env import FSMVRPSMDPEnv as Env
from FSMVRP_Model import FSMVRPModel as Model

from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler

from utils import *


# ---------------------------------------------------------------------------
# Mạng Critic (State-Value Function V_phi(s))
# ---------------------------------------------------------------------------

class CriticNetwork(nn.Module):
    """
    [FIX #1] Nâng cấp Critic: nhận ĐẦY ĐỦ thông tin trạng thái động.

    BẢN CŨ chỉ nhận: mean(encoded_nodes) + avg_capacity + visited_ratio = Emb+2 features
    → Thiếu: current_load, accumulated_cost, route_count, demand_ratio
    → Critic ước tính V(s) kém → A_fleet thiếu chính xác

    BẢN MỚI nhận: mean(encoded_nodes) + 5 state features = Emb+5:
      1. avg_capacity     – tải trọng trung bình của các loại xe (thông tin tĩnh)
      2. visited_ratio    – tỉ lệ node đã thăm (tiến độ)
      3. current_load     – tải trọng còn lại trung bình trên xe hiện tại (trạng thái xe)
      4. accumulated_cost – chi phí đã tích lũy (biết đang tốn bao nhiêu rồi)
      5. demand_ratio     – tỉ lệ demand còn lại (biết còn bao nhiêu hàng cần giao)
    """

    # [FIX #1] Số feature trạng thái tăng từ 2 → 5
    STATE_FEATURES = 5

    def __init__(self, embedding_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Nhánh KHÔNG có state (dùng khi state=None, ví dụ inference)
        self.node_proj = nn.Linear(embedding_dim, hidden_dim)

        # [FIX #1] Nhánh CÓ state: Emb + 5 features thay vì Emb + 2
        self.state_proj = nn.Linear(embedding_dim + self.STATE_FEATURES, hidden_dim)

        # MLP để tính V(s)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, encoded_nodes, step_state=None, reset_state=None, detach_encoder=False):
        """
        [FIX #1] Thêm reset_state để lấy node_demand tính demand_ratio.

        Args:
            encoded_nodes: (B, 1, N+1, Emb) – đầu ra của encoder
            step_state:    Step_State – trạng thái động hiện tại
            reset_state:   Reset_State – thông tin tĩnh bài toán (cần node_demand)

        Returns:
            value: (B,) – giá trị trạng thái V(s)
        """
        # Lấy trung bình các node embedding làm biểu diễn trạng thái
        # (B, 1, N+1, Emb) → (B, Emb)
        # Nếu detach_encoder=True: gradient từ critic KHÔNG flow về encoder
    # Protect encoder khỏi "Gradient Tsunami" khi critic mới khởi tạo từ scratch
    # (sau khi transfer learning với re-init critic).
        if detach_encoder:
            encoded_nodes = encoded_nodes.detach()
        
        node_mean = encoded_nodes.squeeze(1).mean(dim=1)

        if step_state is not None:
            B = node_mean.shape[0]

            # --- Feature 1: capacity trung bình (thông tin tĩnh, chuẩn hóa /10) ---
            # agent_capacity thuộc Reset_State (thông tin tĩnh), KHÔNG thuộc Step_State
            if reset_state is not None and reset_state.agent_capacity is not None:
                cap = reset_state.agent_capacity.float().mean(dim=1, keepdim=True) / 10.0
            else:
                cap = torch.zeros(B, 1, device=node_mean.device)
            # shape: (B, 1)

            # --- Feature 2: tỉ lệ node đã visit (tiến độ hoàn thành) ---
            visited = step_state.visited_mask.float().mean(dim=(1, 2), keepdim=False).unsqueeze(1)
            # shape: (B, 1)

            # --- [FIX #1] Feature 3: tải trọng còn lại trung bình (qua POMO) ---
            # current_load: (B, pomo) → mean qua pomo → (B, 1)
            # Biết xe hiện tại còn chở được bao nhiêu
            load = step_state.current_load.float().mean(dim=1, keepdim=True) / 10.0
            # shape: (B, 1)

            # --- [FIX #1] Feature 4: chi phí đã tích lũy trung bình (qua POMO) ---
            # accumulated_cost: (B, pomo) → mean qua pomo → (B, 1)
            # Biết mình đang tốn bao nhiêu rồi → giúp ước tính V(s) chính xác hơn
            acc_cost = step_state.accumulated_cost.float().mean(dim=1, keepdim=True) / 100.0
            # shape: (B, 1)  (chia 100 để chuẩn hóa scale)

            # --- [FIX #1] Feature 5: tỉ lệ demand còn lại ---
            # Quan trọng: biết còn bao nhiêu hàng cần giao → ước tính chi phí còn lại
            if reset_state is not None and reset_state.node_demand is not None:
                total_demand = reset_state.node_demand.sum(dim=1, keepdim=True)  # (B, 1)
                # visited_mask[:, :, 1:] = mask cho customer nodes (bỏ depot index 0)
                remaining_demand = (reset_state.node_demand.unsqueeze(1) *
                                    (1 - step_state.visited_mask[:, :, 1:].float())).sum(dim=2)
                # remaining_demand: (B, pomo) → mean qua pomo → (B, 1)
                demand_ratio = remaining_demand.mean(dim=1, keepdim=True) / (total_demand + 1e-8)
            else:
                demand_ratio = torch.zeros(B, 1, device=node_mean.device)
            # shape: (B, 1)

            # Ghép tất cả: (B, Emb + 5)
            state_input = torch.cat([node_mean, cap, visited, load, acc_cost, demand_ratio], dim=1)
            hidden = F.relu(self.state_proj(state_input))
        else:
            # Fallback khi không có state (ví dụ test không cần Critic)
            hidden = F.relu(self.node_proj(node_mean))

        value = self.mlp(hidden).squeeze(-1)
        return value

# ---------------------------------------------------------------------------
# Trainer chính
# ---------------------------------------------------------------------------

class FSMVRPTrainer_PPO:
    def __init__(self,
                 env_params,
                 model_params,
                 optimizer_params,
                 trainer_params):

        # Lưu các tham số
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params

        # Tham số PPO (có thể override qua trainer_params)
        ppo_cfg = trainer_params.get('ppo', {})
        self.ppo_epsilon       = ppo_cfg.get('epsilon', 0.2)          # clipping ratio
        self.ppo_epochs        = ppo_cfg.get('ppo_epochs', 3)         # số vòng update PPO / batch
        self.gamma             = ppo_cfg.get('gamma', 0.99)           # hệ số chiết khấu
        self.lambda_future     = ppo_cfg.get('lambda_future', 0.5)    # λ cho L_future
        self.alpha_entropy     = ppo_cfg.get('alpha_entropy', 0.01)   # α cho S_entropy
        self.c_critic          = ppo_cfg.get('c_critic', 0.5)         # c cho L_VF
        self.initial_ppo_epsilon = self.ppo_epsilon
        self.initial_alpha_entropy = self.alpha_entropy
        self.initial_future_beta = model_params.get('future_beta', 1.0)

        # Logging & kết quả
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()
        self.result_log = LogData()

        # DDP Setup
        self.local_rank = trainer_params.get('local_rank', 0)
        self.world_size = trainer_params.get('world_size', 1)
        self.use_multi_gpu = self.world_size > 1
        self.device = torch.device('cuda', self.local_rank)

        # Các thành phần chính (Tự động nằm trên đúng GPU cục bộ)
        primary_env_params = self._env_params_for_device(self.device)
        self.model = Model(primary_env_params, **self.model_params).to(self.device)
        self.env = Env(**primary_env_params)

        # Mạng Critic riêng biệt
        embedding_dim = model_params['embedding_dim']
        critic_hidden = trainer_params.get('critic_hidden_dim', 256)
        self.critic = CriticNetwork(embedding_dim, critic_hidden).to(self.device)

        # Optimizer chung cho cả Actor + Critic
        all_params = list(self.model.parameters()) + list(self.critic.parameters())
        self.optimizer = Optimizer(all_params, **self.optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler'])

        # Gradient checkpointing cho decoder calls
        ckpt_cfg = trainer_params.get('gradient_checkpointing', {})
        self.grad_ckpt_enabled = ckpt_cfg.get('enable', False)
        if self.grad_ckpt_enabled and self.local_rank == 0:
            self.logger.info(
                "Gradient checkpointing ENABLED for decoder calls in _ppo_update. "
                "Expect ~60%% less VRAM but ~20%% slower per batch."
            )

        # Khôi phục checkpoint nếu có
       # Khôi phục checkpoint nếu có
        self.start_epoch = 1
        model_load = trainer_params['model_load']
        if model_load['enable']:
            checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=self.device)
            
            # CHỈ LOAD BỘ NÃO (Trọng số Model và Critic)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            
            self.logger.info(f'Warm Start: Đã load thành công chất xám từ {checkpoint_fullname} !!')
            
            if self.use_multi_gpu:
                self._sync_replicas_from_primary()
            
            # XÓA HOẶC COMMENT LẠI CÁC DÒNG DƯỚI ĐÂY:
            # Không load lại optimizer (để dùng learning rate mới từ Optuna)
            # Không load lại epoch (để chạy lại từ Epoch 1 cho 50 nodes)
            # self.start_epoch = 1 + model_load['epoch']
            # self.result_log.set_raw_data(checkpoint['result_log'])
            # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # self.scheduler.last_epoch = model_load['epoch'] - 1

        # Tiện ích thời gian
        self.time_estimator = TimeEstimator()

        # [FIX #9] Thêm Validation Set cố định
        # BẢN CŨ: Không có validation → không phát hiện overfitting
        #          best_train_score dựa trên dữ liệu random → artifact
        # BẢN MỚI: Tạo env riêng với seed cố định → eval trên cùng bộ dữ liệu mỗi epoch
        val_cfg = trainer_params.get('validation', {})
        self.val_enabled = val_cfg.get('enable', True)
        self.val_episodes = val_cfg.get('episodes', 100)
        self.val_batch_size = val_cfg.get('batch_size', 100)
        self.val_seed = val_cfg.get('seed', 9999)
        self.val_aug_factor = val_cfg.get('aug_factor', 8)
        self.val_primary_eval_type = val_cfg.get('primary_eval_type', 'softmax')
        self.val_secondary_eval_type = val_cfg.get('secondary_eval_type', None)
        self.objective_metric = val_cfg.get(
            'objective_metric',
            f"val_{self.val_primary_eval_type}_aug_score"
        )
        if self.val_enabled:
            self.val_env = Env(**primary_env_params)
            self.val_env.set_random_seed(self.val_seed, self.val_episodes)

        anneal_cfg = trainer_params.get('annealing', {})
        self.anneal_enabled = anneal_cfg.get('enable', False)
        self.anneal_start_epoch = anneal_cfg.get(
            'start_epoch',
            max(self.start_epoch, int(self.trainer_params['epochs'] * 0.8))
        )
        self.final_ppo_epsilon = anneal_cfg.get('final_ppo_epsilon', self.initial_ppo_epsilon)
        self.final_alpha_entropy = anneal_cfg.get('final_alpha_entropy', self.initial_alpha_entropy)
        self.final_future_beta = anneal_cfg.get('final_future_beta', self.initial_future_beta)

        transfer_cfg = trainer_params.get('transfer_learning', {})
        self.detach_encoder_epochs = transfer_cfg.get('detach_encoder_epochs', 0)
        self.should_detach_encoder_now = False  # sẽ update mỗi epoch
        if self.detach_encoder_epochs > 0 and self.local_rank == 0:
            self.logger.info(
                "Transfer learning: encoder will be DETACHED from critic "
                "for the first %d epochs.",
                self.detach_encoder_epochs,
            )
    def _log_diagnostics(self, tag: str, **kwargs):
        """
        [PATCH 3] Log diagnostics dạng key=value, dễ parse bởi analyze_diagnostics.py.
        Chỉ log trên rank 0, theo interval (mặc định 20 batches).

        Args:
            tag: nhãn dòng log, ví dụ 'ENTROPY', 'LOSS_COMP', 'ADV_STATS'
            **kwargs: các metric, ví dụ route_ent=1.5, fleet_ent=0.8
        """
        if self.local_rank != 0:
            return

        # Initialize counter & config nếu chưa có
        if not hasattr(self, '_diag_counter'):
            self._diag_counter = {}
            diag_cfg = self.trainer_params.get('diagnostics', {})
            self._diag_enabled = diag_cfg.get('enable', True)
            self._diag_interval = diag_cfg.get('interval_batches', 20)

        if not self._diag_enabled:
            return

        # Counter riêng cho từng tag
        self._diag_counter[tag] = self._diag_counter.get(tag, 0) + 1
        if self._diag_counter[tag] % self._diag_interval != 0:
            return

        # Format các giá trị: float dùng %.4f, các loại khác dùng str()
        msg_parts = []
        for k, v in kwargs.items():
            if isinstance(v, float):
                msg_parts.append(f"{k}={v:.4f}")
            else:
                msg_parts.append(f"{k}={v}")
        self.logger.info(f"[DIAG-{tag}] " + " ".join(msg_parts))
        
    def _sync_gradients(self):
        """Hòa trộn (All-Reduce) gradient từ tất cả các GPU qua cổng NCCL"""
        if not self.use_multi_gpu:
            return

        # Đồng bộ model
        for param in self.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= self.world_size

        # Đồng bộ critic
        for param in self.critic.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= self.world_size

    def _env_params_for_device(self, device):
        env_params = dict(self.env_params)
        env_params['device'] = device
        return env_params

    def _set_future_beta(self, beta_value):
        self.model.model_params['future_beta'] = beta_value
        if hasattr(self.model, 'fleet_decoder'):
            self.model.fleet_decoder.beta = beta_value
        if self.use_multi_gpu:
            for replica in self.replicas[1:]:
                replica['model'].model_params['future_beta'] = beta_value
                if hasattr(replica['model'], 'fleet_decoder'):
                    replica['model'].fleet_decoder.beta = beta_value

    def _interpolate_schedule(self, epoch, start_epoch, final_value, initial_value):
        if epoch <= start_epoch:
            return initial_value
        total_span = max(1, self.trainer_params['epochs'] - start_epoch)
        progress = min(1.0, (epoch - start_epoch) / total_span)
        return initial_value + (final_value - initial_value) * progress

    def _apply_epoch_schedules(self, epoch):
        if not self.anneal_enabled:
            return

        self.ppo_epsilon = self._interpolate_schedule(
            epoch, self.anneal_start_epoch, self.final_ppo_epsilon, self.initial_ppo_epsilon
        )
        self.alpha_entropy = self._interpolate_schedule(
            epoch, self.anneal_start_epoch, self.final_alpha_entropy, self.initial_alpha_entropy
        )
        current_future_beta = self._interpolate_schedule(
            epoch, self.anneal_start_epoch, self.final_future_beta, self.initial_future_beta
        )
        self._set_future_beta(current_future_beta)
        self.logger.info(
            "Schedule @ epoch %d -> ppo_epsilon=%.6f, alpha_entropy=%.6f, future_beta=%.6f",
            epoch, self.ppo_epsilon, self.alpha_entropy, current_future_beta
        )

    def _sync_replicas_from_primary(self):
        if not self.use_multi_gpu:
            return
        model_state = self.model.state_dict()
        critic_state = self.critic.state_dict()
        for replica in self.replicas[1:]:
            replica['model'].load_state_dict(model_state)
            replica['critic'].load_state_dict(critic_state)

    def _split_batch_sizes(self, batch_size):
        if not self.use_multi_gpu:
            return [batch_size]
        replica_count = min(len(self.replicas), batch_size)
        base = batch_size // replica_count
        remainder = batch_size % replica_count
        return [base + (1 if idx < remainder else 0) for idx in range(replica_count)]

    def _zero_replica_grads(self, replicas):
        for replica in replicas:
            replica['model'].zero_grad(set_to_none=True)
            replica['critic'].zero_grad(set_to_none=True)

    def _aggregate_replica_grads(self, replicas):
        primary_params = list(self.model.parameters()) + list(self.critic.parameters())
        secondary_param_groups = [
            list(replica['model'].parameters()) + list(replica['critic'].parameters())
            for replica in replicas[1:]
        ]
        for param_index, primary_param in enumerate(primary_params):
            for params in secondary_param_groups:
                replica_grad = params[param_index].grad
                if replica_grad is None:
                    continue
                replica_grad = replica_grad.to(primary_param.device)
                if primary_param.grad is None:
                    primary_param.grad = replica_grad.clone()
                else:
                    primary_param.grad.add_(replica_grad)

    def _score_from_rollout(self, rollout_data):
        reward = rollout_data['episode_reward']
        max_pomo_reward, _ = reward.max(dim=1)
        score_mean = -max_pomo_reward.float().mean()
        return score_mean.item()

    @staticmethod
    def _best_augmented_cost(cost_tensor, aug_factor, batch_size):
        aug_cost = cost_tensor.reshape(aug_factor, batch_size, -1)
        best_pomo_cost = aug_cost.min(dim=2).values
        best_aug_cost = best_pomo_cost.min(dim=0).values
        return best_aug_cost.float().mean().item()

    def _backward_replica_loss(self, replica, rollout_data, advantages_fleet, advantages_route, returns_fleet, weight):
        replica['model'].train()
        loss = self._compute_ppo_loss(
            rollout_data,
            advantages_fleet,
            advantages_route,
            returns_fleet,
            model=replica['model'],
            critic=replica['critic'],
            env=replica['env'],
        )
        (loss * weight).backward()
        return loss.item()

    def cleanup(self):
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=False)
            self._executor = None

        if hasattr(self, 'replicas'):
            for replica in self.replicas:
                replica['model'] = None
                replica['critic'] = None
                replica['env'] = None
            self.replicas = []

        self.model = None
        self.critic = None
        self.env = None
        if hasattr(self, 'val_env'):
            self.val_env = None

        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # Vòng lặp huấn luyện chính
    # -----------------------------------------------------------------------

    def run(self, trial=None):
        self.time_estimator.reset(self.start_epoch)
        best_train_score = float('inf')
        best_train_loss = None
        best_epoch = None
        final_train_score = None
        final_train_loss = None
        final_val_score = None
        # [FIX #9b] Tracking best validation score
        best_val_score = float('inf')
        best_objective_value = float('inf')
        best_objective_epoch = None
        final_objective_value = None
        best_eval_metrics = {}
        final_eval_metrics = {}

        for epoch in range(self.start_epoch, self.trainer_params['epochs'] + 1):
            self.logger.info('=================================================================')
            self._apply_epoch_schedules(epoch)

            train_score, train_loss = self._train_one_epoch(epoch)
            final_train_score = train_score
            final_train_loss = train_loss
            self.scheduler.step()
            self.result_log.append('train_score', epoch, train_score)
            self.result_log.append('train_loss', epoch, train_loss)

            # [FIX #9b] Chạy validation cuối mỗi epoch
            # BẢN CŨ: Không có validation → không biết model generalize tốt hay không
            # BẢN MỚI: Eval trên bộ dữ liệu cố định (seed=9999) → phát hiện overfitting
            if self.val_enabled:
                eval_types = [self.val_primary_eval_type]
                if self.val_secondary_eval_type and self.val_secondary_eval_type not in eval_types:
                    eval_types.append(self.val_secondary_eval_type)

                epoch_eval_metrics = {}
                for eval_type in eval_types:
                    eval_metrics = self._validate(eval_type=eval_type)
                    if self.local_rank == 0:
                        for metric_name, metric_value in eval_metrics.items():
                            epoch_eval_metrics[metric_name] = metric_value
                            self.result_log.append(metric_name, epoch, metric_value)
                            best_eval_metrics[metric_name] = min(
                                best_eval_metrics.get(metric_name, float('inf')),
                                metric_value
                            )
                            self.logger.info(
                                "  %s: %.4f (best: %.4f)",
                                metric_name, metric_value, best_eval_metrics[metric_name]
                            )
                if self.local_rank == 0:
                    final_eval_metrics = epoch_eval_metrics
                    if self.objective_metric in epoch_eval_metrics:
                        final_objective_value = epoch_eval_metrics[self.objective_metric]
                        final_val_score = final_objective_value
                        best_val_score = min(best_val_score, final_objective_value)
                        if final_objective_value < best_objective_value:
                            best_objective_value = final_objective_value
                            best_objective_epoch = epoch
                            self._save_checkpoint(epoch, suffix='best')
                    else:
                        raise KeyError(
                            f"Validation metric '{self.objective_metric}' was not produced. "
                            f"Available metrics: {sorted(epoch_eval_metrics.keys())}"
                        )

            if train_score < best_train_score:
                best_train_score = train_score
                best_train_loss = train_loss
                best_epoch = epoch

            # --- Logging thời gian ---
            elapsed_str, remain_str = self.time_estimator.get_est_string(
                epoch, self.trainer_params['epochs'])
            self.logger.info(
                "Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
                    epoch, self.trainer_params['epochs'], elapsed_str, remain_str))

            all_done = (epoch == self.trainer_params['epochs'])
            model_save_interval = self.trainer_params['logging']['model_save_interval']
            img_save_interval = self.trainer_params['logging']['img_save_interval']
            score_labels = ['train_score']
            if self.val_enabled:
                for eval_key in sorted(best_eval_metrics.keys()):
                    if self.result_log.has_key(eval_key):
                        score_labels.append(eval_key)

            # Lưu ảnh log (mỗi epoch)
            if epoch > 1:
                self.logger.info("Saving log_image")
                image_prefix = '{}/latest'.format(self.result_folder)
                # [FIX #9e] Thêm val_score vào biểu đồ cùng train_score
                # BẢN CŨ: Chỉ vẽ train_score → không thấy overfitting
                # BẢN MỚI: Vẽ cả val_score (nếu có) → dễ phát hiện overfitting
                util_save_log_image_with_label(
                    image_prefix,
                    self.trainer_params['logging']['log_image_params_1'],
                    self.result_log, labels=score_labels)
                util_save_log_image_with_label(
                    image_prefix,
                    self.trainer_params['logging']['log_image_params_2'],
                    self.result_log, labels=['train_loss'])

            # Lưu Model
            if all_done or ((epoch % model_save_interval) == 0) or (epoch <= 10):
                self.logger.info("Saving trained_model")
                self._save_checkpoint(epoch)

            # Lưu ảnh theo interval
            if all_done or (epoch % img_save_interval) == 0:
                image_prefix = '{}/img/checkpoint-{}'.format(self.result_folder, epoch)
                util_save_log_image_with_label(
                    image_prefix,
                    self.trainer_params['logging']['log_image_params_1'],
                    self.result_log, labels=score_labels)
                util_save_log_image_with_label(
                    image_prefix,
                    self.trainer_params['logging']['log_image_params_2'],
                    self.result_log, labels=['train_loss'])

            if trial is not None:
                report_value = best_objective_value if self.val_enabled else best_train_score
                trial.report(report_value, step=epoch)
                if (not math.isfinite(train_score)) or (not math.isfinite(train_loss)):
                    raise ValueError("Non-finite training metric encountered.")
                if trial.should_prune():
                    self.logger.info("Trial pruned at epoch {}.".format(epoch))
                    from optuna.exceptions import TrialPruned
                    raise TrialPruned()

            if all_done:
                self.logger.info(" *** Training Done *** ")
                self.logger.info("Now, printing log array...")
                util_print_log_array(self.logger, self.result_log)

        return {
            'objective_name': self.objective_metric if self.val_enabled else 'best_train_score',
            'objective_value': best_objective_value if self.val_enabled else best_train_score,
            'best_train_score': best_train_score,
            'best_train_loss': best_train_loss,
            'best_epoch': best_objective_epoch if self.val_enabled else best_epoch,
            'final_train_score': final_train_score,
            'final_train_loss': final_train_loss,
            'best_val_score': best_objective_value if self.val_enabled else None,
            'final_val_score': final_objective_value if self.val_enabled else None,
            'best_eval_metrics': best_eval_metrics if self.val_enabled else None,
            'final_eval_metrics': final_eval_metrics if self.val_enabled else None,
            'result_folder': self.result_folder,
        }

    def _save_checkpoint(self, epoch, suffix=None):
        checkpoint_dict = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'result_log': self.result_log.get_raw_data()
        }
        checkpoint_name = f'checkpoint-{epoch}.pt' if suffix is None else f'checkpoint-{suffix}.pt'
        torch.save(checkpoint_dict, '{}/{}'.format(self.result_folder, checkpoint_name))

    # -----------------------------------------------------------------------
    # Huấn luyện một epoch
    # -----------------------------------------------------------------------

    def _train_one_epoch(self, epoch):
        score_AM = AverageMeter()
        loss_AM = AverageMeter()

        # [TRANSFER LEARNING] Update flag detach encoder theo epoch.
        self.should_detach_encoder_now = (
            self.detach_encoder_epochs > 0
            and epoch < self.start_epoch + self.detach_encoder_epochs
        )
        if self.detach_encoder_epochs > 0 and self.local_rank == 0:
            status = "DETACHED" if self.should_detach_encoder_now else "flowing"
            self.logger.info(
                "Encoder gradient from critic @ epoch %d: %s",
                epoch, status,
            )

        train_num_episode = self.trainer_params['train_episodes']
        episode = 0
        loop_cnt = 0

        while episode < train_num_episode:
            remaining = train_num_episode - episode
            batch_size = min(self.trainer_params['train_batch_size'], remaining)

            avg_score, avg_loss = self._train_one_batch(batch_size)
            score_AM.update(avg_score, batch_size)
            loss_AM.update(avg_loss, batch_size)

            episode += batch_size

            # Log 10 batch đầu của epoch đầu tiên
            if epoch == self.start_epoch:
                loop_cnt += 1
                if loop_cnt <= 10:
                    self.logger.info(
                        'Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%)  '
                        'Score: {:.4f},  Loss: {:.4f}'.format(
                            epoch, episode, train_num_episode,
                            100. * episode / train_num_episode,
                            score_AM.avg, loss_AM.avg))

        self.logger.info(
            'Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f},  Loss: {:.4f}'.format(
                epoch, 100. * episode / train_num_episode,
                score_AM.avg, loss_AM.avg))

        return score_AM.avg, loss_AM.avg

    # -----------------------------------------------------------------------
    # [FIX #9c] Validation — eval trên bộ dữ liệu cố định
    # -----------------------------------------------------------------------

    def _validate(self, eval_type=None):
        """
        [FIX #9c] Chạy inference (không gradient) trên validation set cố định.

        BẢN CŨ: Không có validation → chỉ theo dõi train_score (trên data ngẫu nhiên)
                 → Không biết model generalize tốt hay đang overfitting
        BẢN MỚI: Dùng val_env với seed cố định → cùng bộ bài toán mỗi epoch
                  → So sánh val_score qua các epoch → phát hiện overfitting

        Returns:
            dict[str, float]
        """
        self.model.eval()
        raw_score_AM = AverageMeter()
        penalized_score_AM = AverageMeter()
        eval_type = eval_type or self.val_primary_eval_type
        aug_factor = self.val_aug_factor if self.val_aug_factor > 1 else 1
        previous_eval_type = self.model.model_params.get('eval_type', 'softmax')
        self.model.model_params['eval_type'] = eval_type

        # Reset lại index random để bắt đầu từ đầu danh sách validation
        self.val_env.random_list_index = 0
        episode = 0

        try:
            with torch.no_grad():
                while episode < self.val_episodes:
                    remaining = self.val_episodes - episode
                    batch_size = min(self.val_batch_size, remaining)

                    reset_state, _, _ = self.val_env.load_problems(batch_size, aug_factor=aug_factor)
                    self.val_env.reset()
                    self.model.pre_forward(reset_state)

                    state, _, done = self.val_env.pre_step()

                    while not done:
                        if state.need_fleet_action.any():
                            selected, _ = self.model.forward_fleet(state)
                            state, _, done = self.val_env.fleet_step(selected)
                        else:
                            selected, _ = self.model.forward_route(state)
                            state, _, done = self.val_env.route_step(selected)

                    raw_cost = self.val_env.get_raw_total_cost()
                    penalized_cost = self.val_env.get_total_cost()

                    raw_score_AM.update(
                        self._best_augmented_cost(raw_cost, aug_factor, batch_size),
                        batch_size
                    )
                    penalized_score_AM.update(
                        self._best_augmented_cost(penalized_cost, aug_factor, batch_size),
                        batch_size
                    )
                    episode += batch_size
        finally:
            self.model.model_params['eval_type'] = previous_eval_type

        return {
            f'val_{eval_type}_aug_score': raw_score_AM.avg,
            f'val_{eval_type}_aug_raw_score': raw_score_AM.avg,
            f'val_{eval_type}_aug_penalized_score': penalized_score_AM.avg,
        }

    # -----------------------------------------------------------------------
    # Huấn luyện một batch – lõi của PPO
    # -----------------------------------------------------------------------

    def _train_one_batch(self, batch_size: int):
        # 1. Thu thập dữ liệu trên GPU cục bộ
        rollout_data = self._collect_rollout(batch_size)

        # 2. Tính Advantage
        advantages_fleet, advantages_route, returns_fleet = \
            self._compute_decoupled_advantages(rollout_data)

        # 3. PPO Update
        self.model.train()
        total_loss = 0.0

        for _ in range(self.ppo_epochs):
            loss = self._ppo_update(
                rollout_data, advantages_fleet, advantages_route, returns_fleet
            )
            total_loss += loss

        avg_loss = total_loss / self.ppo_epochs

        # 4. Gom điểm số từ tất cả GPU về GPU 0 để in log
        score = self._score_from_rollout(rollout_data)

        if self.use_multi_gpu:
            score_tensor = torch.tensor(score, device=self.device)
            dist.all_reduce(score_tensor, op=dist.ReduceOp.SUM)
            score = (score_tensor / self.world_size).item()

        return score, avg_loss

    def _collect_rollout(self, batch_size: int, model=None, critic=None, env=None) -> dict:
        """
        Chạy một episode hoàn chỉnh và lưu lại:
        - Xác suất hành động fleet & route (cho PPO ratio)
        - Giá trị trạng thái từ Critic (cho A_fleet)
        - Phần thưởng từng tuyến (route reward) để tính future cost
        - Phần thưởng toàn tập (episode reward)
        """
        model = self.model if model is None else model
        critic = self.critic if critic is None else critic
        env = self.env if env is None else env

        with torch.no_grad():
            # Đẻ bài toán MỚI và hứng reset_state
            reset_state, _, _ = env.load_problems(batch_size)

            # Reset trạng thái xe cộ, hứng step_state
            step_state, _, _ = env.reset()

            # Nhét bản đồ vào Encoder
            model.pre_forward(reset_state)

            # Lưu encoded_nodes để tính Critic
            encoded_nodes = model.encoded_nodes  # (B, 1, N+1, Emb)

            # Danh sách lưu xác suất và action
            fleet_probs_list = []
            route_probs_list = []
            fleet_actions_list = []  # THÊM: Lưu lại quyết định chọn xe
            fleet_action_masks_list = []  # Mask POMO nào thực sự có fleet action ở mỗi slot
            route_actions_list = []  # THÊM: Lưu lại quyết định chọn node
            route_rewards_list = []

            # [FIX #3a] Thu thập reward RIÊNG cho route steps (chỉ từ route_step)
            # BẢN CŨ: route_rewards_list trộn lẫn fleet reward + route reward
            # BẢN MỚI: route_step_rewards chỉ chứa reward từ route actions
            #           → dùng để tính reward-to-go per-step cho A_route
            route_step_rewards_list = []
            route_rewards_per_fleet_step = []
            active_route_slot = torch.full(
                (batch_size, env.pomo_size), -1, dtype=torch.long, device=encoded_nodes.device
            )

            # Giá trị trạng thái tại đầu mỗi tuyến (cho A_fleet)
            critic_values_list = []

            state, reward, done = env.pre_step()


            while not done:
                need_fleet = state.need_fleet_action  # (B, pomo)

                if need_fleet.any():
                    # ── Hành động HIGH-LEVEL: Chọn xe ─
                    # [FIX #1b] Truyền thêm reset_state để Critic tính demand_ratio
                    v_s = critic(encoded_nodes, step_state=state, reset_state=reset_state)  # (B,)
                    critic_values_list.append(v_s)
                    fleet_action_masks_list.append(need_fleet.clone())

                    selected, prob = model.forward_fleet(state)
                    fleet_probs_list.append(prob)
                    fleet_actions_list.append(selected)  # Lưu action
                    route_rewards_per_fleet_step.append(torch.zeros_like(prob))

                    current_slot = len(route_rewards_per_fleet_step) - 1
                    active_route_slot = torch.where(
                        need_fleet,
                        torch.full_like(active_route_slot, current_slot),
                        active_route_slot
                    )

                    state, step_reward, done = env.fleet_step(selected)

                    if step_reward is not None:
                        route_rewards_list.append(step_reward)
                        route_rewards_per_fleet_step[current_slot] = (
                            route_rewards_per_fleet_step[current_slot] + step_reward * need_fleet.float()
                        )

                else:
                    # ── Hành động LOW-LEVEL: Chọn điểm ──
                    selected, prob = model.forward_route(state)
                    route_probs_list.append(prob)
                    route_actions_list.append(selected)  # Lưu action

                    state, step_reward, done = env.route_step(selected)

                    if step_reward is not None:
                        route_rewards_list.append(step_reward)
                        # [FIX #3a] Lưu reward CHỈ từ route step (không lẫn fleet reward)
                        route_step_rewards_list.append(step_reward)
                        for slot_idx, slot_reward in enumerate(route_rewards_per_fleet_step):
                            slot_mask = (active_route_slot == slot_idx)
                            if slot_mask.any():
                                route_rewards_per_fleet_step[slot_idx] = (
                                    slot_reward + step_reward * slot_mask.float()
                                )

                        route_finished = state.need_fleet_action | state.finished
                        active_route_slot = active_route_slot.masked_fill(route_finished, -1)

            # Phần thưởng toàn tập (B, pomo) – từ env
            episode_reward = env._get_episode_reward()

            # Stack thành tensor
        fleet_log_probs = self._stack_log_probs(fleet_probs_list, episode_reward.device)
        route_log_probs = self._stack_log_probs(route_probs_list, episode_reward.device)

        if len(route_rewards_list) > 0:
            route_rewards = torch.stack(route_rewards_list, dim=-1)
        else:
            route_rewards = episode_reward.unsqueeze(-1)

        if len(critic_values_list) > 0:
            critic_values = torch.stack(critic_values_list, dim=-1)
        else:
            critic_values = episode_reward.mean(dim=1, keepdim=True).expand(batch_size, 1)

        # [FIX #3a] Stack route-only rewards cho per-step advantage
        if len(route_step_rewards_list) > 0:
            route_step_rewards = torch.stack(route_step_rewards_list, dim=-1)  # (B, pomo, T_route)
        else:
            route_step_rewards = episode_reward.unsqueeze(-1)

        return {
            'encoded_nodes': encoded_nodes,
            'fleet_log_probs': fleet_log_probs,
            'route_log_probs': route_log_probs,
            'fleet_actions': torch.stack(fleet_actions_list, dim=2) if fleet_actions_list else None,
            'fleet_action_mask': torch.stack(fleet_action_masks_list, dim=2) if fleet_action_masks_list else None,
            'route_actions': torch.stack(route_actions_list, dim=2) if route_actions_list else None,
            'route_rewards': route_rewards,
            'route_rewards_per_fleet_step': (
                torch.stack(route_rewards_per_fleet_step, dim=-1)
                if route_rewards_per_fleet_step else None
            ),
            'route_step_rewards': route_step_rewards,  # [FIX #3a] reward per route step
            'critic_values': critic_values,
            'episode_reward': episode_reward,
        }

    # -----------------------------------------------------------------------
    # Tính Advantage phân tách
    # -----------------------------------------------------------------------

    def _compute_decoupled_advantages(self, rollout_data: dict):
        """
        [FIX #3b] Nâng cấp A_route: per-step reward-to-go + POMO baseline.

        BẢN CŨ:
            A_route(τ^j) = R(τ^j) - mean_j[R(τ^j)]
            → Một giá trị advantage GIỐNG NHAU cho TẤT CẢ route steps
            → Credit assignment kém: step tốt và step xấu nhận cùng tín hiệu

        BẢN MỚI:
            G_t = Σ_{k=t}^{T} r_k                    (reward-to-go từ step t)
            A_route(t, j) = G_t^j - mean_j[G_t^j]    (so sánh POMO tại từng step)
            → Mỗi step có advantage RIÊNG
            → Step tốt (reward-to-go cao) được khuyến khích
            → Step xấu (reward-to-go thấp) bị phạt

        A_fleet: TD-style dùng Critic (giữ nguyên)
            A_fleet(s_{r,0}, k_r) = R_route + γ * V(s_{r+1,0}) - V(s_{r,0})
        """
        episode_reward     = rollout_data['episode_reward']       # (B, pomo)
        critic_values      = rollout_data['critic_values']        # (B, T_fleet)
        route_rewards_fleet = rollout_data.get('route_rewards_per_fleet_step')  # (B, pomo, T_fleet)
        route_step_rewards = rollout_data['route_step_rewards']   # (B, pomo, T_route) [FIX #3a]

        # ══════════════════════════════════════════════════════════
        # ── A_route: Per-step Reward-to-Go + POMO baseline ──
        # ══════════════════════════════════════════════════════════

        T_route = route_step_rewards.shape[2]  # số route steps

        # Bước 1: Tính reward-to-go cho mỗi step
        # G_t = r_t + r_{t+1} + ... + r_T
        # Dùng cumsum ngược: hiệu quả O(T) thay vì O(T²)
        rewards_flipped = route_step_rewards.flip(dims=[2])          # Đảo ngược thời gian
        rtg_flipped = rewards_flipped.cumsum(dim=2)                  # Cumsum trên thời gian đảo
        reward_to_go = rtg_flipped.flip(dims=[2])                    # Đảo lại → (B, pomo, T_route)
        # reward_to_go[b, j, t] = tổng reward từ step t đến hết cho POMO instance j

        # Bước 2: POMO baseline cho TỪNG step
        # baseline_t = mean_j[G_t^j]  (trung bình qua POMO instances tại step t)
        pomo_mean_per_step = reward_to_go.float().mean(dim=1, keepdim=True)  # (B, 1, T_route)

        # Bước 3: Advantage per-step = G_t^j - baseline_t
        advantages_route_per_step = reward_to_go.float() - pomo_mean_per_step  # (B, pomo, T_route)

        # [PATCH 1] BỎ double normalization
        # Chuẩn hóa toàn cục để ổn định huấn luyện
        #advantages_route_per_step = self._normalize(advantages_route_per_step)

        # ══════════════════════════════════════════════════════════
        # ── A_fleet: TD-style với Critic (GIỮ NGUYÊN) ──
        # ══════════════════════════════════════════════════════════
        B, T = critic_values.shape

        if route_rewards_fleet is not None and route_rewards_fleet.shape[2] == T:
            route_r_mean = route_rewards_fleet.float().mean(dim=1)  # (B, T_fleet)
            v_next = torch.cat(
                [critic_values[:, 1:], torch.zeros(B, 1, device=critic_values.device)],
                dim=1
            )
            advantages_fleet_per_route = (
                route_r_mean + self.gamma * v_next - critic_values
            )
            returns_fleet = route_r_mean + self.gamma * torch.cat(
                [critic_values[:, 1:].detach(),
                 torch.zeros(B, 1, device=critic_values.device)], dim=1)
        else:
            # Fallback an toàn nếu rollout không có fleet-slot hợp lệ
            episode_adv = episode_reward.float() - episode_reward.float().mean(dim=1, keepdim=True)
            advantages_fleet_per_route = episode_adv.mean(dim=1, keepdim=True).expand(B, T)
            returns_fleet = episode_reward.float().mean(dim=1, keepdim=True).expand(B, T)

        # Chuẩn hóa A_fleet
        advantages_fleet = self._normalize(advantages_fleet_per_route)

        # [PATCH 6] Log advantage stats theo step group (early/mid/late)
        with torch.no_grad():
            T_r = advantages_route_per_step.shape[2] if advantages_route_per_step.dim() == 3 else 0
            adv_std_early = adv_std_mid = adv_std_late = 0.0
            adv_mean_abs = 0.0
            if T_r >= 3:
                adv_std_early = advantages_route_per_step[:, :, :T_r // 3].std().item()
                adv_std_mid = advantages_route_per_step[:, :, T_r // 3:2 * T_r // 3].std().item()
                adv_std_late = advantages_route_per_step[:, :, 2 * T_r // 3:].std().item()
                adv_mean_abs = advantages_route_per_step.abs().mean().item()

            # MSE-related raw stats: returns_fleet và route_step_rewards scale
            returns_abs_mean = returns_fleet.abs().mean().item() if returns_fleet.numel() > 0 else 0.0
            route_step_abs_mean = route_step_rewards.abs().mean().item() if route_step_rewards.numel() > 0 else 0.0

            self._log_diagnostics(
                'ADV_STATS',
                adv_std_early=adv_std_early,
                adv_std_mid=adv_std_mid,
                adv_std_late=adv_std_late,
                adv_mean_abs=adv_mean_abs,
                ratio_late_early=adv_std_late / max(adv_std_early, 1e-6),
                returns_fleet_abs_mean=returns_abs_mean,
                route_step_reward_abs_mean=route_step_abs_mean,
                T_route=T_r,
            )

        # [FIX #3b] Trả về advantages_route_per_step thay vì advantages_route (episode-level)
        return advantages_fleet, advantages_route_per_step, returns_fleet

    # -----------------------------------------------------------------------
    # Vòng cập nhật PPO
    # -----------------------------------------------------------------------

    def _compute_ppo_loss(self,
                          rollout_data: dict,
                          advantages_fleet,
                          advantages_route,
                          returns_fleet,
                          model=None,
                          critic=None,
                          env=None):
        """
        Một bước cập nhật PPO:
        L_total = L_fleet_PPO + L_route_PPO + λ*L_future - α*S_entropy + c*L_VF
        """

        model = self.model if model is None else model
        critic = self.critic if critic is None else critic
        env = self.env if env is None else env

        encoded_nodes = rollout_data['encoded_nodes']
        old_fleet_lp = rollout_data['fleet_log_probs'].detach()
        fleet_action_mask = rollout_data.get('fleet_action_mask')
        old_route_lp = rollout_data['route_log_probs'].detach()
        episode_reward = rollout_data['episode_reward']
        critic_values_old = rollout_data['critic_values'].detach()

        # Lấy Action cũ ra để ép Model đi lại đường cũ
        old_fleet_actions = rollout_data['fleet_actions']
        old_route_actions = rollout_data['route_actions']

        batch_size = encoded_nodes.shape[0]
        pomo_size = episode_reward.shape[1]

        # ─── Chạy lại forward pass để lấy probs MỚI trên BẢN ĐỒ CŨ ───

        # [FIX #10] Dùng restore_problem() thay vì truy cập trực tiếp env.reset_state
        # BẢN CŨ: reset_state = self.env.reset_state → mong env vẫn giữ nguyên data
        #          Nếu env bị thay đổi giữa các PPO epochs → dữ liệu sai
        # BẢN MỚI: Restore CHÍNH XÁC data từ rollout → đảm bảo replay cùng bài toán
        saved_problem = {
            'depot_xy':            env.reset_state.depot_xy,
            'node_xy':             env.reset_state.node_xy,
            'node_demand':         env.reset_state.node_demand,
            'agent_capacity':      env.reset_state.agent_capacity,
            'agent_fixed_cost':    env.reset_state.agent_fixed_cost,
            'agent_variable_cost': env.reset_state.agent_variable_cost,
        }
        env.restore_problem(saved_problem)
        reset_state = env.reset_state

        # 2. Reset xe cộ về Depot
        step_state, _, _ = env.reset()

        model.pre_forward(reset_state)

        new_fleet_probs_list = []
        new_route_probs_list = []
        fleet_entropy_list = []
        route_entropy_list = []
        future_pred_list = []
        future_true_list = []
        v_pred_list = []

        state, _, done = env.pre_step()

        route_step_idx = 0
        fleet_step_idx = 0
        accumulated_route_cost = torch.zeros(batch_size, pomo_size, device=episode_reward.device)

        # [FIX #4] Thêm biến tích lũy TỔNG chi phí (fixed + variable)
        # BẢN CŨ: accumulated_route_cost chỉ chứa variable cost từ route_step
        #          → future_true = total_cost - only_variable_cost ≠ remaining_cost thực
        # BẢN MỚI: accumulated_total_cost bao gồm CẢ fixed cost từ fleet_step
        accumulated_total_cost = torch.zeros(batch_size, pomo_size, device=episode_reward.device)
        agent_feats_static = torch.cat([
            reset_state.agent_capacity.unsqueeze(-1),
            reset_state.agent_fixed_cost.unsqueeze(-1),
            reset_state.agent_variable_cost.unsqueeze(-1)
        ], dim=-1)
        while not done:
            need_fleet = state.need_fleet_action

            if need_fleet.any():
                #Critic
                # [FIX #1b] Truyền thêm reset_state để Critic tính demand_ratio
                v_s_new = critic(
                    model.encoded_nodes,
                    step_state=state,
                    reset_state=reset_state,
                    detach_encoder=self.should_detach_encoder_now,
                )
                v_pred_list.append(v_s_new)
                # HIGH-LEVEL fleet action
                if self.grad_ckpt_enabled:
                    probs = _grad_checkpoint(
                        model.fleet_decoder,
                        model.encoded_nodes, state, reset_state,
                        use_reentrant=False,
                    )
                else:
                    probs = model.fleet_decoder(model.encoded_nodes, state, reset_state)  # (B, pomo, K)

                # ÉP CHỌN ACTION CŨ
                if old_fleet_actions is not None and fleet_step_idx < old_fleet_actions.shape[2]:
                    selected = old_fleet_actions[:, :, fleet_step_idx]
                else:
                    selected, _ = model.forward_fleet(state)

                # Log prob mới (Có clamp để chống lỗi NaN)
                 # [PATCH 2] Đồng bộ clamp với _stack_log_probs (1e-12) để tránh ratio explosion
                new_log_prob = probs.clamp(min=1e-12).log().gather(2, selected.unsqueeze(2)).squeeze(2)
                new_fleet_probs_list.append(new_log_prob)

                # Entropy fleet
                fleet_entropy = -(probs * probs.clamp(min=1e-12).log()).sum(dim=-1)
                fleet_entropy_list.append(fleet_entropy)

                # Future cost prediction
                # Chỉ truyền 1 tham số agent_feats_static vào vehicle_encoder
                agent_emb = model.fleet_decoder.vehicle_encoder(agent_feats_static)
                # Tính future_pred_base: (B, K, Emb) -> MLP -> (B, K, 1) -> squeeze -> (B, K) -> mean -> (B,)
                future_pred_base = model.fleet_decoder.future_cost_mlp(agent_emb).squeeze(-1).mean(dim=-1)
                # Mở rộng ra shape (Batch, POMO) để khớp kích thước với future_true
                future_pred = future_pred_base.unsqueeze(1).expand(batch_size, pomo_size)
                future_pred_list.append(future_pred)
                # [FIX #4] Dùng accumulated_total_cost (bao gồm cả fixed cost)
                # BẢN CŨ: cost_so_far = accumulated_route_cost.mean(dim=1)
                #   → Chỉ tính variable cost → future_true sai (cao hơn thực tế)
                # BẢN MỚI: cost_so_far = accumulated_total_cost.mean(dim=1)
                #   → Bao gồm cả fixed + variable → future_true chính xác
                cost_so_far = accumulated_total_cost
                future_true = (-episode_reward.float()) - cost_so_far
                future_true_list.append(future_true.detach())

                state, step_reward, done = env.fleet_step(selected)

                # [FIX #4] Tích lũy fixed cost từ fleet_step vào accumulated_total_cost
                if step_reward is not None:
                    accumulated_total_cost += step_reward.float().abs()

                fleet_step_idx += 1

            else:
                # LOW-LEVEL route action
                if self.grad_ckpt_enabled:
                    probs = _grad_checkpoint(
                        model.route_decoder,
                        model.encoded_nodes, state, reset_state,
                        use_reentrant=False,
                    )
                else:
                    probs = model.route_decoder(model.encoded_nodes, state, reset_state)
                # ÉP CHỌN ACTION CŨ
                if old_route_actions is not None and route_step_idx < old_route_actions.shape[2]:
                    selected = old_route_actions[:, :, route_step_idx]
                else:
                    selected, _ = model.forward_route(state)
                    # [PATCH 2] Đồng bộ clamp
                # Log prob mới (Có clamp chống NaN)
                new_log_prob = probs.clamp(min=1e-12).log().gather(2, selected.unsqueeze(2)).squeeze(2)
                new_route_probs_list.append(new_log_prob)

                # Entropy route
                safe_probs = probs.clamp(min=1e-12)
                route_entropy = -(safe_probs * safe_probs.log()).sum(dim=-1)
                route_entropy_list.append(route_entropy)

                state, step_reward, done = env.route_step(selected)
                if step_reward is not None:
                    accumulated_route_cost += step_reward.float().abs()
                    # [FIX #4] Cũng tích lũy vào total cost (cho future_true chính xác)
                    accumulated_total_cost += step_reward.float().abs()
                route_step_idx += 1

        # ─── Tính các thành phần Loss ───

        # 1. L_fleet_PPO
        if new_fleet_probs_list and old_fleet_lp.shape[2] > 0:
            new_fleet_lp = torch.stack(new_fleet_probs_list, dim=2)
            T_f = min(new_fleet_lp.shape[2], old_fleet_lp.shape[2])
            new_fleet_lp = new_fleet_lp[:, :, :T_f]
            old_fleet_lp_t = old_fleet_lp[:, :, :T_f]

            adv_f = advantages_fleet[:, :T_f].unsqueeze(1).expand_as(new_fleet_lp)

            ratio_fleet = (new_fleet_lp - old_fleet_lp_t).exp()
            surr1 = ratio_fleet * adv_f
            surr2 = ratio_fleet.clamp(1 - self.ppo_epsilon, 1 + self.ppo_epsilon) * adv_f
            if fleet_action_mask is not None:
                fleet_mask_t = fleet_action_mask[:, :, :T_f].float()
                denom = fleet_mask_t.sum().clamp_min(1.0)
                L_fleet_ppo = -(torch.min(surr1, surr2) * fleet_mask_t).sum() / denom
            else:
                L_fleet_ppo = -torch.min(surr1, surr2).mean()
        else:
            L_fleet_ppo = encoded_nodes.new_tensor(0.0)

        # 2. L_route_PPO
        # [FIX #3c] advantages_route giờ là (B, pomo, T_route) — per-step
        # BẢN CŨ: advantages_route là (B, pomo), broadcast cùng 1 giá trị cho mọi step
        # BẢN MỚI: mỗi step t có advantage riêng A_t = G_t - mean_j(G_t)
        if new_route_probs_list and old_route_lp.shape[2] > 0:
            new_route_lp = torch.stack(new_route_probs_list, dim=2)
            T_r = min(new_route_lp.shape[2], old_route_lp.shape[2])
            new_route_lp = new_route_lp[:, :, :T_r]
            old_route_lp_t = old_route_lp[:, :, :T_r]

            # [FIX #3c] Cắt per-step advantage cho khớp số step
            # advantages_route: (B, pomo, T_route) → cắt thành (B, pomo, T_r)
            T_a = min(advantages_route.shape[2], T_r) if advantages_route.dim() == 3 else T_r
            if advantages_route.dim() == 3:
                adv_r = advantages_route[:, :, :T_a]
                # Nếu T_a < T_r, pad bằng 0 (step thừa không có advantage)
                if T_a < T_r:
                    pad = torch.zeros(advantages_route.shape[0], advantages_route.shape[1],
                                      T_r - T_a, device=advantages_route.device)
                    adv_r = torch.cat([adv_r, pad], dim=2)
            else:
                # Fallback: nếu vẫn là 2D (tương thích ngược)
                adv_r = advantages_route.unsqueeze(2).expand_as(new_route_lp)

            ratio_route = (new_route_lp - old_route_lp_t).exp()
            surr1 = ratio_route * adv_r
            surr2 = ratio_route.clamp(1 - self.ppo_epsilon, 1 + self.ppo_epsilon) * adv_r
            L_route_ppo = -torch.min(surr1, surr2).mean()
        else:
            L_route_ppo = encoded_nodes.new_tensor(0.0)

        # 3. L_future (MSE cho Future Cost Estimator)
        if future_pred_list:
            future_pred_stack = torch.stack(future_pred_list, dim=1)
            future_true_stack = torch.stack(future_true_list, dim=1)
            L_future = F.mse_loss(future_pred_stack, future_true_stack)
        else:
            L_future = encoded_nodes.new_tensor(0.0)

        # 4. S_entropy (Entropy bonus)
        S_entropy = encoded_nodes.new_tensor(0.0)
        if fleet_entropy_list:
            S_entropy = S_entropy + torch.stack(fleet_entropy_list, dim=2).mean()
        if route_entropy_list:
            S_entropy = S_entropy + torch.stack(route_entropy_list, dim=2).mean()
        with torch.no_grad():
            fleet_ent_mean = (torch.stack(fleet_entropy_list, dim=2).mean().item()
                              if fleet_entropy_list else 0.0)
            route_ent_mean = 0.0
            route_ent_early = 0.0
            route_ent_mid = 0.0
            route_ent_late = 0.0
            if route_entropy_list:
                route_ent_tensor = torch.stack(route_entropy_list, dim=2)  # (B, pomo, T)
                route_ent_mean = route_ent_tensor.mean().item()
                T_r = route_ent_tensor.shape[2]
                if T_r >= 3:
                    route_ent_per_step = route_ent_tensor.mean(dim=(0, 1))  # (T,)
                    route_ent_early = route_ent_per_step[:T_r // 3].mean().item()
                    route_ent_mid = route_ent_per_step[T_r // 3:2 * T_r // 3].mean().item()
                    route_ent_late = route_ent_per_step[2 * T_r // 3:].mean().item()
            N = self.env_params.get('max_problem_size', 50)
            import math as _math
            max_ent_route = _math.log(max(N, 2))

            self._log_diagnostics(
                'ENTROPY',
                fleet_ent=fleet_ent_mean,
                route_ent=route_ent_mean,
                route_ent_early=route_ent_early,
                route_ent_mid=route_ent_mid,
                route_ent_late=route_ent_late,
                max_ent_route=max_ent_route,
                ent_ratio_pct=route_ent_mean / max_ent_route * 100 if max_ent_route > 0 else 0,
                N=N,
            )
        # 5. L_VF (Critic Loss – MSE)
        if len(v_pred_list) > 0:
            # Stack lại thành shape (B, T_fleet)
            v_pred_stack = torch.stack(v_pred_list, dim=1)

            # Đảm bảo target cũng có cùng kích thước
            v_target = returns_fleet.detach()

            # Nếu số bước fleet chênh lệch do ép action cũ, cắt cho bằng nhau
            T_v = min(v_pred_stack.shape[1], v_target.shape[1])
            v_pred_stack = v_pred_stack[:, :T_v]
            v_target = v_target[:, :T_v]

            L_VF = F.mse_loss(v_pred_stack, v_target)
        else:
            L_VF = encoded_nodes.new_tensor(0.0)


        # [PATCH 5] Log breakdown từng component để biết cái nào dominate
        with torch.no_grad():
            l_ppo_f = L_fleet_ppo.item()
            l_ppo_r = L_route_ppo.item()
            l_vf_raw = L_VF.item()
            l_vf_w = self.c_critic * l_vf_raw
            l_fut_raw = L_future.item()
            l_fut_w = self.lambda_future * l_fut_raw
            s_ent_raw = S_entropy.item()
            s_ent_w = self.alpha_entropy * s_ent_raw

            total_abs = abs(l_ppo_f) + abs(l_ppo_r) + abs(l_vf_w) + abs(l_fut_w) + abs(s_ent_w)
            if total_abs > 1e-9:
                ratio_ppo_f = abs(l_ppo_f) / total_abs * 100
                ratio_ppo_r = abs(l_ppo_r) / total_abs * 100
                ratio_vf = abs(l_vf_w) / total_abs * 100
                ratio_fut = abs(l_fut_w) / total_abs * 100
                ratio_ent = abs(s_ent_w) / total_abs * 100
            else:
                ratio_ppo_f = ratio_ppo_r = ratio_vf = ratio_fut = ratio_ent = 0

            self._log_diagnostics(
                'LOSS_COMP',
                L_PPO_fleet=l_ppo_f,
                L_PPO_route=l_ppo_r,
                L_VF_raw=l_vf_raw,
                L_VF_w=l_vf_w,
                L_future_raw=l_fut_raw,
                L_future_w=l_fut_w,
                S_entropy_raw=s_ent_raw,
                S_entropy_w=s_ent_w,
                ratio_ppo_f_pct=ratio_ppo_f,
                ratio_ppo_r_pct=ratio_ppo_r,
                ratio_vf_pct=ratio_vf,
                ratio_fut_pct=ratio_fut,
                ratio_ent_pct=ratio_ent,
            )
        # ─── Tổng hợp Loss ───
        L_total = (L_fleet_ppo
                   + L_route_ppo
                   + self.lambda_future * L_future
                   - self.alpha_entropy * S_entropy
                   + self.c_critic * L_VF)

        # ─── Backprop ───
        return L_total

    def _ppo_update(self,
                    rollout_data: dict,
                    advantages_fleet,
                    advantages_route,
                    returns_fleet) -> float:
        loss = self._compute_ppo_loss(
            rollout_data,
            advantages_fleet,
            advantages_route,
            returns_fleet,
            model=self.model,
            critic=self.critic,
            env=self.env,
        )
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self._sync_gradients()
        nn.utils.clip_grad_norm_(list(self.model.parameters()) + list(self.critic.parameters()), max_norm=1.0)
        self.optimizer.step()
        return loss.item()

    # -----------------------------------------------------------------------
    # Tiện ích
    # -----------------------------------------------------------------------

    @staticmethod
    def _stack_log_probs(probs_list, device=None):
        """Stack list of (B, pomo) thành (B, pomo, T), tính log."""
        if not probs_list:
            return torch.zeros(1, 1, 0, device=device)
        stacked = torch.stack(probs_list, dim=2)            # (B, pomo, T)
        return stacked.clamp(min=1e-12).log()

    @staticmethod
    def _normalize(tensor):
        """Chuẩn hóa zero-mean, unit-variance."""
        mean = tensor.mean()
        std = tensor.std() + 1e-8
        return (tensor - mean) / std
