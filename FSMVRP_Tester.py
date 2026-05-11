"""
FSMVRPTester_PPO.py
===================
Tester cho mô hình FSMVRP được huấn luyện bởi FSMVRPTrainer_PPO.

Các điểm tinh chỉnh so với FSMVRPTester gốc:
  1. Import đúng class môi trường FSMVRPSMDPEnv (thay vì FSMVRPEnv cũ).
  2. Vòng lặp suy luận PHÂN CẤP (hierarchical rollout):
       - Khi need_fleet_action → gọi model.forward_fleet()
       - Ngược lại           → gọi model.forward_route()
     (thay thế model(state) + env.step() không còn tồn tại trong env mới)
  3. Load checkpoint bao gồm cả critic_state_dict (tương thích trainer mới).
  4. Sửa deprecated torch.set_default_tensor_type → set_default_dtype/device.
  5. Sửa thứ tự load_problems() → reset() giống trainer mới (lấy đúng reset_state).
  6. Tính score từ env._get_episode_reward() thay vì reward cuối vòng lặp.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import csv
import os
import importlib.util
from logging import getLogger

from FSMVRP_Env import FSMVRPSMDPEnv as Env
from FSMVRP_Model import FSMVRPModel as Model

from utils import *


def _load_replay_module():
    """Load the standalone replay utility so tester integration stays optional."""

    try:
        # Prefer a normal import first because it keeps the implementation easy
        # to follow when the package path resolves cleanly.
        from Benchmarks.GoldenBC.scripts import replay_standard_cost as replay_module

        return replay_module
    except Exception:
        # Fall back to loading the script by filesystem path so the tester still
        # works even when namespace-package imports are unavailable.
        module_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'Benchmarks',
            'GoldenBC',
            'scripts',
            'replay_standard_cost.py',
        )
        spec = importlib.util.spec_from_file_location('goldenbc_replay_standard_cost', module_path)
        if spec is None or spec.loader is None:
            return None
        replay_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(replay_module)
        return replay_module


REPLAY_STANDARD_COST_MODULE = _load_replay_module()


# ---------------------------------------------------------------------------
# Mạng Critic – phải khai báo lại để load checkpoint (giống trong Trainer)
# ---------------------------------------------------------------------------

class CriticNetwork(nn.Module):
    """
    [FIX #2] Kiến trúc PHẢI GIỐNG HỆT Trainer để load checkpoint không lỗi.

    BẢN CŨ: Thiếu state_proj → load_state_dict(strict=True) sẽ báo missing keys.
    BẢN MỚI: Copy nguyên kiến trúc từ Trainer (state_proj, STATE_FEATURES=5).

    Trong quá trình test, mạng này KHÔNG được dùng để tính loss —
    chỉ load vào để checkpoint khớp hoàn toàn.
    """

    # [FIX #2] Phải giống Trainer: 5 state features
    STATE_FEATURES = 5

    def __init__(self, embedding_dim: int, hidden_dim: int = 256):
        super().__init__()
        # Nhánh không state (giữ nguyên)
        self.node_proj = nn.Linear(embedding_dim, hidden_dim)

        # [FIX #2] THÊM nhánh có state — THIẾU CÁI NÀY sẽ lỗi khi load checkpoint
        self.state_proj = nn.Linear(embedding_dim + self.STATE_FEATURES, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, encoded_nodes):
        # Tester chỉ cần nhánh đơn giản (không dùng Critic để ra quyết định)
        node_mean = encoded_nodes.squeeze(1).mean(dim=1)
        hidden = F.relu(self.node_proj(node_mean))
        return self.mlp(hidden).squeeze(-1)


# ---------------------------------------------------------------------------
# Tester chính
# ---------------------------------------------------------------------------

class FSMVRPTester_PPO:
    def __init__(self,
                 env_params,
                 model_params,
                 tester_params):

        # Lưu tham số
        self.env_params    = env_params
        self.model_params  = model_params
        self.tester_params = tester_params

        # Logger & thư mục kết quả
        self.logger        = getLogger(name='tester')
        self.result_folder = get_result_folder()

        # ── CUDA setup (dùng API hiện đại, tránh DeprecationWarning) ──
        USE_CUDA = self.tester_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.tester_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            self.device = torch.device('cuda', cuda_device_num)
        else:
            self.device = torch.device('cpu')

        torch.set_default_dtype(torch.float32)
        torch.set_default_device(self.device)

        # ── Khởi tạo môi trường & mô hình ──
        self.env   = Env(**self.env_params)
        self.model = Model(self.env_params, **self.model_params)

        # ── Khởi tạo Critic (chỉ để load checkpoint cho khớp kiến trúc) ──
        embedding_dim  = model_params['embedding_dim']
        critic_hidden  = tester_params.get('critic_hidden_dim', 256)
        self.critic    = CriticNetwork(embedding_dim, critic_hidden)

        # ── Load checkpoint ──
        model_load = tester_params['model_load']
        checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])

        # Load critic nếu có trong checkpoint (checkpoint từ trainer mới)
        if 'critic_state_dict' in checkpoint:
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.logger.info('Critic weights loaded from checkpoint.')
        else:
            self.logger.info('No critic_state_dict found – critic kept random (OK for inference).')

        self.logger.info('Model checkpoint loaded: epoch {}'.format(model_load['epoch']))

        # Tiện ích thời gian
        self.time_estimator = TimeEstimator()
        detail_cfg = tester_params.get('solution_detail', {})
        self.solution_detail_enable = detail_cfg.get('enable', False)
        self.solution_detail_max_episodes = detail_cfg.get('max_episodes', 3)
        self.solution_detail_compare_aug = detail_cfg.get('compare_augmented_best', True)
        self.solution_detail_skip_empty = detail_cfg.get('skip_empty_routes', True)
        csv_cfg = tester_params.get('csv_export', {})
        self.csv_export_enable = csv_cfg.get('enable', False)
        self.csv_summary_path = csv_cfg.get('summary_path', './solution_summary_demo.csv')
        self.csv_routes_path = csv_cfg.get('routes_path', './solution_routes_demo.csv')
        self.csv_summary_rows = []
        self.csv_route_rows = []

        # Keep replay settings separate from the normalized export so the user
        # can compare benchmark-standard costs without changing inference logic.
        replay_cfg = tester_params.get('standard_replay', {})
        test_filename = tester_params.get('test_data_load', {}).get('filename', '')
        self.source_pt_filename = test_filename if tester_params.get('test_data_load', {}).get('enable', False) else ''
        self.standard_replay_enable = replay_cfg.get('enable', self._looks_like_benchmark_file(self.source_pt_filename))
        self.standard_replay_benchmark_csv_root = replay_cfg.get(
            'benchmark_csv_root',
            './Benchmarks/GoldenBC/csv',
        )
        self.standard_replay_summary_path = replay_cfg.get(
            'summary_path',
            self._derive_standard_csv_path(self.csv_summary_path),
        )
        self.standard_replay_routes_path = replay_cfg.get(
            'routes_path',
            self._derive_standard_csv_path(self.csv_routes_path),
        )
        self.saved_problem_sequence = self._resolve_problem_sequence(self.source_pt_filename)

    # -----------------------------------------------------------------------
    # Vòng lặp test chính
    # -----------------------------------------------------------------------

    def run(self):
        self.time_estimator.reset()

        score_AM = AverageMeter()
        aug_score_AM = AverageMeter()
        penalized_score_AM = AverageMeter()
        penalized_aug_score_AM = AverageMeter()

        # Cấu hình nguồn dữ liệu
        if self.tester_params['test_data_load']['enable']:
            self.env.use_saved_problems(
                self.tester_params['test_data_load']['filename'],
                self.device
            )
        else:
            self.env.set_random_seed(
                self.tester_params['test_random_seed'],
                self.tester_params['test_episodes']
            )

        test_num_episode = self.tester_params['test_episodes']
        episode = 0
        self.env.random_list_index = 0

        while episode < test_num_episode:
            batch_size = 1  # Giống tester gốc: test từng instance

            score, aug_score, penalized_score, penalized_aug_score = self._test_one_batch(
                batch_size, episode_index=episode
            )

            if not self.tester_params['test_data_load']['enable']:
                self.env.random_list_index += 1

            score_AM.update(score, batch_size)
            aug_score_AM.update(aug_score, batch_size)
            penalized_score_AM.update(penalized_score, batch_size)
            penalized_aug_score_AM.update(penalized_aug_score, batch_size)
            episode += batch_size

            # ── Log tiến độ ──
            elapsed_str, remain_str = self.time_estimator.get_est_string(
                episode, test_num_episode)
            self.logger.info(
                "episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], "
                "raw_score:{:.3f}, raw_aug_score:{:.3f}, penalized_score:{:.3f}, penalized_aug_score:{:.3f}".format(
                    episode, test_num_episode,
                    elapsed_str, remain_str,
                    score, aug_score, penalized_score, penalized_aug_score
                )
            )

            if episode == test_num_episode:
                self.logger.info(" *** Test Done *** ")
                self.logger.info(" RAW NO-AUG SCORE:        {:.4f}".format(score_AM.avg))
                self.logger.info(" RAW AUGMENTATION SCORE:  {:.4f}".format(aug_score_AM.avg))
                self.logger.info(" PENALIZED NO-AUG SCORE:  {:.4f}".format(penalized_score_AM.avg))
                self.logger.info(" PENALIZED AUG SCORE:     {:.4f}".format(penalized_aug_score_AM.avg))
                self._write_csv_exports()
                print(f"avg raw no augment cost: {score_AM.avg:.4f}", flush=True)
                print(f"avg raw augment cost: {aug_score_AM.avg:.4f}", flush=True)
                print(f"avg penalized no augment cost: {penalized_score_AM.avg:.4f}", flush=True)
                print(f"avg penalized augment cost: {penalized_aug_score_AM.avg:.4f}", flush=True)
                return score_AM.avg, aug_score_AM.avg

    # -----------------------------------------------------------------------
    # Test một batch
    # -----------------------------------------------------------------------

    def _test_one_batch(self, batch_size: int, episode_index: int = 0):
        """
        Suy luận hoàn chỉnh một batch với vòng lặp phân cấp SMDP:
          - need_fleet_action == True  → model.forward_fleet()  (chọn xe)
          - need_fleet_action == False → model.forward_route()  (chọn điểm)

        Hỗ trợ data augmentation (8-fold symmetry) giống tester gốc.
        """

        # ── Benchmarks Augmentation ──
        aug_factor = (self.tester_params['aug_factor']
                      if self.tester_params['augmentation_enable']
                      else 1)

        # ── Reset môi trường ──
        self.model.eval()
        with torch.no_grad():
            # load_problems() trả về reset_state (thông tin tĩnh)
            reset_state, _, _ = self.env.load_problems(batch_size, aug_factor)
            # reset()         trả về step_state (trạng thái động ban đầu)
            self.env.reset()
            # pre_forward cần reset_state để mã hóa bản đồ
            self.model.pre_forward(reset_state)

            # ── Vòng lặp suy luận PHÂN CẤP ──
            state, _, done = self.env.pre_step()
            fleet_history = []
            fleet_mask_history = []
            route_history = []
            route_mask_history = []

            while not done:
                need_fleet = state.need_fleet_action  # (B*aug, pomo) bool

                if need_fleet.any():
                    # HIGH-LEVEL: Chọn loại xe cho tuyến mới
                    selected, _ = self.model.forward_fleet(state)
                    fleet_history.append(selected.detach().clone())
                    fleet_mask_history.append(need_fleet.detach().clone())
                    state, _, done = self.env.fleet_step(selected)
                else:
                    # LOW-LEVEL: Chọn điểm đến tiếp theo
                    selected, _ = self.model.forward_route(state)
                    route_history.append(selected.detach().clone())
                    route_mask_history.append((~need_fleet).detach().clone())
                    state, _, done = self.env.route_step(selected)

            # ── Tính phần thưởng cuối tập ──
            # _get_episode_reward() trả về phần thưởng âm (−cost), shape (B*aug, pomo)
            raw_cost = self.env.get_raw_total_cost()
            penalized_cost = self.env.get_total_cost()

        # ── Tính score (giống tester gốc) ──
        # Reshape về (aug_factor, batch_size, pomo_size)
        no_aug_score, aug_score = self._summarize_costs(raw_cost, aug_factor, batch_size)
        penalized_no_aug_score, penalized_aug_score = self._summarize_costs(
            penalized_cost, aug_factor, batch_size
        )

        # POMO best: lấy giá trị tốt nhất trong pomo_size trajectories

        # No-aug score: chỉ dùng bản gốc (aug index 0)

        # Aug score: lấy best trong tất cả aug_factor bản

        if self.solution_detail_enable and episode_index < self.solution_detail_max_episodes:
            self._log_solution_details(
                episode_index=episode_index,
                reset_state=reset_state,
                aug_factor=aug_factor,
                raw_cost=raw_cost,
                penalized_cost=penalized_cost,
                fleet_history=fleet_history,
                fleet_mask_history=fleet_mask_history,
                route_history=route_history,
                route_mask_history=route_mask_history,
            )
        elif self.csv_export_enable:
            self._collect_solution_rows(
                episode_index=episode_index,
                reset_state=reset_state,
                aug_factor=aug_factor,
                raw_cost=raw_cost,
                penalized_cost=penalized_cost,
                fleet_history=fleet_history,
                fleet_mask_history=fleet_mask_history,
                route_history=route_history,
                route_mask_history=route_mask_history,
            )

        return no_aug_score, aug_score, penalized_no_aug_score, penalized_aug_score

    @staticmethod
    def _summarize_costs(cost_tensor, aug_factor, batch_size):
        aug_cost = cost_tensor.reshape(aug_factor, batch_size, -1)
        best_pomo_cost = aug_cost.min(dim=2).values
        no_aug_score = best_pomo_cost[0, :].float().mean().item()
        aug_score = best_pomo_cost.min(dim=0).values.float().mean().item()
        return no_aug_score, aug_score

    def _log_solution_details(self, episode_index, reset_state, aug_factor, raw_cost, penalized_cost,
                              fleet_history, fleet_mask_history, route_history, route_mask_history):
        aug_raw_cost = raw_cost.reshape(aug_factor, 1, self.env.pomo_size)
        aug_penalized_cost = penalized_cost.reshape(aug_factor, 1, self.env.pomo_size)
        min_pomo_cost, best_pomo_idx = aug_raw_cost.min(dim=2)

        no_aug_pomo = int(best_pomo_idx[0, 0].item())
        no_aug_cost = float(min_pomo_cost[0, 0].item())
        no_aug_penalized_cost = float(aug_penalized_cost[0, 0, no_aug_pomo].item())
        self._emit_solution_report(self._format_solution_report(
            title=f"Episode {episode_index + 1} | No-Aug Best",
            solution=self._extract_solution(
                reset_state, fleet_history, fleet_mask_history, route_history, route_mask_history,
                aug_index=0, pomo_index=no_aug_pomo
            ),
            total_cost=no_aug_cost,
            penalized_cost=no_aug_penalized_cost,
        ))

        if self.solution_detail_compare_aug and aug_factor > 1:
            flat_raw_cost = aug_raw_cost[:, 0, :]
            flat_best = flat_raw_cost.argmin()
            best_aug_index = int((flat_best // self.env.pomo_size).item())
            best_aug_pomo = int((flat_best % self.env.pomo_size).item())
            best_aug_cost = float(flat_raw_cost[best_aug_index, best_aug_pomo].item())
            best_aug_penalized_cost = float(aug_penalized_cost[best_aug_index, 0, best_aug_pomo].item())
            self._emit_solution_report(self._format_solution_report(
                title=f"Episode {episode_index + 1} | Aug Best",
                solution=self._extract_solution(
                    reset_state, fleet_history, fleet_mask_history, route_history, route_mask_history,
                    aug_index=best_aug_index, pomo_index=best_aug_pomo
                ),
                total_cost=best_aug_cost,
                penalized_cost=best_aug_penalized_cost,
            ))
        if self.csv_export_enable:
            self._collect_solution_rows(
                episode_index, reset_state, aug_factor, raw_cost, penalized_cost,
                fleet_history, fleet_mask_history, route_history, route_mask_history
            )

    def _collect_solution_rows(self, episode_index, reset_state, aug_factor, raw_cost, penalized_cost,
                               fleet_history, fleet_mask_history, route_history, route_mask_history):
        aug_raw_cost = raw_cost.reshape(aug_factor, 1, self.env.pomo_size)
        aug_penalized_cost = penalized_cost.reshape(aug_factor, 1, self.env.pomo_size)
        min_pomo_cost, best_pomo_idx = aug_raw_cost.min(dim=2)

        no_aug_pomo = int(best_pomo_idx[0, 0].item())
        no_aug_cost = float(min_pomo_cost[0, 0].item())
        no_aug_solution = self._extract_solution(
            reset_state, fleet_history, fleet_mask_history, route_history, route_mask_history,
            aug_index=0, pomo_index=no_aug_pomo
        )
        self._append_csv_rows(
            episode_index,
            'no_aug',
            no_aug_cost,
            float(aug_penalized_cost[0, 0, no_aug_pomo].item()),
            no_aug_solution
        )

        flat_raw_cost = aug_raw_cost[:, 0, :]
        flat_best = flat_raw_cost.argmin()
        best_aug_index = int((flat_best // self.env.pomo_size).item())
        best_aug_pomo = int((flat_best % self.env.pomo_size).item())
        best_aug_cost = float(flat_raw_cost[best_aug_index, best_aug_pomo].item())
        best_aug_solution = self._extract_solution(
            reset_state, fleet_history, fleet_mask_history, route_history, route_mask_history,
            aug_index=best_aug_index, pomo_index=best_aug_pomo
        )
        self._append_csv_rows(
            episode_index,
            'aug',
            best_aug_cost,
            float(aug_penalized_cost[best_aug_index, 0, best_aug_pomo].item()),
            best_aug_solution
        )

    def _extract_solution(self, reset_state, fleet_history, fleet_mask_history, route_history,
                          route_mask_history, aug_index, pomo_index):
        depot_xy = reset_state.depot_xy[aug_index, 0].detach().cpu()
        node_xy = reset_state.node_xy[aug_index].detach().cpu()
        node_demand = reset_state.node_demand[aug_index].detach().cpu()
        agent_capacity = reset_state.agent_capacity[aug_index].detach().cpu()
        agent_fixed_cost = reset_state.agent_fixed_cost[aug_index].detach().cpu()
        agent_variable_cost = reset_state.agent_variable_cost[aug_index].detach().cpu()

        fleet_seq = [
            int(step[aug_index, pomo_index].item())
            for step, mask in zip(fleet_history, fleet_mask_history)
            if bool(mask[aug_index, pomo_index].item())
        ]
        route_seq = [
            int(step[aug_index, pomo_index].item())
            for step, mask in zip(route_history, route_mask_history)
            if bool(mask[aug_index, pomo_index].item())
        ]

        routes = []
        route_ptr = 0
        total_served_demand = 0.0
        customer_count = 0

        for vehicle_type in fleet_seq:
            served_nodes = []
            while route_ptr < len(route_seq):
                node = route_seq[route_ptr]
                route_ptr += 1
                if node == 0:
                    break
                served_nodes.append(node)

            node_indices = [node - 1 for node in served_nodes]
            delivered_demand = float(node_demand[node_indices].sum().item()) if node_indices else 0.0
            capacity = float(agent_capacity[vehicle_type].item())
            fixed_cost = float(agent_fixed_cost[vehicle_type].item())
            variable_cost = float(agent_variable_cost[vehicle_type].item())
            route_distance = self._compute_route_distance(depot_xy, node_xy, served_nodes)
            route_variable_cost = route_distance * variable_cost
            route_total_cost = fixed_cost + route_variable_cost
            load_ratio = delivered_demand / capacity if capacity > 0 else 0.0

            if self.solution_detail_skip_empty and delivered_demand <= 0:
                continue

            total_served_demand += delivered_demand
            customer_count += len(served_nodes)
            routes.append({
                'vehicle_type': vehicle_type,
                'capacity': capacity,
                'served_nodes': served_nodes,
                'delivered_demand': delivered_demand,
                'load_ratio': load_ratio,
                'route_distance': route_distance,
                'route_total_cost': route_total_cost,
            })

        total_demand = float(node_demand.sum().item())
        avg_load_ratio = sum(route['load_ratio'] for route in routes) / len(routes) if routes else 0.0
        return {
            'routes': routes,
            'route_count': len(routes),
            'customer_count': customer_count,
            'visited_demand': total_served_demand,
            'total_demand': total_demand,
            'avg_load_ratio': avg_load_ratio,
        }

    @staticmethod
    def _compute_route_distance(depot_xy, node_xy, served_nodes):
        if not served_nodes:
            return 0.0

        current_xy = depot_xy
        total_distance = 0.0
        for node in served_nodes:
            next_xy = node_xy[node - 1]
            total_distance += float(torch.norm(next_xy - current_xy, p=2).item())
            current_xy = next_xy
        total_distance += float(torch.norm(depot_xy - current_xy, p=2).item())
        return total_distance

    def _format_solution_report(self, title, solution, total_cost, penalized_cost):
        lines = [
            f"[Solution Detail] {title}",
            f"raw total cost: {total_cost:.4f}",
            f"penalized total cost: {penalized_cost:.4f}",
        ]

        for route_idx, route in enumerate(solution['routes'], start=1):
            lines.append(
                "  Route {:02d}: distance={:.4f}, cost={:.4f}, tai_trong_xe={:.2f}, tai_trong_hang={:.2f}, ratio={:.3f}".format(
                    route_idx,
                    route['route_distance'],
                    route['route_total_cost'],
                    route['capacity'],
                    route['delivered_demand'],
                    route['load_ratio'],
                )
            )

        return "\n".join(lines)

    def _emit_solution_report(self, report_text):
        print(report_text, flush=True)

    @staticmethod
    def _derive_standard_csv_path(path):
        """Create a sibling CSV path for benchmark-standard replay outputs."""

        root, ext = os.path.splitext(path)
        if root.endswith('_demo'):
            root = root[:-len('_demo')]
        return root + '_standard' + ext

    @staticmethod
    def _looks_like_benchmark_file(filename):
        """Detect benchmark-derived PT inputs from their path naming."""

        if not filename:
            return False
        lowered = filename.replace('\\', '/').lower()
        return '/benchmarks/' in lowered or 'goldenbc' in lowered or 'goldennnn' in lowered

    def _resolve_problem_sequence(self, filename):
        """Resolve benchmark problem ids for saved PT files when possible."""

        if not filename or REPLAY_STANDARD_COST_MODULE is None:
            return []

        try:
            return REPLAY_STANDARD_COST_MODULE.resolve_problem_sequence_from_pt_file(filename)
        except Exception as exc:
            # Keep inference robust even if replay metadata cannot be resolved,
            # and surface the issue through the tester log for debugging.
            self.logger.warning('Unable to resolve benchmark problem ids from %s: %s', filename, exc)
            return []

    def _benchmark_problem_id_for_episode(self, episode_index):
        """Map the current episode to its underlying benchmark problem id."""

        if not self.saved_problem_sequence:
            return ''
        if len(self.saved_problem_sequence) == 1:
            return str(self.saved_problem_sequence[0])
        if 0 <= episode_index < len(self.saved_problem_sequence):
            return str(self.saved_problem_sequence[episode_index])
        return ''

    def _append_csv_rows(self, episode_index, mode, total_cost, penalized_total_cost, solution):
        benchmark_problem_id = self._benchmark_problem_id_for_episode(episode_index)
        self.csv_summary_rows.append({
            'episode': episode_index + 1,
            'mode': mode,
            'benchmark_problem_id': benchmark_problem_id,
            'source_pt_filename': self.source_pt_filename,
            'total_cost': total_cost,
            'penalized_total_cost': penalized_total_cost,
            'route_count': solution['route_count'],
            'customer_count': solution['customer_count'],
            'total_demand': solution['total_demand'],
            'served_demand': solution['visited_demand'],
            'avg_load_ratio': solution['avg_load_ratio'],
        })
        for route_idx, route in enumerate(solution['routes'], start=1):
            self.csv_route_rows.append({
                'episode': episode_index + 1,
                'mode': mode,
                'benchmark_problem_id': benchmark_problem_id,
                'source_pt_filename': self.source_pt_filename,
                'route_index': route_idx,
                'vehicle_type': route['vehicle_type'],
                'distance': route['route_distance'],
                'cost': route['route_total_cost'],
                'tai_trong_xe': route['capacity'],
                'tai_trong_hang': route['delivered_demand'],
                'ratio': route['load_ratio'],
                'served_nodes': ' '.join(str(node) for node in route['served_nodes']),
            })

    def _write_csv_exports(self):
        if not self.csv_export_enable:
            return
        summary_dir = os.path.dirname(self.csv_summary_path)
        routes_dir = os.path.dirname(self.csv_routes_path)
        if summary_dir:
            os.makedirs(summary_dir, exist_ok=True)
        if routes_dir:
            os.makedirs(routes_dir, exist_ok=True)

        with open(self.csv_summary_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'episode', 'mode', 'benchmark_problem_id', 'source_pt_filename',
                'total_cost', 'penalized_total_cost', 'route_count', 'customer_count',
                'total_demand', 'served_demand', 'avg_load_ratio'
            ])
            writer.writeheader()
            writer.writerows(self.csv_summary_rows)

        with open(self.csv_routes_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'episode', 'mode', 'benchmark_problem_id', 'source_pt_filename',
                'route_index', 'vehicle_type', 'distance', 'cost',
                'tai_trong_xe', 'tai_trong_hang', 'ratio', 'served_nodes'
            ])
            writer.writeheader()
            writer.writerows(self.csv_route_rows)

        if self.standard_replay_enable and self.source_pt_filename and REPLAY_STANDARD_COST_MODULE is not None:
            try:
                outputs = REPLAY_STANDARD_COST_MODULE.replay_standard_cost_exports(
                    summary_csv_path=self.csv_summary_path,
                    routes_csv_path=self.csv_routes_path,
                    benchmark_csv_root=self.standard_replay_benchmark_csv_root,
                    source_pt_filename=self.source_pt_filename,
                    standard_summary_path=self.standard_replay_summary_path,
                    standard_routes_path=self.standard_replay_routes_path,
                )
                self.logger.info('Standard benchmark replay written: %s | %s', outputs.summary_path, outputs.routes_path)
            except Exception as exc:
                # Replay errors should not hide the normalized inference result,
                # so log them and keep the main tester run successful.
                self.logger.warning('Standard benchmark replay failed: %s', exc)


