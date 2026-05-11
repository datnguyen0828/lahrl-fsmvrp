from dataclasses import dataclass
from typing import Optional
import torch

from problemdef import get_random_problems, augment_xy_data_by_8_fold


# ---------------------------------------------------------------------------
# Dataclasses for states
# ---------------------------------------------------------------------------

@dataclass
class Reset_State:
    """The state returned after a reset contains all the static information of the problem."""
    depot_xy: Optional[torch.Tensor] = None          # (batch, 1, 2)
    node_xy: Optional[torch.Tensor] = None           # (batch, problem, 2)
    node_demand: Optional[torch.Tensor] = None       # (batch, problem)
    agent_capacity: Optional[torch.Tensor] = None    # (batch, agent)
    agent_fixed_cost: Optional[torch.Tensor] = None  # (batch, agent)
    agent_variable_cost: Optional[torch.Tensor] = None  # (batch, agent)


@dataclass
class Step_State:
    """
    The SMDP state returned after each step clearly separates
    the high-level (fleet) and low-level (route) states.
    """
    # --- Index helpers ---
    BATCH_IDX: Optional[torch.Tensor] = None   # (batch, pomo)
    POMO_IDX: Optional[torch.Tensor] = None    # (batch, pomo)

    # --- General Info ---
    selected_count: int = 0
    route_count: Optional[torch.Tensor] = None        # (batch, pomo) - no. of completed routes
    graph_size: int = 0

    # --- High-level: Fleet state ---
    # The vehicle currently being used for the current route (-1 = not selected)
    current_vehicle_type: Optional[torch.Tensor] = None  # (batch, pomo)
    fleet_mask: Optional[torch.Tensor] = None         # (batch, pomo, agent)
    need_fleet_action: Optional[torch.Tensor] = None  # (batch, pomo)

    # --- Low-level: Route state ---
    current_node: Optional[torch.Tensor] = None       # (batch, pomo) - recent node (0 = depot)
    current_load: Optional[torch.Tensor] = None       # (batch, pomo) - remaining capacity of the current vehicle
    visited_mask: Optional[torch.Tensor] = None       # (batch, pomo, problem+1) bool – True = visited
    ninf_mask: Optional[torch.Tensor] = None          # (batch, pomo, problem+1) float – -inf if invalid

    # --- Tracking cost ---
    accumulated_cost: Optional[torch.Tensor] = None   # (batch, pomo)

    # --- Terminal ---
    finished: Optional[torch.Tensor] = None           # (batch, pomo)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class FSMVRPSMDPEnv:
    """
    SMDP environment for FSMVRP with hierarchical action space.
    Vòng lặp tương tác:
        reset_state, _, _ = env.load_problems(batch_size)
        step_state, _, _  = env.reset()
        step_state, _, _  = env.pre_step()

        while not done:
            if step_state.need_fleet_action.any():
                fleet_selected = agent.select_vehicle(step_state)   # (batch, pomo)
                step_state, reward, done = env.fleet_step(fleet_selected)
            else:
                node_selected  = agent.select_node(step_state)      # (batch, pomo)
                step_state, reward, done = env.route_step(node_selected)
    """

    def __init__(self, **env_params):
        # --- Hyper-params ---
        self.env_params = env_params
        self.device = env_params.get('device', None)
        self.min_problem_size = env_params['min_problem_size']
        self.max_problem_size = env_params['max_problem_size']
        self.min_agent_num = env_params['min_agent_num']
        self.max_agent_num = env_params['max_agent_num']
        self.pomo_size = env_params['pomo_size']
        penalty_cfg = env_params.get('utilization_penalty', {})
        self.util_penalty_enable = penalty_cfg.get('enable', False)
        self.util_penalty_ratio_threshold = penalty_cfg.get('ratio_threshold', 0.8)
        self.util_penalty_weight = penalty_cfg.get('weight', 0.0)
        self.util_penalty_power = penalty_cfg.get('power', 2.0)
        self.util_penalty_min_demand = penalty_cfg.get('min_demand', 0.0)

        # --- Flags for data source ---
        self.FLAG__use_saved_problems = False
        self.FLAG__use_random_seed = False
        self.saved_depot_xy = None
        self.saved_node_xy = None
        self.saved_node_demand = None
        self.saved_agent_capacity = None
        self.saved_agent_fixed_cost = None
        self.saved_agent_variable_cost = None
        self.saved_index = None

        # --- Problem dimensions (set at load_problems) ---
        self.batch_size = None
        self.problem_size = None
        self.agent_num = None

        # --- Static problem data ---
        self.depot_node_xy = None        # (batch, problem+1, 2)
        self.depot_node_demand = None    # (batch, problem+1)

        # --- Agent cost tensors ---
        self.agent_capacity = None       # (batch, agent)
        self.agent_fixed_cost = None     # (batch, agent)
        self.agent_variable_cost = None  # (batch, agent)

        # --- Index tensors ---
        self.BATCH_IDX = None            # (batch, pomo)
        self.POMO_IDX = None             # (batch, pomo)

        # --- State objects ---
        self.reset_state = Reset_State()
        self.step_state = Step_State()

        # --- Dynamic state variables ---
        self.selected_count = 0
        self.route_count = None          # (batch, pomo)
        self.current_node = None         # (batch, pomo)
        self.current_vehicle_type = None # (batch, pomo)  int index into agent dim
        self.current_load = None         # (batch, pomo)
        self.need_fleet_action = None    # (batch, pomo) bool
        self.visited_mask = None         # (batch, pomo, problem+1) bool
        self.accumulated_cost = None     # (batch, pomo)
        self.raw_accumulated_cost = None # (batch, pomo)
        self.finished = None             # (batch, pomo) bool
        self.current_route_capacity = None  # (batch, pomo)
        self.current_route_demand = None    # (batch, pomo)

        # Step reward (per-step, not episode)
        self._step_reward = None

    # -----------------------------------------------------------------------
    # Benchmarks loading helpers
    # -----------------------------------------------------------------------

    def use_saved_problems(self, filename: str, device):
        self.FLAG__use_saved_problems = True
        loaded_dict = torch.load(filename, map_location=device)
        self.saved_depot_xy = loaded_dict['depot_xy']
        self.saved_node_xy = loaded_dict['node_xy']
        self.saved_node_demand = loaded_dict['node_demand']
        self.saved_agent_capacity = loaded_dict['agent_capacity']
        self.saved_agent_fixed_cost = loaded_dict['agent_fixed_cost']
        self.saved_agent_variable_cost = loaded_dict['agent_variable_cost']
        self.saved_index = 0

    def set_random_seed(self, random_seed: int, test_num: int):
        self.FLAG__use_random_seed = True
        torch.manual_seed(random_seed)
        self.random_list = torch.randint(0, 100_000, size=(test_num,))
        self.random_list_index = 0
        torch.seed()

    # -----------------------------------------------------------------------
    # load_problems
    # -----------------------------------------------------------------------

    def load_problems(self, batch_size: int, aug_factor: int = 1):
        """Tải / tạo batch bài toán và khởi tạo các tensor tĩnh."""
        self.batch_size = batch_size

        # --- Lấy dữ liệu ---
        if not self.FLAG__use_saved_problems and not self.FLAG__use_random_seed:
            depot_xy, node_xy, node_demand, agent_capacity, agent_fixed_cost, agent_variable_cost = \
                get_random_problems(batch_size,
                                    self.min_problem_size, self.max_problem_size,
                                    self.min_agent_num, self.max_agent_num)
        elif self.FLAG__use_random_seed:
            seed = self.random_list[self.random_list_index].item()
            self.random_list_index += 1
            depot_xy, node_xy, node_demand, agent_capacity, agent_fixed_cost, agent_variable_cost = \
                get_random_problems(batch_size,
                                    self.min_problem_size, self.max_problem_size,
                                    self.min_agent_num, self.max_agent_num,
                                    random_seed=seed)
        else:
            s, e = self.saved_index, self.saved_index + batch_size
            depot_xy = self.saved_depot_xy[s:e]
            node_xy = self.saved_node_xy[s:e]
            node_demand = self.saved_node_demand[s:e]
            agent_capacity = self.saved_agent_capacity[s:e]
            agent_fixed_cost = self.saved_agent_fixed_cost[s:e]
            agent_variable_cost = self.saved_agent_variable_cost[s:e]
            self.saved_index += batch_size

        if self.device is not None:
            depot_xy = depot_xy.to(self.device)
            node_xy = node_xy.to(self.device)
            node_demand = node_demand.to(self.device)
            agent_capacity = agent_capacity.to(self.device)
            agent_fixed_cost = agent_fixed_cost.to(self.device)
            agent_variable_cost = agent_variable_cost.to(self.device)

        # --- Benchmarks augmentation (8-fold symmetry) ---
        if aug_factor > 1:
            if aug_factor == 8:
                depot_xy = augment_xy_data_by_8_fold(depot_xy)
                node_xy = augment_xy_data_by_8_fold(node_xy)
                node_demand = node_demand.repeat(8, 1)
                agent_capacity = agent_capacity.repeat(8, 1)
                agent_fixed_cost = agent_fixed_cost.repeat(8, 1)
                agent_variable_cost = agent_variable_cost.repeat(8, 1)
            else:
                raise NotImplementedError("Chỉ hỗ trợ aug_factor = 1 hoặc 8.")

        # --- Save the actual size after augmentation. ---
        self.batch_size = depot_xy.size(0)
        self.problem_size = node_xy.size(1)
        self.agent_num = agent_capacity.size(1)

        # --- Ghép depot vào đầu danh sách node ---
        self.depot_node_xy = torch.cat((depot_xy, node_xy), dim=1)
        # shape: (batch, problem+1, 2)
        depot_demand = torch.zeros(self.batch_size, 1, device=node_demand.device)
        self.depot_node_demand = torch.cat((depot_demand, node_demand), dim=1)
        # shape: (batch, problem+1)

        # --- Save cost tensors ---
        self.agent_capacity = agent_capacity          # (batch, agent)
        self.agent_fixed_cost = agent_fixed_cost      # (batch, agent)
        self.agent_variable_cost = agent_variable_cost  # (batch, agent)

        # --- Index helpers ---
        self.BATCH_IDX = torch.arange(self.batch_size, device=depot_xy.device)[:, None].expand(self.batch_size, self.pomo_size)
        # shape: (batch, pomo)
        self.POMO_IDX = torch.arange(self.pomo_size, device=depot_xy.device)[None, :].expand(self.batch_size, self.pomo_size)
        # shape: (batch, pomo)

        # --- Fill reset_state ---
        self.reset_state.depot_xy = depot_xy
        self.reset_state.node_xy = node_xy
        self.reset_state.node_demand = node_demand
        self.reset_state.agent_capacity = agent_capacity
        self.reset_state.agent_fixed_cost = agent_fixed_cost
        self.reset_state.agent_variable_cost = agent_variable_cost

        # Fill static infor to step_state
        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.POMO_IDX = self.POMO_IDX
        self.step_state.graph_size = self.problem_size + 1

        return self.reset_state, None, False

    # -----------------------------------------------------------------------
    # restore_problem  (used by PPO trainer to replay the same batch)
    # -----------------------------------------------------------------------

    def restore_problem(self, saved_problem: dict):
        """
        Restore a previously generated problem batch so that _ppo_update
        replays the EXACT same instances used during rollout.

        Args:
            saved_problem: dict with keys depot_xy, node_xy, node_demand,
                           agent_capacity, agent_fixed_cost, agent_variable_cost
                           – all tensors saved from a prior load_problems() call.
        """
        depot_xy            = saved_problem['depot_xy']
        node_xy             = saved_problem['node_xy']
        node_demand         = saved_problem['node_demand']
        agent_capacity      = saved_problem['agent_capacity']
        agent_fixed_cost    = saved_problem['agent_fixed_cost']
        agent_variable_cost = saved_problem['agent_variable_cost']

        self.batch_size   = depot_xy.size(0)
        self.problem_size = node_xy.size(1)
        self.agent_num    = agent_capacity.size(1)

        self.depot_node_xy = torch.cat((depot_xy, node_xy), dim=1)
        depot_demand = torch.zeros(self.batch_size, 1, device=depot_xy.device)
        self.depot_node_demand = torch.cat((depot_demand, node_demand), dim=1)
        self.agent_capacity      = agent_capacity
        self.agent_fixed_cost    = agent_fixed_cost
        self.agent_variable_cost = agent_variable_cost
        self.BATCH_IDX = torch.arange(self.batch_size, device=depot_xy.device)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX  = torch.arange(self.pomo_size, device=depot_xy.device)[None, :].expand(self.batch_size, self.pomo_size)
        self.reset_state.depot_xy            = depot_xy
        self.reset_state.node_xy             = node_xy
        self.reset_state.node_demand         = node_demand
        self.reset_state.agent_capacity      = agent_capacity
        self.reset_state.agent_fixed_cost    = agent_fixed_cost
        self.reset_state.agent_variable_cost = agent_variable_cost
        self.step_state.BATCH_IDX  = self.BATCH_IDX
        self.step_state.POMO_IDX   = self.POMO_IDX
        self.step_state.graph_size = self.problem_size + 1

    # -----------------------------------------------------------------------
    # reset
    # -----------------------------------------------------------------------

    def reset(self):
        """
        Re-initialize all dynamics.
        Immediately after the reset, all pomo are in the state requiring vehicle selection (need_fleet_action=True).
        """
        B, P = self.batch_size, self.pomo_size
        N1 = self.problem_size + 1  # số node gồm cả depot
        dev = self.depot_node_xy.device
        self.selected_count = 0
        self.route_count = torch.zeros(B, P, dtype=torch.long, device=dev)
        # shape: (batch, pomo)

        # All begin at depot (index 0)
        self.current_node = torch.zeros(B, P, dtype=torch.long, device=dev)
        # shape: (batch, pomo)

        # If not choose: =-1
        self.current_vehicle_type = torch.full((B, P), -1, dtype=torch.long, device=dev)
        # shape: (batch, pomo)
        # Remaining Capacity = 0 (B/c havent chosen vehicle yet)
        self.current_load = torch.zeros(B, P, device=dev)
        self.need_fleet_action = torch.ones(B, P, dtype=torch.bool, device=dev)
        self.visited_mask = torch.zeros(B, P, N1, dtype=torch.bool, device=dev)
        # shape: (batch, pomo, problem+1)
        self.visited_mask[:, :, 0] = True  # depot != customer
        self.accumulated_cost = torch.zeros(B, P, device=dev)
        # shape: (batch, pomo)
        self.raw_accumulated_cost = torch.zeros(B, P, device=dev)
        self.finished = torch.zeros(B, P, dtype=torch.bool, device=dev)
        # shape: (batch, pomo)
        self.current_route_capacity = torch.zeros(B, P, device=dev)
        self.current_route_demand = torch.zeros(B, P, device=dev)
        self._step_reward = torch.zeros(B, P, device=dev)
        self._sync_step_state()
        return self.step_state, None, False

    # -----------------------------------------------------------------------
    # pre_step  (optional – call before first action to get initial step_state)
    # -----------------------------------------------------------------------

    def pre_step(self):
        self._sync_step_state()
        return self.step_state, None, False

    # -----------------------------------------------------------------------
    # fleet_step  –  High-level action: chọn loại xe
    # -----------------------------------------------------------------------

    def fleet_step(self, vehicle_selected: torch.Tensor):
        """
        Fleet action processing (high-level).

        Args:
            vehicle_selected: (batch, pomo) – chỉ số loại xe k ∈ {0, ..., agent_num-1}
                              With need_fleet_action=False cells, the value is ignored.
        Returns:
            step_state, reward (per-step, None nếu chưa xong), done (bool)
        """
        assert self.need_fleet_action.any(), \
            "fleet_step was called but no pomo were waiting for fleet action."

        B, P = self.batch_size, self.pomo_size
        mask = self.need_fleet_action  # (batch, pomo)

        # --- Update vehicle type and capacity ---
        # Only update the cells that require fleet action
        self.current_vehicle_type = torch.where(mask, vehicle_selected, self.current_vehicle_type)
        # Chosen vehicle capacity: agent_capacity[batch, vehicle_selected]
        cap_all = self.agent_capacity  # (batch, agent)
        # agent_capacity: (batch, agent) → gather by agent dirextion
        # vehicle_selected: (batch, pomo) → cần (batch, pomo)
        new_load = cap_all[
            self.BATCH_IDX,                     # (batch, pomo)
            vehicle_selected.clamp(min=0)       # (batch, pomo)
        ]  # shape: (batch, pomo)

        self.current_load = torch.where(mask, new_load, self.current_load)
        self.current_route_capacity = torch.where(mask, new_load, self.current_route_capacity)
        self.current_route_demand = torch.where(mask, torch.zeros_like(self.current_route_demand), self.current_route_demand)

        # --- Tính chi phí cố định (I_new_route * F_k) ---
        # I_new_route = 1
        fixed_cost = self.agent_fixed_cost[
            self.BATCH_IDX,
            vehicle_selected.clamp(min=0)
        ]  # (batch, pomo)

        step_reward = -fixed_cost * mask.float()
        # shape: (batch, pomo) – negative result of fixed cost

        self.accumulated_cost -= step_reward  # accumulated_cost increases
        self.raw_accumulated_cost -= step_reward
        self._step_reward = step_reward

        # --- rRoute action ---

        self.need_fleet_action = torch.zeros(B, P, dtype=torch.bool, device=self.depot_node_xy.device)

        # Increase route_count
        self.route_count += mask.long()
        self._sync_step_state()
        done = self.finished.all().item()
        return self.step_state, step_reward if not done else self._get_episode_reward(), done

    # -----------------------------------------------------------------------
    # route_step  –  Low-level action: choose next node
    # -----------------------------------------------------------------------

    def route_step(self, node_selected: torch.Tensor):
        """
        Xử lý route action (low-level).
        Args:
            node_selected: (batch, pomo) – index node j ∈ {0, ..., problem}
                           node 0 = depot (return to depot = end recent route)
        Returns:
            step_state, reward (per-step), done (bool)
        """
        assert (~self.need_fleet_action).any(), "route_step is called, but all POMOs are waiting for fleet_step"
        B, P = self.batch_size, self.pomo_size
        N1 = self.problem_size + 1

        # --- Calculate delivery cost c_ij ---
        all_xy = self.depot_node_xy  # (batch, N1, 2)

        # current_node: (batch, pomo) → (batch, pomo, 1, 2) to gather
        last_xy = all_xy[
            self.BATCH_IDX,           # (batch, pomo)
            self.current_node         # (batch, pomo)
        ]  # (batch, pomo, 2)
        next_xy = all_xy[
            self.BATCH_IDX,
            node_selected
        ]  # (batch, pomo, 2)
        travel_cost = ((next_xy - last_xy) ** 2).sum(-1).sqrt()
        # shape: (batch, pomo)
        # Var cost (variable cost × distance)
        var_cost = self.agent_variable_cost[
            self.BATCH_IDX,
            self.current_vehicle_type.clamp(min=0)
        ]  # (batch, pomo)

        c_ij = travel_cost * var_cost  # (batch, pomo)

        # --- Step reward: R = -(c_ij)  (fixed cost, calculated in fleet_step) ---
        step_reward = -c_ij * (~self.finished).float()
        self._step_reward = step_reward
        self.accumulated_cost += c_ij * (~self.finished).float()
        self.raw_accumulated_cost += c_ij * (~self.finished).float()

        # --- Update visited mask ---
        # Mark the visited nodes (except depot index 0)
        is_customer = (node_selected != 0)  # (batch, pomo)
        # Update visited_mask for non-depot cells.
        self.visited_mask[
            self.BATCH_IDX[is_customer],
            self.POMO_IDX[is_customer],
            node_selected[is_customer]
        ] = True

        # --- Update remained capacity ---
        demand_all = self.depot_node_demand  # (batch, N1)
        selected_demand = demand_all[self.BATCH_IDX, node_selected]  # (batch, pomo)
        # Depot demand = 0, no affect
        self.current_load = self.current_load - selected_demand
        self.current_route_demand = self.current_route_demand + selected_demand * is_customer.float()
        # Secure nonnegative load
        self.current_load = self.current_load.clamp(min=0)

        # --- Update current_node ---
        self.current_node = node_selected
        self.selected_count += 1

        # --- Determine if further fleet action is needed. ---
        returning_to_depot = (node_selected == 0)  # (batch, pomo)
        route_penalty = self._compute_utilization_penalty(returning_to_depot & (~self.finished))
        if route_penalty is not None:
            step_reward = step_reward - route_penalty
            self._step_reward = step_reward
            self.accumulated_cost += route_penalty

        # Check all customers who have been visited
        all_visited = self.visited_mask[:, :, 1:].all(dim=-1)  # (batch, pomo)

        # If you return to the depot and not finished → need to choose a new vehicle for the next leg of the journey.
        # If you return to the depot and everything is finished → finished.
        new_finished = returning_to_depot & all_visited & (~self.finished)
        self.finished = self.finished | new_finished

        # Fleet action is required when returning to the depot and before finishing.
        self.need_fleet_action = returning_to_depot & (~self.finished)
        self.current_route_capacity = torch.where(returning_to_depot, torch.zeros_like(self.current_route_capacity), self.current_route_capacity)
        self.current_route_demand = torch.where(returning_to_depot, torch.zeros_like(self.current_route_demand), self.current_route_demand)


        self._sync_step_state()

        done = self.finished.all().item()
        if done:
            final_reward = self._get_episode_reward()
            return self.step_state, final_reward, True

        return self.step_state, step_reward, False

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _build_ninf_mask(self) -> torch.Tensor:
        """
        Build ninf_mask for route action.

        Rules:
          - Visited Node (visited_mask=True) → -inf
          - Depot (index 0) can choose anytime ̣to end route)
          - Nếu demand > current_load → -inf (insufficient load)
          - If need_fleet_action → mask all
        """
        B, P = self.batch_size, self.pomo_size
        N1 = self.problem_size + 1
        ninf_mask = torch.zeros(B, P, N1, device=self.depot_node_xy.device)
        ninf_mask[self.visited_mask] = float('-inf')
        ninf_mask[:, :, 0] = 0.0

        # Preventing "Empty Routes":
        # If the vehicle is at the depot (just started) AND the episode hasn't finished -> Do not re-select the depot.
        # Be careful to exclude finished POMOs, otherwise the entire episode will be -inf-infused (causing an argmax error).
        just_started = (self.current_node == 0) & (~self.finished)  # (batch, pomo)
        ninf_mask[self.BATCH_IDX[just_started], self.POMO_IDX[just_started], 0] = float('-inf')

        # Đánh -inf khi demand vượt quá tải còn lại
        demand_all = self.depot_node_demand[:, None, :].expand(B, P, N1)  # (batch, pomo, N1)
        load_expand = self.current_load[:, :, None].expand(B, P, N1)     # (batch, pomo, N1)
        round_eps = 1e-5
        exceeds_capacity = (demand_all > load_expand + round_eps)
        ninf_mask[exceeds_capacity] = float('-inf')

        # Depot is not limited by capacity
        ninf_mask[:, :, 0] = torch.where(just_started, float('-inf'), 0.0)

        # If need fleet action → all route action masked
        need_fleet_expand = self.need_fleet_action[:, :, None].expand(B, P, N1)
        ninf_mask[need_fleet_expand] = float('-inf')

        return ninf_mask

    def _build_fleet_mask(self) -> torch.Tensor:
        """
        Xây dựng fleet_mask cho fleet action.
        Loại xe k hợp lệ khi tồn tại ít nhất 1 khách hàng chưa thăm
        có demand ≤ capacity[k].
        """
        B, P = self.batch_size, self.pomo_size
        A = self.agent_num

        # Demand các node chưa thăm (chỉ xét node khách hàng, không xét depot)
        # visited_mask: (batch, pomo, N1)
        customer_visited = self.visited_mask[:, :, 1:]  # (batch, pomo, problem)
        customer_demand = self.depot_node_demand[:, 1:]  # (batch, problem)

        # Với mỗi loại xe k, capacity[k]: (batch,)
        cap = self.agent_capacity  # (batch, agent)

        # Min demand trong các node chưa thăm
        # Nếu không còn node nào → loại xe nào cũng không cần → cho phép tất cả (sẽ không cần fleet action)
        demand_expand = customer_demand[:, None, :].expand(B, P, self.problem_size)
        # Đặt demand của node đã thăm = inf để không ảnh hưởng min
        masked_demand = demand_expand.clone()
        masked_demand[customer_visited] = float('inf')
        min_demand, _ = masked_demand.min(dim=-1)  # (batch, pomo)

        # Xe k hợp lệ khi cap[k] >= min_demand
        cap_expand = cap[:, None, :].expand(B, P, A)        # (batch, pomo, agent)
        min_demand_expand = min_demand[:, :, None].expand(B, P, A)  # (batch, pomo, agent)

        fleet_valid = (cap_expand >= min_demand_expand)  # (batch, pomo, agent)

        # Chuyển thành float mask: 0 = hợp lệ, -inf = không hợp lệ
        # [FIX #8b] BẢN CŨ: torch.zeros(B, P, A) → luôn trên CPU
        fleet_mask = torch.zeros(B, P, A, device=self.depot_node_xy.device)
        fleet_mask[~fleet_valid] = float('-inf')

        # Nếu không cần fleet action → mask toàn bộ xe thành -inf
        # rồi mở khóa xe 0 làm dummy action (tránh softmax NaN)
        not_needed = ~self.need_fleet_action  # (B, P)
        not_needed_3d = not_needed[:, :, None].expand(B, P, A)  # (B, P, A)

        # Bước 1: mask toàn bộ xe = -inf cho các POMO không cần fleet action
        fleet_mask = torch.where(not_needed_3d, torch.tensor(float('-inf'), device=fleet_mask.device), fleet_mask)

        # Bước 2: mở khóa xe 0 (dummy) cho các POMO không cần fleet action
        # Tạo mask chỉ cho cột 0: (B, P, A) với chỉ cột 0 = True
        dummy_mask = torch.zeros(B, P, A, dtype=torch.bool, device=fleet_mask.device)
        dummy_mask[:, :, 0] = True
        restore = not_needed_3d & dummy_mask  # (B, P, A) — True chỉ ở [b,p,0] khi not_needed
        fleet_mask = torch.where(restore, torch.tensor(0.0, device=fleet_mask.device), fleet_mask)

        return fleet_mask

    def _sync_step_state(self):
        """Cập nhật step_state từ các biến động hiện tại."""
        self.step_state.selected_count = self.selected_count
        self.step_state.route_count = self.route_count
        self.step_state.current_vehicle_type = self.current_vehicle_type
        self.step_state.current_node = self.current_node
        self.step_state.current_load = self.current_load
        self.step_state.need_fleet_action = self.need_fleet_action
        self.step_state.visited_mask = self.visited_mask
        self.step_state.ninf_mask = self._build_ninf_mask()
        self.step_state.fleet_mask = self._build_fleet_mask()
        self.step_state.accumulated_cost = self.accumulated_cost
        self.step_state.finished = self.finished

    def _compute_utilization_penalty(self, route_finished_mask: torch.Tensor):
        if (not self.util_penalty_enable) or self.util_penalty_weight <= 0:
            return None

        active_mask = route_finished_mask & (self.current_route_capacity > 1e-8)
        active_mask = active_mask & (self.current_route_demand > self.util_penalty_min_demand)
        if not active_mask.any():
            return None

        route_ratio = self.current_route_demand / self.current_route_capacity.clamp(min=1e-8)
        deficit = (self.util_penalty_ratio_threshold - route_ratio).clamp(min=0.0)
        if self.util_penalty_power != 1.0:
            deficit = deficit.pow(self.util_penalty_power)

        current_fixed_cost = self.agent_fixed_cost[
            self.BATCH_IDX,
            self.current_vehicle_type.clamp(min=0)
        ]
        penalty = self.util_penalty_weight * current_fixed_cost * deficit
        return penalty * active_mask.float()

    def _get_episode_reward(self) -> torch.Tensor:
        """
        Tính phần thưởng cuối episode = âm tổng chi phí.
        Gọi khi done=True.
        """
        return -self.accumulated_cost  # (batch, pomo)

    def get_total_cost(self) -> torch.Tensor:
        """Trả về tổng chi phí tích lũy (batch, pomo)."""
        return self.accumulated_cost.clone()

    def get_raw_total_cost(self) -> torch.Tensor:
        return self.raw_accumulated_cost.clone()


# ---------------------------------------------------------------------------
# Quick sanity-check
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=== FSMVRPSMDPEnv – Sanity Check ===\n")

    env_params = {
        'min_problem_size': 5,
        'max_problem_size': 10,
        'min_agent_num': 2,
        'max_agent_num': 4,
        'pomo_size': 3,
    }

    env = FSMVRPSMDPEnv(**env_params)
    reset_state, _, _ = env.load_problems(batch_size=2)
    _, _, _ = env.reset()          # reset state
    step_state, _, _ = env.pre_step()  # initial

    print(f"Batch size    : {env.batch_size}")
    print(f"Problem size  : {env.problem_size}")
    print(f"Agent num     : {env.agent_num}")
    print(f"POMO size     : {env.pomo_size}")
    print(f"depot_node_xy : {env.depot_node_xy.shape}")
    print(f"need_fleet_action (init): {step_state.need_fleet_action}")
    print()

    step = 0
    done = False
    while not done:
        if step_state.need_fleet_action.any():
            # Pick valid vehicle
            B, P, A = env.batch_size, env.pomo_size, env.agent_num
            fleet_mask = step_state.fleet_mask  # (B, P, A)
            # Greedy: choose best valid
            valid = (fleet_mask > float('-inf'))   # (B, P, A)
            logits = fleet_mask.clone()
            logits[~valid] = -1e9
            vehicle_sel = logits.argmax(dim=-1)    # (B, P)
            step_state, reward, done = env.fleet_step(vehicle_sel)
            print(f"[Step {step:3d}] FLEET action → xe {vehicle_sel[0].tolist()} | reward={reward[0].tolist() if reward is not None else None}")
        else:
            ninf_mask = step_state.ninf_mask     # (B, P, N1)
            logits = torch.zeros_like(ninf_mask) + ninf_mask
            rand_scores = torch.rand_like(logits)
            rand_scores[ninf_mask == float('-inf')] = -1e9
            node_sel = rand_scores.argmax(dim=-1)  # (B, P)
            step_state, reward, done = env.route_step(node_sel)
            print(f"[Step {step:3d}] ROUTE action → node {node_sel[0].tolist()} | reward={reward[0].tolist() if reward is not None else None} | done={done}")
        step += 1
        if step > 500:
            print("Exceed the maximum step size, stop..")
            break

    print(f"\nTổng chi phí (batch 0): {env.get_total_cost()[0].tolist()}")
    print("=== Hoàn thành ===")
