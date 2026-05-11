import torch
import os

def generate_data(data_size, min_problem_size, max_problem_size, min_agent_num, max_agent_num, seed):

    torch.manual_seed(seed)

    depot_xy = torch.rand(size=(data_size, 1, 2))
    # shape: (batch, 1, 2)

    problem_size = torch.randint(min_problem_size, max_problem_size + 1, size=(1, 1))[0][0]

    node_xy = torch.rand(size=(data_size, problem_size, 2))
    # shape: (batch, problem, 2)

    agent_num = torch.randint(min_agent_num, max_agent_num + 1, size=(1, 1))[0][0]

    agent_capacity = torch.rand(size=(data_size, agent_num)) * 2.5 + 0.5
    # agent_speed = torch.rand(size=(data_size, agent_num)) + 0.5
    mean_values = torch.rand(data_size) * 19 + 1
    fixed_cost_factor = torch.clamp(torch.rand(data_size, agent_num) * 2 - 1 + mean_values.unsqueeze(-1), 1)
    fixed_cost = fixed_cost_factor * agent_capacity
    variable_cost = torch.rand(size=(data_size, agent_num)) * 2 + 1

    demand_scaler = 100

    node_demand = torch.randint(1, 51, size=(data_size, problem_size)) / float(demand_scaler)
    # shape: (batch, problem)

    current_directory = os.path.dirname(os.path.abspath(__file__))

    PATH = current_directory + '/data/' + 'test_' + str(problem_size) + '_' + str(max_agent_num) + '_' + str(data_size) + '_' + str(seed) + '.pt'

    torch.save({
        'depot_xy': depot_xy,
        'node_xy': node_xy,
        'node_demand': node_demand,
        'agent_capacity': agent_capacity,
        'agent_fixed_cost': fixed_cost,
        'agent_variable_cost': variable_cost
    }, PATH)

    return depot_xy, node_xy, node_demand, agent_capacity, fixed_cost, variable_cost

if __name__ == '__main__':
    generate_data(100, 20, 20, 3, 6, 1234)
