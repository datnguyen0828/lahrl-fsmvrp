import torch
import pandas as pd
import matplotlib.pyplot as plt
import os

def get_random_problems(batch_size, min_problem_size, max_problem_size, min_agent_num, max_agent_num, random_seed=None):

    if random_seed != None:
        torch.manual_seed(random_seed)

    depot_xy = torch.rand(size=(batch_size, 1, 2))
    # shape: (batch, 1, 2)

    problem_size = torch.randint(min_problem_size, max_problem_size + 1, size=(1, 1))[0][0]

    node_xy = torch.rand(size=(batch_size, problem_size, 2))
    # shape: (batch, problem, 2)

    agent_num = torch.randint(min_agent_num, max_agent_num + 1, size=(1, 1))[0][0]

    agent_capacity = torch.rand(size=(batch_size, agent_num)) * 2.5 + 0.5
    # agent_speed = torch.rand(size=(data_size, agent_num)) + 0.5
    mean_values = torch.rand(batch_size) * 19 + 1
    fixed_cost_factor = torch.clamp(torch.rand(batch_size, agent_num) * 2 - 1 + mean_values.unsqueeze(-1), 1)
    fixed_cost = fixed_cost_factor * agent_capacity
    variable_cost = torch.rand(size=(batch_size, agent_num)) * 2 + 1

    demand_scaler = 100

    node_demand = torch.randint(1, 51, size=(batch_size, problem_size)) / float(demand_scaler)
    # shape: (batch, problem)

    return depot_xy, node_xy, node_demand, agent_capacity, fixed_cost, variable_cost


def augment_xy_data_by_8_fold(xy_data):
    # xy_data.shape: (batch, N, 2)

    x = xy_data[:, :, [0]]
    y = xy_data[:, :, [1]]
    # x,y shape: (batch, N, 1)

    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1 - x, y), dim=2)
    dat3 = torch.cat((x, 1 - y), dim=2)
    dat4 = torch.cat((1 - x, 1 - y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1 - y, x), dim=2)
    dat7 = torch.cat((y, 1 - x), dim=2)
    dat8 = torch.cat((1 - y, 1 - x), dim=2)

    aug_xy_data = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
    # shape: (8*batch, N, 2)

    return aug_xy_data

if __name__ == '__main__':
    get_random_problems(10, 50, 80, 3, 8)