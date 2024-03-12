import torch
import numpy as np
"""
Protoss
colossi : 200.0150.01.0
stalkers : 80.080.00.625
zealots : 100.050.00.5

Terran
medivacs  : 150.00.00.75
marauders : 125.00.00.5625
marines   : 45.00.00.375

Zerg
zergling : 35.00.00.375
hydralisk : 80.00.00.625
baneling : 30.00.00.375
spine crawler : 300.00.01.125`
"""
def get_graph_loss(X, A, num_nodes):
    X_i = X.unsqueeze(2)
    X_j = X.unsqueeze(1)
    euclidean_distance = torch.sum((X_i - X_j) ** 2, dim=3).detach()
    laplacian_quadratic = torch.sum(euclidean_distance * A, dim=(1, 2))
    frobenius_norm = (torch.norm(A, p='fro', dim=(1, 2), keepdim=True) ** 2).squeeze(-1).squeeze(-1)
    var = torch.mean(torch.var(A, dim=2), dim=1)

    D = torch.zeros_like(A)
    for i in range(A.size(0)):
        D[i] = torch.diag(A[i].sum(1))

    L = D-A
    sec_eig = np.mean([np.linalg.eigh(L[G, :, :].detach().cpu().numpy())[0][1] for G in range(L.shape[0])])


    lap_quad = laplacian_quadratic.mean()
    sec_eig_upperbound = (num_nodes / (num_nodes - 1)) ** 2 * (frobenius_norm - num_nodes ** 2 * var).mean()
    #print(sec_eig_upperbound)
    return lap_quad, sec_eig_upperbound

def get_agent_type_of_envs(envs):
    agent_type_ids = list()
    type_alliance = list()
    for env in envs:
        for agent_id, _ in env.agents.items():
            agent = env.get_unit_by_id(agent_id)
            agent_type_ids.append(str(agent.health_max)+str(agent.shield_max)+str(agent.radius))
            type_alliance.append([str(agent.health_max)+str(agent.shield_max)+str(agent.radius), agent.alliance])
        for e_id, e_unit in env.enemies.items():
            enemy = list(env.enemies.items())[e_id][1]
            agent_type_ids.append(str(enemy.health_max)+str(enemy.shield_max)+str(enemy.radius))
            type_alliance.append([str(enemy.health_max)+str(enemy.shield_max)+str(enemy.radius), enemy.alliance])
    agent_types_list = list(set(agent_type_ids))
    type_alliance_set = list()
    for x in type_alliance:
        if x not in type_alliance_set:
            type_alliance_set.append(x)
    print(type_alliance_set)
    for id in agent_types_list:
        print("id : ", id, "count : " , agent_type_ids.count(id))

    return len(agent_types_list), agent_types_list