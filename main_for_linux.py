import pandas as pd

from smac_rev import StarCraft2Env
from GDN import Agent
import torch
from pysc2.lib.remote_controller import ConnectError, RequestError
from pysc2.lib.protocol import ProtocolError
from functools import partial
import numpy as np
import sys
import os
import time




regularizer = 0.0
map_name1 = '6h_vs_8z'

GNN = 'FastGTN'
heterogenous = False

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

def evaluation(env, agent, num_eval):
    max_episode_len = env.episode_limit
    t = 0
    win_rates = 0
    for e in range(num_eval):
        env.reset()
        done = False
        episode_reward = 0
        step = 0
        while (not done) and (step < max_episode_len):
            step += 1

            node_feature, edge_index_enemy, edge_index_ally, n_node_features = env.get_heterogeneous_graph(heterogeneous = heterogenous)

            node_representation = agent.get_node_representation(node_feature, edge_index_enemy, edge_index_ally,
                                                                n_node_features,
                                                                mini_batch=False)  # 차원 : n_agents X n_representation_comm
            avail_action = env.get_avail_actions()
            action_feature = env.get_action_feature()  # 차원 : action_size X n_action_feature

            action = agent.sample_action(node_representation, action_feature, avail_action, epsilon=0)
            reward, done, info = env.step(action)
            win_tag = True if done and 'battle_won' in info and info['battle_won'] else False
            episode_reward += reward
            t += 1
        print("map name {} : Evaluation episode {}, episode reward {}, win_tag {}".format(env.map_name, e, episode_reward, win_tag))
        if win_tag == True:
            win_rates += 1 / num_eval
    print("map name : ", env.map_name, "승률", win_rates)
    return win_rates


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



def train(agent, env, e, t, train_start, epsilon, min_epsilon, anneal_epsilon, initializer):
    max_episode_limit = env.episode_limit
    if initializer == False:
        env.reset()
    done = False
    episode_reward = 0
    step = 0
    losses = []
    epi_r = list()
    eval = False
    start = time.time()
    while (not done) and (step < max_episode_limit):
        """
        Note: edge index 추출에 세가지 방법
        1. enemy_visibility에 대한 adjacency matrix 추출(self loop 포함) / 아군 유닛의 시야로부터 적에 대한 visibility relation
        2. ally_communication에 대한에 대한 adjacency matrix 추출                 / 아군 유닛의 시야로부터 적에 대한 visibility
        """
        node_feature, edge_index_enemy, edge_index_ally, n_node_features = env.get_heterogeneous_graph(heterogeneous=heterogenous)

        if GNN == 'GAT':
            node_representation = agent.get_node_representation(node_feature,
                                                                edge_index_enemy,
                                                                edge_index_ally,
                                                                n_node_features,
                                                                mini_batch=False)  # 차원 : n_agents X n_representation_comm
        if GNN == 'FastGTN':
            node_representation = agent.get_node_representation(node_feature,
                                                                edge_index_enemy,
                                                                edge_index_ally,
                                                                n_node_features,
                                                                mini_batch=False)  # 차원 : n_agents X n_representation_comm


        avail_action = env.get_avail_actions()
        action_feature = env.get_action_feature()  # 차원 : action_size X n_action_feature


        action = agent.sample_action(node_representation, action_feature, avail_action, epsilon)

        reward, done, info = env.step(action)
        agent.buffer.memory(node_feature, action, action_feature, edge_index_enemy, edge_index_ally, reward,
                            done, avail_action)


        episode_reward += reward
        t += 1
        step += 1
        if (t % 5000 == 0) and (t >0):
            eval = True

        if e >= train_start:
            loss = agent.learn(regularizer)
            losses.append(loss.detach().item())
        if epsilon >= min_epsilon:
            epsilon = epsilon - anneal_epsilon
        else:
            epsilon = min_epsilon


    if e >= train_start:

        print("{} Total reward in episode {} = {}, loss : {}, epsilon : {}, time_step : {}, episode_duration : {}".format(env.map_name,
                                                                                                e,
                                                                                                episode_reward,
                                                                                                loss,
                                                                                                epsilon,
                                                                                                t, start-time.time()))


    return episode_reward, epsilon, t, eval

def main():
    #writer = SummaryWriter('logs/')
    env1 = StarCraft2Env(map_name=map_name1, step_mul=8, replay_dir="Replays", seed=123)
    env1.reset()
    num_unit_types, unit_type_ids = get_agent_type_of_envs([env1])
    env1.generate_num_unit_types(num_unit_types, unit_type_ids)



    hidden_size_obs = 32       # GAT 해당(action 및 node representation의 hidden_size)
    hidden_size_comm = 60
    hidden_size_Q = 84         # GAT 해당
    hidden_size_meta_path = 42 # GAT 해당
    n_representation_obs = 36  # GAT 해당
    n_representation_comm = 69
    buffer_size = 150000
    batch_size = 32
    gamma = 0.99
    learning_rate = 1.3e-4
    n_multi_head = 1
    dropout = 0.6
    num_episode = 1000000
    train_start = 10
    epsilon = 1
    min_epsilon = 0.05
    anneal_steps = 50000
    teleport_probability = 0.9
    gtn_beta = 0.1

    anneal_epsilon = (epsilon - min_epsilon) / anneal_steps

    initializer = True
    agent1 = Agent(num_agent=env1.get_env_info()["n_agents"],
                   num_enemy=env1.get_env_info()["n_enemies"],
                   feature_size=env1.get_env_info()["node_features"],
                   hidden_size_meta_path = hidden_size_meta_path,
                   hidden_size_obs=hidden_size_obs,
                   hidden_size_comm=hidden_size_comm,
                   hidden_size_Q=hidden_size_Q,
                   n_multi_head=n_multi_head,
                   n_representation_obs=n_representation_obs,
                   n_representation_comm=n_representation_comm,
                   dropout=dropout,
                   action_size=env1.get_env_info()["n_actions"],
                   buffer_size=buffer_size,
                   batch_size=batch_size,
                   max_episode_len=env1.episode_limit,
                   learning_rate=learning_rate,
                   gamma=gamma,
                   GNN=GNN,
                   teleport_probability= teleport_probability,
                   gtn_beta = gtn_beta)

    # agent2 = Agent(num_agent=env2.get_env_info()["n_agents"],
    #               feature_size=env2.get_env_info()["node_features"],
    #               hidden_size_obs=hidden_size_obs,
    #               hidden_size_comm=hidden_size_comm,
    #               hidden_size_Q=hidden_size_Q,
    #               n_multi_head=n_multi_head,
    #               n_representation_obs=n_representation_obs,
    #               n_representation_comm=n_representation_comm,
    #               dropout=dropout,
    #               action_size=env2.get_env_info()["n_actions"],
    #               buffer_size=buffer_size,
    #               batch_size=batch_size,
    #               max_episode_len=env2.episode_limit,
    #               learning_rate=learning_rate,
    #               gamma=gamma)






    #network_sharing([agent1])
    t = 0
    epi_r = []
    win_rates = []
    for e in range(num_episode):
        episode_reward, epsilon, t, eval = train(agent1, env1, e, t, train_start, epsilon, min_epsilon, anneal_epsilon, initializer)
        initializer = False
        epi_r.append(episode_reward)
        if e % 100 == 1:

            r_df= pd.DataFrame(epi_r)
            r_df.to_csv("reward.csv")

        if eval == True:
            win_rate = evaluation(env1, agent1, 32)

            wr_df = pd.DataFrame(win_rates)
            wr_df.to_csv("win_rate.csv")







main()

