import pandas as pd
from utils import *
from smac_rev import StarCraft2Env
from GDN import Agent
from functools import partial
import numpy as np
import sys
import os
import time
from cfg import get_cfg
cfg = get_cfg()
load_model = bool(os.environ.get("load_model", True))


vessl_on = cfg.vessl_on
if vessl_on == True:
    import vessl
    vessl.init()
    output_dir = "/output/"
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
else:
    from torch.utils.tensorboard import SummaryWriter
    output_dir = "output/"
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)




if sys.platform == "linux":
    def env_fn(env, **kwargs):
        return env(**kwargs)
    REGISTRY = {}
    REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
    os.environ.setdefault("SC2PATH", os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))
    env = REGISTRY["sc2"](map_name=cfg.map_name, seed=123, step_mul=8, replay_dir="Replays", )
else:
    env = StarCraft2Env(map_name = cfg.map_name)


map_name1 = cfg.map_name
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
    num_agent = env.get_env_info()["n_agents"]

    win_tags = list()
    for e in range(num_eval):
        env.reset()
        done = False
        episode_reward = 0
        step = 0
        action_history = torch.zeros([num_agent, env.get_env_info()["node_features"] + 5])
        while (not done) and (step < max_episode_len):
            step += 1

            node_feature, edge_index_enemy, edge_index_ally, _, dead_masking = env.get_heterogeneous_graph(heterogeneous = heterogenous)
            agent_feature = torch.concat([torch.tensor(node_feature)[:num_agent, :-1], action_history.to('cpu')], dim=1)

            avail_action = env.get_avail_actions()

            n_agent = len(avail_action)
            if cfg.given_edge == True:
                node_representation = agent.get_node_representation_temp(
                                                                     node_feature,
                                                                     agent_feature,
                                                                     edge_index_enemy,
                                                                     edge_index_ally,
                                                                     n_agent=n_agent,
                                                                     dead_masking=dead_masking,
                                                                     mini_batch=False)  # 차원 : n_agents X n_representation_comm
            else:
                node_representation, A, X = agent.get_node_representation_temp(
                                                                     node_feature,
                                                                     agent_feature,
                                                                     edge_index_enemy,
                                                                     edge_index_ally,
                                                                    n_agent=n_agent,
                                                                    dead_masking=dead_masking,
                                                                    mini_batch=False)

            action_feature = env.get_action_feature()  # 차원 : action_size X n_action_feature
            action, action_history = agent.sample_action(node_representation, action_feature, avail_action, epsilon=0)
            reward, done, info = env.step(action)
            win_tag = True if done and 'battle_won' in info and info['battle_won'] else False
            episode_reward += reward
            t += 1

        print("map name {} : Evaluation episode {}, episode reward {}, win_tag {}".format(env.map_name, e, episode_reward, np.mean(win_tags)))
        if win_tag == True:
            win_rates += 1 / num_eval
            win_tags.append(1)
        else:
            win_tags.append(0)
    print("map name : ", env.map_name, "승률", win_rates)
    return win_rates



def main():
    env.reset()
    num_unit_types, unit_type_ids = get_agent_type_of_envs([env])
    env.generate_num_unit_types(num_unit_types, unit_type_ids)
    hidden_size_obs = int(os.environ.get("hidden_size_obs", 84))#cfg.hidden_size_obs       # GAT 해당(action 및 node representation의 hidden_size)
    hidden_size_comm = int(os.environ.get("hidden_size_comm", 84))#cfg.hidden_size_comm
    hidden_size_action = int(os.environ.get("hidden_size_action", 84))  # cfg.hidden_size_comm
    hidden_size_Q = int(os.environ.get("hidden_size_Q",128)) #cfg.hidden_size_Q         # GAT 해당
    n_representation_obs = int(os.environ.get("n_representation_obs", 54))#cfg.n_representation_obs  # GAT 해당
    n_representation_action = int(os.environ.get("n_representation_action", 56))  # cfg.n_representation_comm
    n_representation_comm = int(os.environ.get("n_representation_comm", 48))#cfg.n_representation_comm
    graph_embedding = int(os.environ.get("graph_embedding", 56))
    graph_embedding_comm = int(os.environ.get("graph_embedding_comm", 84))
    buffer_size = int(os.environ.get("buffer_size", 100000))       # cfg.buffer_size
    batch_size = int(os.environ.get("batch_size", 24))             # cfg.batch_size
    gamma = 0.99                                                            # cfg.gamma
    learning_rate = float(os.environ.get("learning_rate", 5.0e-4))            # cfg.lr
    learning_rate_graph = float(os.environ.get("learning_rate_graph", 5e-4))  # cfg.lr
    num_episode = 500000 #cfg.num_episode
    train_start = int(os.environ.get("train_start", 10))# cfg.train_start
    epsilon = float(os.environ.get("epsilon", 0))#cfg.epsilon
    min_epsilon = float(os.environ.get("min_epsilon", 0)) #cfg.min_epsilon
    anneal_steps = int(os.environ.get("anneal_steps", 50000))#cfg.anneal_steps
    gamma1 = float(os.environ.get("gamma1", 0.0005))
    gamma2 = float(os.environ.get("gamma2", 0.0005))

    anneal_episodes_graph_variance =float(os.environ.get("anneal_episodes_graph_variance",float('inf')))
    min_graph_variance = float(os.environ.get("min_graph_variance", 0.01))

    anneal_epsilon = (epsilon - min_epsilon) / anneal_steps
    initializer = True


    agent = Agent(num_agent=env.get_env_info()["n_agents"],
                   num_enemy=env.get_env_info()["n_enemies"],
                   feature_size=env.get_env_info()["node_features"],
                   hidden_size_obs = hidden_size_obs,
                   hidden_size_comm = hidden_size_comm,
                   hidden_size_action =hidden_size_action,
                   hidden_size_Q = hidden_size_Q,

                   n_representation_obs = n_representation_obs,
                   n_representation_comm = n_representation_comm,
                   n_representation_action = n_representation_action,

                   graph_embedding = graph_embedding,
                   graph_embedding_comm = graph_embedding_comm,

                   buffer_size = buffer_size,
                   batch_size = batch_size,
                   learning_rate = learning_rate,
                   learning_rate_graph = learning_rate_graph,

                   gamma = gamma,
                   gamma1 = gamma1,
                   gamma2 = gamma2,
                   anneal_episodes_graph_variance = anneal_episodes_graph_variance,
                   min_graph_variance = min_graph_variance,
                   env = None
                  )
    agent.load_model("episode7376_t_305009_win_0.84375.pt")
    # 97000 그나마 나음
    # 82000
    # 83000
    t = 0
    epi_r = []
    win_rates = []
    lap_quad = []
    sec_eig = []
    rl_lo = []
    q_t = [] #
    for e in range(num_episode):
        if e % 10 == 1:
            if vessl_on == True:
                vessl.log(step = e, payload = {'reward' : np.mean(epi_r)})
                vessl.log(step = e, payload={'lap_quad': np.mean(lap_quad)})
                vessl.log(step = e, payload={'sec_eig': np.mean(sec_eig)})
                vessl.log(step = e, payload={'rl_lo': np.mean(rl_lo)})
                vessl.log(step = e, payload={'q_t': np.mean(q_t)})
                epi_r = []
                lap_quad = []
                sec_eig = []
                rl_lo = []
                q_t = []
                r_df= pd.DataFrame(epi_r)
                r_df.to_csv(output_dir+"cumulative_reward_map_name_{}__lr_{}_hiddensizeobs_{}_hiddensizeq_{}_nrepresentationobs_{}_nrepresentationcomm_{}.csv".format(map_name1,  learning_rate, hidden_size_obs, hidden_size_Q, n_representation_obs, n_representation_comm))
            else:
                r_df= pd.DataFrame(epi_r)
                r_df.to_csv(output_dir+"cumulative_reward_map_name_{}__lr_{}_hiddensizeobs_{}_hiddensizeq_{}_nrepresentationobs_{}_nrepresentationcomm_{}.csv".format(map_name1,  learning_rate, hidden_size_obs, hidden_size_Q, n_representation_obs, n_representation_comm))

        win_rate = evaluation(env, agent, 1000)
        win_rates.append(win_rate)
        if vessl_on == True:
            vessl.log(step = t, payload = {'win_rate' : win_rate})
            wr_df = pd.DataFrame(win_rates)
            wr_df.to_csv(output_dir+"win_rate_map_name_{}_lr_{}_hiddensizeobs_{}_hiddensizeq_{}_nrepresentationobs_{}_nrepresentationcomm_{}.csv".format(map_name1, learning_rate, hidden_size_obs, hidden_size_Q, n_representation_obs, n_representation_comm))
        else:
            wr_df = pd.DataFrame(win_rates)
            wr_df.to_csv("win_rate_map_name_{}_GNN_{}_lr_{}_hiddensizeobs_{}_hiddensizeq_{}_nrepresentationobs_{}.csv".format(map_name1, learning_rate, hidden_size_obs, hidden_size_Q, n_representation_obs, n_representation_comm))







main()