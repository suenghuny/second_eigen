import pandas as pd
import argparse
import os

from smac_rev import StarCraft2Env
from MAGPO import Agent
from functools import partial
import numpy as np
import sys
import os
import time
from utils import *
from cfg import get_cfg
cfg = get_cfg()

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
    output_dir = "/output/"
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)



if sys.platform == "linux":
    def env_fn(env, **kwargs):
        return env(**kwargs)
    REGISTRY = {}
    REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
    os.environ.setdefault("SC2PATH",
                          os.path.join(os.getcwd(),
                                       "3rdparty",
                                       "StarCraftII"))
else:
    env = StarCraft2Env(map_name = cfg.map_name)

regularizer = 0.0
map_name1 = cfg.map_name
heterogenous = False


def evaluation(env, agent):
    max_episode_len = env.episode_limit
    t = 0
    win_rates = 0
    for e in range(32):
        env.reset()
        done = False
        episode_reward = 0
        step = 0
        num_agent = env.get_env_info()["n_agents"]
        action_history = torch.zeros([num_agent, env.get_env_info()["node_features"] + 6])
        while (not done) and (step < max_episode_len):
            node_feature, edge_index_enemy, edge_index_comm, _, dead_masking = env.get_heterogeneous_graph(heterogeneous=heterogenous)
            agent_feature = torch.concat([torch.tensor(node_feature)[:num_agent, :-1], action_history.to('cpu')], dim=1)
            avail_action = env.get_avail_actions()
            n_agent = len(avail_action)
            if cfg.given_edge == True:
                node_embedding = agent.get_node_representation_gpo(node_feature, agent_feature, edge_index_enemy, edge_index_comm, n_agent = n_agent, dead_masking = dead_masking)
            else:
                node_embedding= agent.get_node_representation_gpo(node_feature, agent_feature, edge_index_enemy, edge_index_comm, n_agent = n_agent, dead_masking = dead_masking)

            action_feature = env.get_action_feature()  # 차원 : action_size X n_action_feature
            summarized_state = env.get_summarized_state()
            agent.eval_check(eval=True)
            action, prob,_, action_history = agent.sample_action(summarized_state, node_embedding, action_feature, avail_action, num_agent=env.get_env_info()["n_agents"],
                                                                 edge_index_obs = edge_index_enemy)
            reward, done, info = env.step(action)

            episode_reward += reward
            win_tag = True if done and 'battle_won' in info and info['battle_won'] else False
            t += 1
            step += 1
        print("map name {} : Evaluation episode {}, episode reward {}, win_tag {}".format(env.map_name, e, episode_reward, win_tag))
        if win_tag == True:
            win_rates += 1 / 32
    print("map name : ", env.map_name, "승률", win_rates)
    return win_rates







def train(agent, env, e, t, monitor, params, current_epsilon):
    max_episode_limit = env.episode_limit
    epsilon = params['epsilon']
    min_epsilon = params['min_epsilon']
    anneal_steps = params['anneal_step']
    anneal_epsilon = (epsilon - min_epsilon) / anneal_steps
    env.reset()
    done = False
    episode_reward = 0
    step = 0
    eval = False
    start = time.time()
    num_agent = env.get_env_info()["n_agents"]
    action_history = torch.zeros([num_agent, env.get_env_info()["node_features"] + 6])
    while (not done) and (step < max_episode_limit):
        node_feature, edge_index_enemy, edge_index_comm, _, dead_masking = env.get_heterogeneous_graph(heterogeneous=heterogenous)
        agent_feature = torch.concat([torch.tensor(node_feature)[:num_agent, :-1], action_history.to('cpu')], dim=1)
        avail_action = env.get_avail_actions()
        state = env.get_state()
        n_agent = len(avail_action)


        if cfg.given_edge == True:
            node_embedding = agent.get_node_representation_gpo(node_feature, agent_feature, edge_index_enemy, edge_index_comm, n_agent = n_agent, dead_masking = dead_masking)
        else:
            node_embedding = agent.get_node_representation_gpo(node_feature, agent_feature, edge_index_enemy, edge_index_comm, n_agent = n_agent, dead_masking = dead_masking)

        summarized_state = env.get_summarized_state()
        action_feature = env.get_action_feature()  # 차원 : action_size X n_action_feature



        agent.eval_check(eval=True)
        action, prob, factorized_probs, action_history = agent.sample_action(summarized_state, node_embedding, action_feature, avail_action, num_agent = env.get_env_info()["n_agents"],
                                                                             epsilon = 0, edge_index_obs=edge_index_enemy)
        reward, done, info = env.step(action)

        transition = (node_feature,
                      edge_index_enemy,
                      avail_action,
                      action,
                      prob,
                      action_feature,
                      reward,
                      done,
                      edge_index_comm,
                      factorized_probs,
                      dead_masking,
                      state,
                      summarized_state,
                      agent_feature,
                      )
        agent.put_data(transition)
        episode_reward += reward
        t += 1
        step += 1

        if current_epsilon >= min_epsilon:
            current_epsilon = current_epsilon - anneal_epsilon
        else:
            current_epsilon = min_epsilon

        if (t % 5000 == 0) and (t > 0):
            eval = True
    if cfg.vessl_on == True:
        vessl.log(step=e, payload={'episode_reward': episode_reward})
    if (e % params["n_data_parallelism"] == 0) and (e > 0):
        cum_surr, cum_value_loss, cum_lap_quad, cum_sec_eig_upperbound= agent.learn()
        monitor.append((e, cum_surr, cum_value_loss, cum_lap_quad, cum_sec_eig_upperbound))
        df = pd.DataFrame(monitor)
        if cfg.vessl_on == True:
            df.to_csv("/output/df.csv")
            #.log(step = e, payload={'fiedler': second_eigenvalue})
            vessl.log(step = e, payload = {'laplacian quadractic': cum_lap_quad})
            vessl.log(step = e, payload = {'ub': cum_sec_eig_upperbound})
        else:
            df.to_csv("df.csv")



    print("{} Total reward in episode {} = {}, time_step : {}, episode_duration : {}, current_epsilon : {}".format(env.map_name,
                                                                                                e,
                                                                                                np.round(episode_reward, 3),
                                                                                                t, np.round(time.time()-start, 3),
                                                                                                                   current_epsilon))
    return episode_reward, t, eval, current_epsilon

def main():
    if sys.platform == "linux":
        env = REGISTRY["sc2"](map_name=cfg.map_name, seed=123, step_mul=8, replay_dir="Replays", )
    else:
        env = StarCraft2Env(map_name=cfg.map_name)
    env.reset()
    num_unit_types, unit_type_ids = get_agent_type_of_envs([env])
    env.generate_num_unit_types(num_unit_types, unit_type_ids)
    if cfg.sweep == True:
        params = {
            "epsilon": float(os.environ.get("epsilon", 1)),
            "anneal_step": int(os.environ.get("anneal_step", 50000)),
            "min_epsilon": float(os.environ.get("min_epsilon", 0.05)),
            "hidden_size_obs": int(os.environ.get("hidden_size_obs", 64)),
            "hidden_size_comm": int(os.environ.get("hidden_size_comm", 128)),
            "hidden_size_action": int(os.environ.get("hidden_size_action", 96)),
            "n_representation_obs": int(os.environ.get("n_representation_obs", 56)),
            "n_representation_comm": int(os.environ.get("n_representation_comm", 56)),
            "n_representation_action": int(os.environ.get("n_representation_action", 56)),
            "graph_embedding": int(os.environ.get("graph_embedding", 64)),
            "graph_embedding_comm": int(os.environ.get("graph_embedding_comm", 128)),
            "learning_rate": float(os.environ.get("learning_rate", 5e-4)),
            "learning_rate_graph": float(os.environ.get("learning_rate_graph", 0.0005387456623850075)),
            "gamma1": float(os.environ.get("gamma1", 1)),
            "gamma2": float(os.environ.get("gamma2", 1)),
            "n_data_parallelism": int(os.environ.get("n_data_parallelism", 5)),

            "gamma": cfg.gamma,
            "ppo_layers": cfg.ppo_layers,
            "lmbda": cfg.lmbda,
            "eps_clip": cfg.eps_clip,
            "K_epoch": int(os.environ.get("K_epoch", 5)),
            "layers": cfg.ppo_layers,
            "feature_size": env.get_env_info()["node_features"],
            "action_size": env.get_env_info()["n_actions"],
        }
    else:
        params = {
            "hidden_size_obs": cfg.hidden_size_obs,
            "hidden_size_action": cfg.hidden_size_action,
            "n_representation_obs": cfg.n_representation_obs,
            "n_representation_action": cfg.n_representation_action,
            "graph_embedding": cfg.graph_embedding,
            "learning_rate": cfg.lr,

            "learning_rate_graph": cfg.lr_graph,
            "gamma1": cfg.gamma1,
            "n_data_parallelism": cfg.n_data_parallelism,

            "gamma": cfg.gamma,
            "ppo_layers": cfg.ppo_layers,
            "lmbda": cfg.lmbda,
            "eps_clip": cfg.eps_clip,
            "K_epoch": cfg.K_epoch,
            "layers": cfg.ppo_layers,
            "feature_size": env.get_env_info()["node_features"],
            "action_size": env.get_env_info()["n_actions"],
        }

    if vessl_on == True:
        output_dir = "/output/map_name_{}_lr_{}_hiddensizeobs_{}_hiddensizeq_{}_nrepresentationobs_{}_nrepresentationcomm_{}/".format(
            map_name1, params["learning_rate"], params["hidden_size_obs"], params["hidden_size_action"], params["n_representation_obs"], params["n_representation_action"])
    else:
        output_dir = "output/map_name_{}_lr_{}_hiddensizeobs_{}_hiddensizeq_{}_nrepresentationobs_{}_nrepresentationcomm_{}/".format(
            map_name1, params["learning_rate"], params["hidden_size_obs"], params["hidden_size_action"], params["n_representation_obs"], params["n_representation_action"])

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    log_dir = './output/logs/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    initializer = True
    agent = Agent(params)

    num_episode = 2000000
    t = 0
    epi_r = []
    win_rates = []
    eval = False
    monitor = []
    current_epsilon = params['epsilon']
    for e in range(num_episode):
        episode_reward, t, eval, current_epsilon = train(agent, env, e, t, monitor, params, current_epsilon)
        epi_r.append(episode_reward)

        if eval == True:
            win_rates = evaluation(env, agent)
            if cfg.vessl_on == True:
                vessl.log(step=t, payload={'win_rates': win_rates})


if __name__ == '__main__':
    main()

