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
        while (not done) and (step < max_episode_len):
            node_feature, edge_index_enemy, edge_index_comm = env.get_heterogeneous_graph(heterogeneous=heterogenous)
            if cfg.given_edge == True:
                node_embedding = agent.get_node_representation_gpo(node_feature, edge_index_enemy, edge_index_comm)
            else:
                node_embedding, _, _ = agent.get_node_representation_gpo(node_feature, edge_index_enemy,
                                                                         edge_index_comm)
            avail_action = env.get_avail_actions()
            action_feature = env.get_action_feature()  # 차원 : action_size X n_action_feature
            agent.eval_check(eval=True)
            action, prob = agent.sample_action(node_embedding, action_feature, avail_action,
                                               num_agent=env.get_env_info()["n_agents"])
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







def train(agent, env, e, t, monitor, params):
    max_episode_limit = env.episode_limit
    env.reset()
    done = False
    episode_reward = 0
    step = 0
    eval = False
    start = time.time()

    while (not done) and (step < max_episode_limit):
        node_feature, edge_index_enemy, edge_index_comm = env.get_heterogeneous_graph(heterogeneous=heterogenous)
        if cfg.given_edge == True:
            node_embedding = agent.get_node_representation_gpo(node_feature, edge_index_enemy, edge_index_comm)
        else:
            node_embedding, _, _ = agent.get_node_representation_gpo(node_feature, edge_index_enemy, edge_index_comm)
        avail_action = env.get_avail_actions()
        action_feature = env.get_action_feature()  # 차원 : action_size X n_action_feature
        agent.eval_check(eval=True)
        action, prob = agent.sample_action(node_embedding, action_feature, avail_action, num_agent = env.get_env_info()["n_agents"])
        reward, done, info = env.step(action)
        transition = (
                      node_feature,
                      edge_index_enemy,
                      avail_action,
                      action,
                      prob,
                      action_feature,
                      reward,
                      done,
                      edge_index_comm,
                      )
        agent.put_data(transition)
        episode_reward += reward
        t += 1
        step += 1
        if (t % 5000 == 0) and (t > 0):
            eval = True
    if cfg.vessl_on == True:
        vessl.log(step=e, payload={'episode_reward': episode_reward})
    if (e % params["n_data_parallelism"] == 0) and (e > 0):
        cum_surr, cum_value_loss, cum_lap_quad, cum_sec_eig_upperbound, second_eigenvalue= agent.learn()
        monitor.append((e, cum_surr, cum_value_loss, cum_lap_quad, cum_sec_eig_upperbound))
        df = pd.DataFrame(monitor)

        if cfg.vessl_on == True:
            df.to_csv("/output/df.csv")
            vessl.log(step = e, payload={'fiedler': second_eigenvalue})
            # vessl.log(step = e, payload = {'surrogate loss' : cum_surr})
            # vessl.log(step = e, payload = {'value loss': cum_value_loss})
            vessl.log(step = e, payload = {'laplacian quadractic': cum_lap_quad})
            vessl.log(step = e, payload = {'ub': cum_sec_eig_upperbound})







    print("{} Total reward in episode {} = {}, time_step : {}, episode_duration : {}".format(env.map_name,
                                                                                                e,
                                                                                                np.round(episode_reward, 3),
                                                                                                t, np.round(time.time()-start, 3)))
    return episode_reward, t, eval

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
            "hidden_size_obs": int(os.environ.get("hidden_size_obs", 52)),
            "hidden_size_comm": int(os.environ.get("hidden_size_comm", 56)),
            "hidden_size_action": int(os.environ.get("hidden_size_action", 52)),
            "n_representation_obs": int(os.environ.get("n_representation_obs", 30)),
            "n_representation_comm": int(os.environ.get("n_representation_comm", 30)),
            "n_representation_action": int(os.environ.get("n_representation_action", 48)),
            "graph_embedding": int(os.environ.get("graph_embedding", 84)),
            "graph_embedding_comm": int(os.environ.get("graph_embedding_comm", 72)),
            "learning_rate": cfg.lr,
            "learning_rate_graph": float(os.environ.get("learning_rate_graph", 0.0005387456623850075)),
            "gamma1": float(os.environ.get("gamma1", 0.1)),
            "gamma2": float(os.environ.get("gamma2", 1e-3)),
            "n_data_parallelism": int(os.environ.get("n_data_parallelism", 5)),


            "gamma": cfg.gamma,
            "ppo_layers": cfg.ppo_layers,
            "lmbda": cfg.lmbda,
            "eps_clip": cfg.eps_clip,
            "K_epoch": cfg.K_epoch,
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

    num_episode = 20000
    t = 0
    epi_r = []
    win_rates = []
    eval = False
    monitor = []
    for e in range(num_episode):
        episode_reward, t, eval = train(agent, env, e, t, monitor, params)
        epi_r.append(episode_reward)

        if eval == True:
            win_rates = evaluation(env, agent)
            if cfg.vessl_on == True:
                vessl.log(step=t, payload={'win_rates': win_rates})


        #     if vessl_on == True:
        #         print("Model save")
        #         agent1.save_model(output_dir+"{}.pt".format(t))
        #     else:
        #         print("Model save")
        #         agent1.save_model(output_dir+"{}.pt".format(t))
        # if e % 100 == 1:
        #     if vessl_on == True:
        #         vessl.log(step = e, payload = {'reward' : np.mean(epi_r)})
        #         epi_r = []
        #         r_df= pd.DataFrame(epi_r)
        #         r_df.to_csv(output_dir+"reward.csv")
        #     else:
        #         r_df= pd.DataFrame(epi_r)
        #         r_df.to_csv(output_dir+"reward.csv")

        # if eval == True:
        #     win_rate = evaluation(env1, agent1, 32)
        #     win_rates.append(win_rate)
        #     if vessl_on == True:
        #         vessl.log(step = t, payload = {'win_rate' : win_rate})
        #         wr_df = pd.DataFrame(win_rates)
        #         wr_df.to_csv(output_dir+"win_rate.csv")
        #     else:
        #         wr_df = pd.DataFrame(win_rates)
        #         wr_df.to_csv(output_dir+"win_rate.csv")




if __name__ == '__main__':
    main()

