import os
from utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque
from torch.distributions import Categorical
import numpy as np

from GLCN.GLCN import GLCN
from cfg import get_cfg
cfg = get_cfg()
from GAT.model import GAT
from GAT.layers import device
from copy import deepcopy


class VDN(nn.Module):

    def __init__(self):
        super(VDN, self).__init__()

    def forward(self, q_local):
        return torch.sum(q_local, dim = 1)

class Network(nn.Module):
    def __init__(self, obs_and_action_size, hidden_size_q):
        super(Network, self).__init__()
        self.obs_and_action_size = obs_and_action_size
        print(obs_and_action_size, hidden_size_q)
        self.fcn_1 = nn.Linear(obs_and_action_size, hidden_size_q)
        self.fcn_2 = nn.Linear(hidden_size_q, int(hidden_size_q/2))
        self.fcn_3 = nn.Linear(int(hidden_size_q/2), int(hidden_size_q/4))
        self.fcn_4 = nn.Linear(int(hidden_size_q/4), int(hidden_size_q/8))
        self.fcn_5 = nn.Linear(int(hidden_size_q/8), 1)
        torch.nn.init.xavier_uniform_(self.fcn_1.weight)
        torch.nn.init.xavier_uniform_(self.fcn_2.weight)
        torch.nn.init.xavier_uniform_(self.fcn_3.weight)
        torch.nn.init.xavier_uniform_(self.fcn_4.weight)
        torch.nn.init.xavier_uniform_(self.fcn_5.weight)

    def forward(self, obs_and_action):

        x = F.relu(self.fcn_1(obs_and_action))
        x = F.relu(self.fcn_2(x))
        x = F.relu(self.fcn_3(x))
        x = F.relu(self.fcn_4(x))
        q = self.fcn_5(x)
        return q

class NodeEmbedding(nn.Module):
    def __init__(self, feature_size, hidden_size, n_representation_obs):
        super(NodeEmbedding, self).__init__()
        self.feature_size = feature_size
        self.fcn_1 = nn.Linear(feature_size, hidden_size+10)
        self.fcn_2 = nn.Linear(hidden_size+10, hidden_size)
        self.fcn_3 = nn.Linear(hidden_size, n_representation_obs)
        torch.nn.init.xavier_uniform_(self.fcn_1.weight)
        torch.nn.init.xavier_uniform_(self.fcn_2.weight)
        torch.nn.init.xavier_uniform_(self.fcn_3.weight)

    def forward(self, node_feature):
        #node_feature = node_feature.double()
        x = F.relu(self.fcn_1(node_feature))
        x = F.relu(self.fcn_2(x))
        node_representation = self.fcn_3(x)
        return node_representation

class Replay_Buffer:
    def __init__(self, buffer_size, batch_size, num_agent):
        self.buffer = deque()


        self.step_count_list = list()
        for _ in range(10):
            self.buffer.append(deque(maxlen=buffer_size))
        self.buffer_size = buffer_size
        self.num_agent = num_agent
        self.agent_id = np.eye(self.num_agent).tolist()
        self.batch_size = batch_size
        self.step_count = 0


    def pop(self):
        self.buffer.pop()

    def memory(self, node_feature, action, action_feature, edge_index_enemy, edge_index_ally, reward, done, avail_action, dead_masking, agent_feature):
        self.buffer[0].append(node_feature)
        self.buffer[1].append(action)
        self.buffer[2].append(action_feature)
        self.buffer[3].append(edge_index_enemy)
        self.buffer[4].append(edge_index_ally)
        self.buffer[5].append(reward)
        self.buffer[6].append(done)
        self.buffer[7].append(avail_action)
        self.buffer[8].append(dead_masking)
        self.buffer[9].append(agent_feature)
        if self.step_count < self.buffer_size - 1:
            self.step_count_list.append(self.step_count)
            self.step_count += 1
    def generating_mini_batch(self, datas, batch_idx, cat):
        for s in batch_idx:
            if cat == 'node_feature':
                yield datas[0][s]
            if cat == 'action':
                yield datas[1][s]
            if cat == 'action_feature':
                yield datas[2][s]
            if cat == 'edge_index_enemy':
                yield datas[3][s]
            if cat == 'edge_index_ally':
                yield datas[4][s]
            if cat == 'reward':
                yield datas[5][s]
            if cat == 'done':
                yield datas[6][s]
            if cat == 'node_feature_next':
                yield datas[0][s+1]
            if cat == 'action_feature_next':
                yield datas[2][s+1]
            if cat == 'edge_index_enemy_next':
                yield datas[3][s+1]
            if cat == 'edge_index_ally_next':
                yield datas[4][s+1]
            if cat == 'avail_action_next':
                yield datas[7][s+1]

            if cat == 'dead_masking':
                yield datas[8][s]

            if cat == 'agent_feature':
                yield datas[9][s]
            if cat == 'agent_feature_next':
                yield datas[9][s+1]



    def sample(self):
        step_count_list = self.step_count_list[:]
        step_count_list.pop()

        sampled_batch_idx = random.sample(step_count_list, self.batch_size)


        node_feature = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='node_feature')
        node_features = list(node_feature)

        action = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='action')

        actions = list(action)


        action_feature = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='action_feature')

        action_features = list(action_feature)

        edge_index_enemy = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='edge_index_enemy')
        edge_indices_enemy = list(edge_index_enemy)

        edge_index_ally = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='edge_index_ally')
        edge_indices_ally = list(edge_index_ally)

        reward = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='reward')
        rewards = list(reward)

        done = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='done')
        dones = list(done)

        node_feature_next = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='node_feature_next')
        node_features_next = list(node_feature_next)

        action_feature_next = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='action_feature_next')
        action_features_next = list(action_feature_next)

        edge_index_enemy_next = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='edge_index_enemy_next')
        edge_indices_enemy_next = list(edge_index_enemy_next)

        edge_index_ally_next = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='edge_index_ally_next')
        edge_indices_ally_next = list(edge_index_ally_next)
        avail_action_next = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='avail_action_next')
        avail_actions_next = list(avail_action_next)

        dead_masking = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='dead_masking')
        dead_masking = list(dead_masking)

        agent_feature = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='agent_feature')
        agent_feature = list(agent_feature)

        agent_feature_next = self.generating_mini_batch(self.buffer, sampled_batch_idx, cat='agent_feature_next')
        agent_feature_next = list(agent_feature_next)

        return node_features, actions, action_features, edge_indices_enemy, edge_indices_ally, rewards, dones, node_features_next, action_features_next, edge_indices_enemy_next, edge_indices_ally_next, avail_actions_next,dead_masking, agent_feature, agent_feature_next

class Agent:
    def __init__(self,
                 num_agent,
                 num_enemy,
                 feature_size,

                 hidden_size_obs,
                 hidden_size_comm,
                 hidden_size_action,
                 hidden_size_Q,

                 n_representation_obs,
                 n_representation_comm,
                 n_representation_action,

                 graph_embedding,
                 graph_embedding_comm,

                 buffer_size,
                 batch_size,
                 learning_rate,
                 learning_rate_graph,
                 gamma,
                 gamma1,
                 gamma2,
                 anneal_episodes_graph_variance,
                 min_graph_variance,
                 env
    ):
        torch.manual_seed(81)
        random.seed(81)
        np.random.seed(81)

        self.num_agent = num_agent
        self.num_enemy = num_enemy
        self.feature_size = feature_size
        self.hidden_size_obs = hidden_size_obs
        self.hidden_size_comm = hidden_size_comm
        self.hidden_size_action = hidden_size_action


        self.n_representation_obs = n_representation_obs
        self.n_representation_comm = n_representation_comm
        self.n_representation_action = n_representation_action

        self.graph_embedding = graph_embedding
        self.graph_embedding_comm = graph_embedding_comm

        self.gamma1 = gamma1
        self.gamma2 = gamma2


        self.gamma = gamma
        self.agent_id = np.eye(self.num_agent).tolist()

        self.max_norm = 10

        self.VDN = VDN().to(device)
        self.VDN_target = VDN().to(device)


        self.VDN_target.load_state_dict(self.VDN.state_dict())
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer = Replay_Buffer(self.buffer_size, self.batch_size, self.num_agent)

        self.anneal_episodes_graph_variance=anneal_episodes_graph_variance
        self.min_graph_variance=min_graph_variance

        self.skip_connection = bool(os.environ.get("skip_connection", False))
        if self.skip_connection == True:
            self.graph_embedding_comm = self.graph_embedding



        # self.node_representation_enemy_obs = NodeEmbedding(feature_size=feature_size,
        #                                                    hidden_size=hidden_size_obs,
        #                                                    n_representation_obs=n_representation_obs).to(device)  # 수정사항
        # self.node_representation = NodeEmbedding(feature_size=feature_size - 1,
        #                                          hidden_size=hidden_size_obs,
        #                                          n_representation_obs=n_representation_obs).to(device)  # 수정사항
        # self.action_representation = NodeEmbedding(feature_size=feature_size + 5,
        #                                            hidden_size=hidden_size_obs,
        #                                            n_representation_obs=n_representation_obs).to(device)  # 수정사항


        self.node_representation = NodeEmbedding(feature_size=self.feature_size,
                                                   hidden_size=self.hidden_size_obs,
                                                   n_representation_obs=self.n_representation_obs).to(device)  # 수정사항
        self.node_representation_comm = NodeEmbedding(feature_size =self.feature_size-1+self.feature_size + 5,
                                                      hidden_size  =self.hidden_size_comm,
                                                      n_representation_obs=self.n_representation_comm).to(device)  # 수정사항
        if env == 'pp':
            self.action_representation = NodeEmbedding(feature_size=5,
                                                       hidden_size=self.hidden_size_action,
                                                       n_representation_obs=self.n_representation_action).to(device)  # 수정사항
        else:
            self.action_representation = NodeEmbedding(feature_size=self.feature_size + 5,
                                                       hidden_size=self.hidden_size_action,
                                                       n_representation_obs=self.n_representation_action).to(device)  # 수정사항


        self.func_obs = GLCN(feature_size=self.n_representation_obs, graph_embedding_size=self.graph_embedding, link_prediction = False).to(device)
        if cfg.given_edge == True:
            self.func_glcn = GLCN(feature_size=self.graph_embedding+self.n_representation_comm,
                                  graph_embedding_size=self.graph_embedding_comm, link_prediction = False).to(device)
            self.func_glcn2 = GLCN(feature_size=self.graph_embedding + self.n_representation_comm,
                                  graph_embedding_size=self.graph_embedding_comm, link_prediction=False).to(device)
        else:
            self.func_glcn = GLCN(feature_size=self.graph_embedding+self.n_representation_comm,
                                  feature_obs_size=self.graph_embedding,
                                  graph_embedding_size=self.graph_embedding_comm, link_prediction = True).to(device)


        self.Q = Network(self.graph_embedding_comm + self.n_representation_action, hidden_size_Q).to(device)
        self.Q_tar = Network(self.graph_embedding_comm + self.n_representation_action, hidden_size_Q).to(device)
        self.Q_tar.load_state_dict(self.Q.state_dict())

        if cfg.given_edge == True:
            self.eval_params = list(self.VDN.parameters()) + \
                               list(self.Q.parameters()) + \
                               list(self.node_representation.parameters()) + \
                               list(self.node_representation_comm.parameters()) + \
                               list(self.func_obs.parameters()) + \
                               list(self.func_glcn.parameters()) + \
                               list(self.func_glcn2.parameters()) + \
                               list(self.action_representation.parameters())
        else:
            self.eval_params = [p for n, p in self.func_glcn.named_parameters() if n != 'a_link'] + \
                       list(self.VDN.parameters()) + \
                       list(self.Q.parameters()) + \
                       list(self.node_representation.parameters()) + \
                       list(self.node_representation_comm.parameters()) + \
                       list(self.func_obs.parameters()) + \
                       list(self.action_representation.parameters())

            self.graph_params = [p for n, p in self.func_glcn.named_parameters() if n == 'a_link']

        param_groups = [
            {'params': self.eval_params},
            {'params': self.graph_params, 'lr': learning_rate_graph}
        ]
        self.optimizer = optim.RMSprop(param_groups, lr=learning_rate)


    def save_model(self, file_dir, e):
        torch.save({
                        "1": self.Q.state_dict(),
                        "2": self.Q.state_dict(),
                        "3": self.func_glcn.state_dict(),
                        "4": self.func_obs.state_dict(),
                        "5": self.action_representation.state_dict(),
                        "6": self.node_representation_comm.state_dict() ,
                        "7": self.node_representation.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict()
                        },
                       file_dir+ "episode%d.pt" % e)



    def load_model(self, path):
        self = torch.load(path)


    def get_node_representation_temp(self, node_feature, agent_feature, edge_index_obs,edge_index_comm, n_agent,
                                    dead_masking,
                                     mini_batch = False):
        if mini_batch == False:
            with torch.no_grad():
                node_feature = torch.tensor(node_feature, dtype=torch.float,device=device)
                agent_feature = agent_feature.to(device)
                node_embedding_obs  = self.node_representation(node_feature)
                node_embedding_comm = self.node_representation_comm(agent_feature)

                edge_index_obs  = torch.tensor(edge_index_obs, dtype=torch.long, device=device)
                edge_index_comm = torch.tensor(edge_index_comm, dtype=torch.long, device=device)

                node_embedding_obs = self.func_obs(X = node_embedding_obs, A = edge_index_obs)[:n_agent,:]

                cat_embedding = torch.cat([node_embedding_obs, node_embedding_comm], dim = 1)

                if cfg.given_edge == True:
                    node_embedding = self.func_glcn(X=cat_embedding[:n_agent,:], dead_masking= dead_masking, A=edge_index_comm)
                    return node_embedding

                else:
                    node_embedding, A, X = self.func_glcn(X = cat_embedding, dead_masking= dead_masking, A = None)
                    # 오류 수정
                    return node_embedding, A, X
        else:
            node_feature = torch.tensor(node_feature, dtype=torch.float, device=device)
            agent_feature = torch.tensor(agent_feature, dtype=torch.float, device=device)
            node_embedding_obs  = self.node_representation(node_feature)
            node_embedding_comm = self.node_representation_comm(agent_feature)

            node_embedding_obs = self.func_obs(X = node_embedding_obs, A = edge_index_obs, mini_batch = mini_batch)[:, :n_agent,:]
            #print(node_embedding_obs.shape, node_embedding_comm.shape)
            cat_embedding = torch.cat([node_embedding_obs, node_embedding_comm], dim=2)

            if cfg.given_edge == True:
                node_embedding = self.func_glcn(X=cat_embedding[:n_agent,:], A=edge_index_comm, dead_masking= dead_masking, mini_batch=mini_batch)
                return node_embedding
            else:
                #print("전", cat_embedding.shape,cat_embedding[:, :n_agent,:].shape)
                node_embedding, A, X, D = self.func_glcn(X = cat_embedding, dead_masking= dead_masking, A = None, mini_batch = mini_batch)
                #writer = SummaryWriter('runs/your_model_experiment')



                return node_embedding, A, X, D





    def cal_Q(self, obs, actions, action_features, avail_actions_next, agent_id, target = False):
        """

        node_representation
        - training 시        : batch_size X num_nodes X feature_size
        - action sampling 시 : num_nodes X feature_size

        """
        if target == False:
            action_features = torch.tensor(action_features).to(device=device, dtype=torch.float32)


            action_size = action_features.shape[1]
            obs_n = obs[:, agent_id].unsqueeze(1).expand([self.batch_size, action_size, self.graph_embedding_comm])

            action_embedding = self.action_representation(action_features)
            obs_and_action = torch.concat([obs_n, action_embedding], dim=2)
            obs_and_action = obs_and_action.float()
            q = self.Q(obs_and_action)
            q = q.squeeze(2)                                             # q.shape :      (batch_size, action_size)
            actions = torch.tensor(actions, device = device).long()
            act_n = actions[:, agent_id].unsqueeze(1)                    # action.shape : (batch_size, 1)
            q = torch.gather(q, 1, act_n).squeeze(1)                     # q.shape :      (batch_size, 1)
            return q
        else:
            with torch.no_grad():
                obs_next = obs
                action_features_next = action_features
                action_features_next = torch.tensor(action_features_next).to(device=device, dtype=torch.float32)
                action_size = action_features_next.shape[1]
                obs_next = obs_next[:, agent_id].unsqueeze(1).expand([self.batch_size, action_size, self.graph_embedding_comm])
                action_embedding_next = self.action_representation(action_features_next)

                obs_and_action_next = torch.concat([obs_next, action_embedding_next], dim=2)
                obs_and_action_next = obs_and_action_next.float()
                q_tar = self.Q_tar(obs_and_action_next)                        # q.shape :      (batch_size, action_size, 1)
                q_tar = q_tar.squeeze(2)                                       # q.shape :      (batch_size, action_size)
                avail_actions_next = torch.tensor(avail_actions_next, device = device).bool()
                mask = avail_actions_next[:, agent_id]

                q_tar = q_tar.masked_fill(mask == 0, float('-inf'))

                q_tar_max = torch.max(q_tar, dim = 1)[0]
                return q_tar_max



    @torch.no_grad()
    def sample_action(self, node_representation, action_feature, avail_action, epsilon):
        """
        node_representation 차원 : n_agents X n_representation_comm
        action_feature 차원      : action_size X n_action_feature
        avail_action 차원        : n_agents X action_size
        """
        mask = torch.tensor(avail_action, device=device).bool()
        action_feature = torch.tensor(action_feature, device=device, dtype = torch.float64).float()
        action_size = action_feature.shape[0]
        action = []
        action_embedding = self.action_representation(action_feature)
        action_space = [i for i in range(action_size)]
        selected_action_feature_list = list()
        for n in range(self.num_agent):
            #print(node_representation.shape)
            obs = node_representation[n].expand(action_size, node_representation[n].shape[0])         # 차원 : action_size X n_representation_comm
            obs_cat_action = torch.concat([obs, action_embedding], dim = 1)    # 차원 : action_size X
            obs_cat_action = obs_cat_action.float()
            Q = self.Q(obs_cat_action).squeeze(1)                                                # 차원 : action_size X 1

            Q = Q.masked_fill(mask[n, :]==0, float('-inf'))
            greedy_u = torch.argmax(Q)

            mask_n = np.array(avail_action[n], dtype=np.float64)

            if np.random.uniform(0, 1) >= epsilon:
                u = greedy_u
                action.append(u.item())
                selected_action_feature_list.append(action_feature[u.item()])
            else:
                u = np.random.choice(action_space, p=mask_n / np.sum(mask_n))
                action.append(u)

                selected_action_feature_list.append(action_feature[u])
        #print(selected_action_feature_list)
        return action, torch.stack(selected_action_feature_list)



    def learn(self, e):
        node_features, actions, action_features, edge_indices_enemy, edge_indices_ally, rewards, dones, node_features_next, action_features_next, edge_indices_enemy_next, edge_indices_ally_next, avail_actions_next,dead_masking, agent_feature, agent_feature_next = self.buffer.sample()

        """
        node_features : batch_size x num_nodes x feature_size
        actions : batch_size x num_agents
        action_feature :     batch_size x action_size x action_feature_size
        avail_actions_next : batch_size x num_agents x action_size 
        """
        # print(torch.tensor(action_features).shape)
        # print(torch.tensor(avail_actions_next).shape)
        num_nodes = torch.tensor(node_features).shape[1]
        n_agent = torch.tensor(avail_actions_next).shape[1]

        if cfg.given_edge == True:
            obs = self.get_node_representation_temp(node_features, agent_feature, edge_indices_enemy, edge_indices_ally,
                                                    n_agent = n_agent,
                                                    dead_masking = dead_masking,
                                                    mini_batch=True)
            obs_next = self.get_node_representation_temp(node_features_next, agent_feature_next, edge_indices_enemy_next, edge_indices_ally_next,
                                                         n_agent=n_agent,
                                                         dead_masking=dead_masking,
                                                         mini_batch=True)
        else:
            obs, A, X, D = self.get_node_representation_temp(node_features, agent_feature, edge_indices_enemy, edge_indices_ally,
                                                             n_agent=n_agent,
                                                             dead_masking=dead_masking,
                                                             mini_batch=True)
            obs_next, _, _, _ = self.get_node_representation_temp(node_features_next, agent_feature_next, edge_indices_enemy_next, edge_indices_ally_next,
                                                                  n_agent=n_agent,
                                                                  dead_masking=dead_masking,
                                                                  mini_batch=True)
            gamma1 = self.gamma1
            gamma2 = self.gamma2
            lap_quad, sec_eig_upperbound, L = get_graph_loss(X, A, num_nodes, e, self.anneal_episodes_graph_variance,self.min_graph_variance)



        dones = torch.tensor(dones, device = device, dtype = torch.float)
        rewards = torch.tensor(rewards, device = device, dtype = torch.float)
        q = [self.cal_Q(obs=obs,
                         actions=actions,
                         action_features=action_features,
                         avail_actions_next=None,
                         agent_id=agent_id,
                         target=False) for agent_id in range(self.num_agent)]

        q_tar = [self.cal_Q(obs=obs_next,
                             actions=None,
                             action_features=action_features_next,
                             avail_actions_next=avail_actions_next,
                             agent_id=agent_id,
                             target=True) for agent_id in range(self.num_agent)]

        q_tot = torch.stack(q, dim=1)
        q_tot_tar = torch.stack(q_tar, dim=1)
        q_tot = self.VDN(q_tot)
        q_tot_tar = self.VDN_target(q_tot_tar)
        td_target = rewards*self.num_agent + self.gamma* (1-dones)*q_tot_tar
###
        if cfg.given_edge == True:
            rl_loss = F.mse_loss(q_tot, td_target.detach())
            loss = rl_loss
        else:

            rl_loss = F.mse_loss(q_tot, td_target.detach())
            graph_loss = gamma1 * lap_quad - gamma2 * gamma1 * sec_eig_upperbound
            loss = graph_loss+rl_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_params, float(os.environ.get("grad_clip", 10)))
        torch.nn.utils.clip_grad_norm_(self.graph_params, float(os.environ.get("grad_clip_graph", 0.51)))
        self.optimizer.step()
        self.optimizer.zero_grad()


        #self.Q_tar.load_state_dict(self.Q.state_dict())

        tau = 1e-4
        for target_param, local_param in zip(self.Q_tar.parameters(), self.Q.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)
        for target_param, local_param in zip(self.VDN_target.parameters(), self.VDN.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)
        if cfg.given_edge == True:
            return loss
        else:
            return loss, lap_quad.tolist(), sec_eig_upperbound.tolist(), rl_loss.tolist(), q_tot.tolist()