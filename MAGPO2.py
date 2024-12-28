import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import torch
import torch.optim as optim
import os
from torch.optim.lr_scheduler import StepLR
from collections import OrderedDict
from GDN import NodeEmbedding
from utils import *
from GAT.model import GAT
from GAT.layers import device

import numpy as np
from GLCN.GLCN import GLCN
from cfg import get_cfg
from copy import deepcopy
cfg = get_cfg()
from torch.distributions import Categorical

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def huber_loss(y_pred, y_true, delta=1.0):
    # 예측값과 실제값의 차이 계산
    error = y_pred - y_true

    # 절대값이 delta보다 작은 경우와 큰 경우를 구분
    is_small_error = torch.abs(error) <= delta

    # Huber Loss 계산
    squared_loss = 0.5 * error ** 2
    linear_loss = delta * torch.abs(error) - 0.5 * delta ** 2

    # 조건에 따라 loss 선택
    return torch.where(is_small_error, squared_loss, linear_loss)

class PolicyNoAttack(nn.Module):
    def __init__(self, state_action_size, layers=[8, 12]):
        super(PolicyNoAttack, self).__init__()
        self.NN_sequential = OrderedDict()
        layers = eval(layers)
        self.fc_pi = nn.Linear(state_action_size, layers[0])
        self.fcn = OrderedDict()
        last_layer = layers[0]
        for i in range(1, len(layers)):
            layer = layers[i]
            if i <= len(layers)-2:
                self.fcn['linear{}'.format(i)] = nn.Linear(last_layer, layer)
                self.fcn['activation{}'.format(i)] = nn.ELU()
                last_layer = layer

        self.forward_cal = nn.Sequential(self.fcn)
        self.output_pi = nn.Linear(last_layer, 7)



    def forward(self, x):

        x = self.fc_pi(x)
        x = F.elu(x)
        x = self.forward_cal(x)
        pi = self.output_pi(x)
        return pi



class PPONetwork(nn.Module):
    def __init__(self, state_action_size, layers=[8, 12]):
        super(PPONetwork, self).__init__()

        self.NN_sequential = OrderedDict()
        layers = eval(layers)
        self.fc_pi = nn.Linear(state_action_size, layers[0])

        self.fcn = OrderedDict()
        last_layer = layers[0]
        for i in range(1, len(layers)):
            layer = layers[i]
            if i <= len(layers)-2:
                self.fcn['linear{}'.format(i)] = nn.Linear(last_layer, layer)
                self.fcn['activation{}'.format(i)] = nn.ELU()
                last_layer = layer

        self.forward_cal = nn.Sequential(self.fcn)
        self.output_pi = nn.Linear(last_layer, 1)


    def forward(self, x, visualize = False):
        x = self.fc_pi(x)
        x = F.elu(x)
        x = self.forward_cal(x)
        pi = self.output_pi(x)
        return pi

class ObservationNetwork(nn.Module):
    def __init__(self, feature_size):
        super(ObservationNetwork, self).__init__()
        self.obs_n = nn.Parameter(torch.FloatTensor(feature_size))

class ValueNetwork(nn.Module):
    def __init__(self, state_size, layers=[8, 12]):
        super(ValueNetwork, self).__init__()
        self.state_size = state_size
        self.NN_sequential = OrderedDict()
        layers = eval(layers)
        self.fc_v = nn.Linear(state_size, layers[0])
        self.fcn = OrderedDict()
        last_layer = layers[0]
        for i in range(1, len(layers)):
            layer = layers[i]
            if i <= len(layers)-2:
                self.fcn['linear{}'.format(i)] = nn.Linear(last_layer, layer)
                self.fcn['activation{}'.format(i)] = nn.ELU()
                last_layer = layer
        self.forward_cal = nn.Sequential(self.fcn)
        self.output_v = nn.Linear(last_layer, 1)

    def forward(self, x):
        x = self.fc_v(x)
        x = F.elu(x)
        x = self.forward_cal(x)
        v = self.output_v(x)
        return v


class Agent:
    def __init__(self,
                 params
                 ):
        self.action_size = params["action_size"]
        self.feature_size = params["feature_size"]
        self.hidden_size_obs = params["hidden_size_obs"]
        self.hidden_size_comm = params["hidden_size_comm"]
        self.hidden_size_action = params["hidden_size_action"]
        self.n_representation_obs = params["n_representation_obs"]
        self.n_representation_comm = params["n_representation_comm"]
        self.n_representation_action = params["n_representation_action"]
        self.graph_embedding = params["graph_embedding"]
        self.graph_embedding_comm = params["graph_embedding_comm"]
        self.learning_rate = params["learning_rate"]
        self.gamma = params["gamma"]
        self.gamma1 = params["gamma1"]
        self.gamma2 = params["gamma2"]
        self.learning_rate_graph = params["learning_rate_graph"]
        self.n_data_parallelism = params["n_data_parallelism"]



        self.lmbda = params["lmbda"]
        self.eps_clip = params["eps_clip"]
        self.K_epoch = params["K_epoch"]
        self.layers = params["ppo_layers"]
        self.data = []





        """
        
        NodeEmbedding 수정해야 함
        
        """
        embedding_size_node = 48
        embedding_size_sum_state = 28

        embedding_size_action = 48
        embedding_size_agent = 64

        graph_embedding_obs_size = 56
        graph_embedding_agent_size = 72
        graph_embedding_action_size = 64


        self.obs_net = ObservationNetwork(self.n_representation_obs).to(device)
        self.node_representation_sum_state = NodeEmbedding(feature_size=12,
                                                       hidden_size=self.hidden_size_obs,
                                                       n_representation_obs=embedding_size_sum_state).to(device)  # 수정사항

        self.node_representation_agent = NodeEmbedding(feature_size=2*self.feature_size+6,
                                                 hidden_size=self.hidden_size_obs,
                                                 n_representation_obs=embedding_size_agent).to(device)  # 수정사항

        self.node_representation =   NodeEmbedding(feature_size=self.feature_size,
                                                   hidden_size=self.hidden_size_obs,
                                                   n_representation_obs=embedding_size_node).to(device)  # 수정사항

        self.node_representation_comm = NodeEmbedding(feature_size =self.feature_size-1,
                                                      hidden_size  =self.hidden_size_comm,
                                                      n_representation_obs=self.n_representation_comm).to(device)  # 수정사항

        self.action_representation = NodeEmbedding(feature_size=self.feature_size,
                                                   hidden_size=self.hidden_size_action,
                                                   n_representation_obs=embedding_size_action).to(device)  # 수정사항



        self.func_obs = GLCN(feature_size=embedding_size_node,
                              graph_embedding_size=graph_embedding_obs_size, link_prediction = False).to(device)
        self.func_glcn = GLCN(feature_size=embedding_size_agent+graph_embedding_obs_size,
                                graph_embedding_size=graph_embedding_agent_size, link_prediction = False).to(device)

        self.func_action = GLCN(feature_size=embedding_size_action,
                                graph_embedding_size=embedding_size_action, link_prediction=False).to(device)

        self.func_action2 = GLCN(feature_size=embedding_size_action,
                                graph_embedding_size=graph_embedding_action_size, link_prediction=False).to(device)


        # self.func_action = GLCN(feature_size=self.n_representation_obs,
        #                         graph_embedding_size=self.graph_embedding, link_prediction=False).to(device)



        self.network_no_attack = PolicyNoAttack(
                                  state_action_size = embedding_size_sum_state+
                                                      graph_embedding_agent_size,
                                  layers=self.layers).to(device)

        self.network = PPONetwork(state_action_size=embedding_size_sum_state+
                                                      graph_embedding_agent_size+
                                                      graph_embedding_action_size,
                                  layers=self.layers).to(device)



        self.valuenetwork = ValueNetwork(state_size=embedding_size_sum_state+
                                                      graph_embedding_agent_size,

                                  layers=self.layers).to(device)

        if cfg.given_edge == True:
            self.eval_params = list(self.network.parameters()) + \
                               list(self.valuenetwork.parameters()) + \
                               list(self.node_representation.parameters()) + \
                               list(self.node_representation_comm.parameters()) + \
                               list(self.action_representation.parameters()) + \
                               list(self.func_obs.parameters()) + \
                               list(self.func_glcn.parameters()) + \
                               list(self.func_glcn2.parameters())
        else:
            self.eval_params = list(self.network_no_attack.parameters()) + \
                               list(self.network.parameters()) + \
                               list(self.valuenetwork.parameters()) + \
                               list(self.obs_net.parameters())+ \
                               list(self.node_representation_sum_state.parameters()) + \
                               list(self.node_representation_agent.parameters()) + \
                               list(self.node_representation.parameters()) + \
                               list(self.node_representation_comm.parameters()) + \
                               list(self.action_representation.parameters()) + \
                               list(self.func_obs.parameters()) + \
                               list(self.func_action.parameters()) + \
                               list(self.func_action2.parameters()) + \
                               list(self.func_glcn.parameters())

        if cfg.optimizer == 'ADAM':
            self.optimizer1 = optim.Adam(self.eval_params, lr=self.learning_rate)  #
        if cfg.optimizer == 'ADAMW':
            self.optimizer = optim.AdamW(self.eval_params, lr=self.learning_rate)  #
        #self.scheduler = StepLR(optimizer=self.optimizer1, step_size=cfg.scheduler_step, gamma=cfg.scheduler_ratio)

        self.node_features_list = list()
        self.edge_index_enemy_list = list()
        self.avail_action_list = list()
        self.action_list = list()
        self.prob_list = list()
        self.action_feature_list = list()
        self.reward_list = list()
        self.done_list = list()
        self.edge_index_comm_list = list()
        self.factorize_pi_list = list()
        self.dead_masking = list()
        self.state_list = list()
        self.summarized_state_list = list()
        self.agent_feature_list = list()
        self.batch_store = []
        self.no_attack_identity = np.eye(7)

        self.fully_connected_edge =[[],[]]
        for i in range(6):
            for j in range(6):
                self.fully_connected_edge[0].append(i)
                self.fully_connected_edge[1].append(j)
        self.fully_connected_edge  = torch.tensor(self.fully_connected_edge)
        self.batched_fully_connected_edge = [self.fully_connected_edge  for _ in range(self.n_data_parallelism)]



    def batch_reset(self):
        self.batch_store = []

    @torch.no_grad()
    def get_td_target(self, ship_features, node_features_missile, heterogenous_edges, possible_actions, action_feature, reward, done):
        obs_next, act_graph = self.get_node_representation(ship_features,node_features_missile, heterogenous_edges,mini_batch=False)
        td_target = reward + self.gamma * self.network.v(obs_next) * (1 - done)
        return td_target.tolist()[0][0]

    # num_agent = env1.get_env_info()["n_agents"],
    # num_enemy = env1.get_env_info()["n_enemies"],
    # feature_size = env1.get_env_info()["node_features"],
    # action_size = env1.get_env_info()["n_actions"],

    @torch.no_grad()
    def sample_action(self, summarized_state_feature, node_representation, action_feature, num_agent,
                      edge_index_obs,
                      no_attack_mask, attack_mask, epsilon = 0, ):
        """
        node_representation 차원 : n_agents X n_representation_comm
        action_feature 차원      : action_size X n_action_feature
        avail_action 차원        : n_agents X action_size
        """

        no_attack_mask = torch.tensor(deepcopy(no_attack_mask), device=device).bool()
        attack_mask = torch.tensor(deepcopy(attack_mask), device=device).bool()

        action_feature = torch.tensor(
                                      action_feature,
                                      device=device,
                                      dtype = torch.float64
                                      ).float()
        action_size = action_feature.shape[0]-num_agent+1
        action = []
        probs = []
        selected_action_feature = []
        action_embedding = self.action_representation(action_feature)
        edge_index_obs2 = deepcopy(edge_index_obs)
        edge_index_obs2 = torch.tensor(edge_index_obs2).long().to(device)


        action_embedding = self.func_action(X=action_embedding, A=edge_index_obs2)
        action_embedding = self.func_action2(X=action_embedding, A=edge_index_obs2)
        action_embedding = action_embedding[num_agent:, :]
        # print("전", action_embedding.shape)
        action_embedding = torch.concat([action_embedding, torch.zeros(1, action_embedding.shape[1]).to(device)], axis = 0)
        summarized_state_feature = torch.tensor(summarized_state_feature, dtype = torch.float).to(device)
        summarized_state_embedding = self.node_representation_sum_state(summarized_state_feature)

        converted_no_attack_masks = list()
        converted_attack_masks = list()
        for n in range(num_agent):
            no_attack_mask_n = no_attack_mask[n, :]
            attack_mask_n = attack_mask[n, :]

            obs = node_representation[n].expand(action_size, node_representation[n].shape[0])
            sum_state = torch.tensor(summarized_state_embedding).to(device).repeat(action_size, 1)


            obs_cat_action = torch.concat([sum_state, obs, action_embedding], dim = 1)                           # shape :
            obs_cat = torch.concat([sum_state, obs], dim=1)  # shape :
            #print(True in attack_mask_n, attack_mask_n)
            if True in attack_mask_n:
                no_attack_mask_n = torch.cat([no_attack_mask_n, torch.tensor([True]).to(device)])
            else:
                no_attack_mask_n = torch.cat([no_attack_mask_n, torch.tensor([False]).to(device)])

            h = self.network_no_attack(obs_cat[0])
            h = h.masked_fill(no_attack_mask_n ==0, -1e8)
            prob_no_attack = torch.softmax(h, dim=-1)  # 에이전트 별 확률
            #print(prob_no_attack, no_attack_mask_n)
            m = Categorical(prob_no_attack)
            u_no_attack = m.sample().item()

            pi_no_attack = prob_no_attack[u_no_attack]

            if u_no_attack == 6:
                attack_mask_n = torch.cat([
                    attack_mask_n,
                    torch.tensor([False]).to(device)])
            else:
                attack_mask_n = torch.cat([
                    torch.tensor([False]*len(attack_mask_n), dtype = torch.bool).to(device),
                    torch.tensor([True]).to(device)])

            attack_mask_n = attack_mask_n.bool()


            logit = self.network(obs_cat_action).squeeze(1)
            logit = logit.masked_fill(attack_mask_n==0, -1e8)


            prob_attack = torch.softmax(logit, dim=-1) # 에이전트 별 확률

            m = Categorical(prob_attack)
            u_attack = m.sample().item()

            pi_attack = prob_attack[u_attack]
            pi = pi_no_attack*pi_attack
            u = [u_no_attack, u_attack]
            action.append(u)
            probs.append(pi.cpu().item())
            selected_action = torch.tensor(self.no_attack_identity[u_no_attack].tolist()+action_feature[u_attack].tolist())
            selected_action_feature.append(selected_action)
            converted_no_attack_masks.append(no_attack_mask_n.tolist())
            converted_attack_masks.append(attack_mask_n.tolist())



        selected_action_feature = torch.stack(selected_action_feature)
        return action, probs, selected_action_feature, converted_no_attack_masks, converted_attack_masks


    def get_node_representation_gpo(self, node_feature, agent_feature,
                                    edge_index_obs,
                                    edge_index_comm, n_agent, dead_masking, mini_batch = False):
        if mini_batch == False:
            with torch.no_grad():
                node_feature = torch.tensor(node_feature, dtype=torch.float,device=device)
                agent_feature = torch.tensor(agent_feature, dtype=torch.float, device=device)

                node_embedding_obs = self.node_representation(node_feature)
                node_embedding_agent = self.node_representation_agent(agent_feature)

                embedding = node_embedding_obs

                edge_index_obs = torch.tensor(edge_index_obs, dtype=torch.long, device=device)
                edge_index_comm = torch.tensor(edge_index_comm, dtype=torch.long, device=device)

                node_embedding_obs = self.func_obs(X = embedding, A = edge_index_obs)
                agent_embedding = torch.concat([node_embedding_obs[:n_agent, :], node_embedding_agent], axis=1)

                if cfg.given_edge == True:
                    node_embedding = self.func_glcn(X=node_embedding_obs[:n_agent, :], A=edge_index_comm)
                    #node_embedding = self.func_glcn2(X=node_embedding, A=edge_index_comm)
                    return node_embedding
                else:
                    node_embedding = self.func_glcn(dead_masking= dead_masking, X = agent_embedding, A = edge_index_comm)

                    return node_embedding
        else:
            node_feature = torch.tensor(node_feature, dtype=torch.float, device=device)
            agent_feature = torch.tensor(agent_feature, dtype=torch.float, device=device)
            node_embedding_obs = self.node_representation(node_feature)
            node_embedding_agent = self.node_representation_agent(agent_feature)

            time_length = node_embedding_obs.shape[0]
            obs_embedding = self.obs_net.obs_n.repeat(time_length, n_agent, 1)
            enemy_embedding = node_embedding_obs
            embedding =node_embedding_obs #torch.concat([obs_embedding, enemy_embedding], axis = 1)
            node_embedding_obs = self.func_obs(X = embedding, A = edge_index_obs, mini_batch = mini_batch)
            agent_embedding = torch.concat([node_embedding_obs[:, :n_agent, :], node_embedding_agent], axis=2)

            if cfg.given_edge == True:
                node_embedding = self.func_glcn(X=node_embedding_obs[:, :n_agent, :], A=edge_index_comm, mini_batch=mini_batch)
                node_embedding = self.func_glcn2(X=node_embedding , A=edge_index_comm, mini_batch=mini_batch)
                return node_embedding
            else:
                fully_connected_edges = [self.fully_connected_edge for _ in range(agent_embedding.shape[0])]
                node_embedding = self.func_glcn(dead_masking= dead_masking, X = agent_embedding, A = edge_index_comm, mini_batch = mini_batch)
                #node_embedding = self.func_glcn2(dead_masking=dead_masking, X=node_embedding, A=A)
                return node_embedding


    def put_data(self, transition):
        self.node_features_list.append(transition[0])
        self.edge_index_enemy_list.append(transition[1])
        self.avail_action_list.append(transition[2])
        self.action_list.append(transition[3])
        self.prob_list.append(transition[4])
        self.action_feature_list.append(transition[5])
        self.reward_list.append(transition[6])
        self.done_list.append(transition[7])
        self.edge_index_comm_list.append(transition[8])
        self.factorize_pi_list.append(transition[9])
        self.dead_masking.append(transition[10])
        self.state_list.append(transition[11])
        self.summarized_state_list.append(transition[12])
        self.agent_feature_list.append(transition[13])


        if transition[7] == True:
            batch_data = (
                self.node_features_list,
                self.edge_index_enemy_list,
                self.avail_action_list,
                self.action_list,
                self.prob_list,
                self.action_feature_list,
                self.reward_list,
                self.done_list,
                self.edge_index_comm_list,
                self.factorize_pi_list,
                self.dead_masking,
                self.state_list,
                self.summarized_state_list,
                self.agent_feature_list
                )

            self.batch_store.append(batch_data) # batch_store에 저장함
            self.node_features_list = list()
            self.edge_index_enemy_list = list()
            self.avail_action_list = list()
            self.action_list = list()
            self.prob_list = list()
            self.action_feature_list = list()
            self.reward_list = list()
            self.done_list = list()
            self.edge_index_comm_list = list()
            self.factorize_pi_list = list()
            self.dead_masking = list()
            self.state_list = list()
            self.summarized_state_list = list()
            self.agent_feature_list = list()


    def make_batch(self, batch_data):
        node_features_list = batch_data[0]
        edge_index_enemy_list = batch_data[1]
        converted_no_attack_masks = batch_data[2]
        action_list = batch_data[3]
        prob_list = batch_data[4]
        action_feature_list = batch_data[5]
        reward_list = batch_data[6]
        done_list = batch_data[7]
        edge_index_comm_list = batch_data[8]
        converted_attack_masks = batch_data[9]
        dead_masking = batch_data[10]
        state_list = batch_data[11]

        summarized_state_list = batch_data[12]
        agent_feature_list = batch_data[13]



        dead_masking = torch.tensor(dead_masking, dtype=torch.float).to(device)
        node_features_list = torch.tensor(node_features_list, dtype = torch.float).to(device)
        edge_index_enemy_list = edge_index_enemy_list
        action_list = torch.tensor(action_list, dtype=torch.float).to(device)
        state_list = torch.tensor(state_list, dtype=torch.float).to(device)
        summarized_state_list = torch.tensor(summarized_state_list, dtype=torch.float).to(device)
        agent_feature_list = torch.stack(agent_feature_list).to(device)
        converted_no_attack_masks = torch.tensor(converted_no_attack_masks, dtype = torch.bool).to(device)
        converted_attack_masks = torch.tensor(converted_attack_masks, dtype=torch.bool).to(device)
        #print(converted_no_attack_masks)
        return node_features_list, edge_index_enemy_list, converted_no_attack_masks,\
               action_list,prob_list,action_feature_list,reward_list,done_list, \
               edge_index_comm_list, converted_attack_masks,dead_masking,state_list, \
               summarized_state_list, agent_feature_list





    def learn(self, cum_loss = 0):

        cum_surr = 0
        cum_value_loss = 0
        cum_lap_quad = 0
        cum_sec_eig_upperbound = 0


        for i in range(self.K_epoch):
            if i == 0:
                v_s_old_list = list()
                v_s_next_old_list = list()
            for l in range(len(self.batch_store)):
                batch_data = self.batch_store[l]
                node_features_list, \
                edge_index_enemy_list, \
                converted_no_attack_masks,\
                action_list,\
                factorize_pi_list,\
                action_feature_list,\
                reward_list,\
                done_list, \
                edge_index_comm_list,  converted_attack_masks, dead_masking, state_list, \
                summarized_state_list, agent_feature_list= self.make_batch(batch_data)
                self.eval_check(eval=False)
                action_feature = torch.tensor(action_feature_list, dtype= torch.float).to(device)
                action_list = torch.tensor(action_list, dtype = torch.long).to(device)

                done = torch.tensor(done_list, dtype = torch.float).to(device)
                reward = torch.tensor(reward_list, dtype= torch.float).to(device)

                factorize_pi_old =torch.tensor(factorize_pi_list, dtype= torch.float).to(device)

                summarized_state = torch.tensor(summarized_state_list, dtype= torch.float).to(device)
                summarized_state =  self.node_representation_sum_state(summarized_state)
                sum_state = summarized_state
                agent_feature=torch.tensor(agent_feature_list, dtype= torch.float).to(device)

                num_nodes = node_features_list.shape[1]

                num_agent = converted_attack_masks.shape[1]
                num_action = action_feature.shape[1]-num_agent+1
                summarized_state_agent_wise = summarized_state.unsqueeze(1).repeat(1, num_agent, 1)
                summarized_state= summarized_state.unsqueeze(1).repeat(1, num_action, 1)

                time_step = node_features_list.shape[0]
                if cfg.given_edge == True:
                    node_embedding = self.get_node_representation_gpo(node_features_list,
                                                                      agent_feature,
                                                                      edge_index_enemy_list,
                                                                      edge_index_comm_list,
                                                                      mini_batch=True,
                                                                      dead_masking=dead_masking,
                                                                      n_agent = num_agent
                                                                      )

                else:
                    node_embedding = self.get_node_representation_gpo(
                                                                            node_features_list,
                                                                            agent_feature,
                                                                            edge_index_enemy_list,
                                                                            edge_index_comm_list,
                                                                            dead_masking = dead_masking,
                                                                            mini_batch=True,
                                                                            n_agent=num_agent
                                                                            )

                node_embedding2 = torch.cat([summarized_state_agent_wise, node_embedding], dim = 2)
                empty = torch.zeros(1, num_agent, node_embedding2.shape[2]).to(device)
                action_feature = action_feature.reshape(time_step*num_nodes, -1).to(device)
                action_embedding = self.action_representation(action_feature)
                action_embedding = action_embedding.reshape(time_step, num_nodes, -1).to(device)

                #action_embedding = action_embedding.reshape(time_step, (num_action+6), -1)

                edge_index_obs2 = list()
                for e in range(len(edge_index_enemy_list)):
                    new_edge_index = torch.tensor(edge_index_enemy_list[e]).long().to(device)
                    edge_index_obs2.append(new_edge_index)


                action_embedding = self.func_action(X=action_embedding, A=edge_index_obs2, mini_batch = True)
                action_embedding = self.func_action2(X=action_embedding, A=edge_index_obs2, mini_batch=True)
                action_embedding = action_embedding[:, num_agent:, ]
                action_embedding = torch.concat([action_embedding, torch.zeros(action_embedding.shape[0], 1, action_embedding.shape[2]).to(device)], dim=1)




                node_embedding2 = node_embedding2[:, :num_agent, :]
                node_embedding_next2 = torch.cat((node_embedding2, empty), dim = 0)[1:, :, :]


                v_s = self.valuenetwork(node_embedding2).squeeze(2)
                v_next = self.valuenetwork(node_embedding_next2).squeeze(2)

                #print(v_s.shape, v_s[-3, :])
                # print(v_next.shape, v_next[:, 0])
                #print('=======================================')

                # v_s = v_s.reshape(time_step, num_agent)
                # v_next = v_next.reshape(time_step, num_agent)


                done =  done.unsqueeze(1).repeat(1, num_agent)
                reward =  reward.unsqueeze(1).repeat(1, num_agent)

                td_target = reward + self.gamma * v_next * (1-done)

                delta = td_target - v_s
                delta = delta.cpu().detach().numpy()
                advantage_lst = []
                advantage = torch.zeros(num_agent)
                # print(delta)
                # print("===================================")
                for delta_t in delta[: :-1]:
                    # print(delta_t)
                    advantage = self.gamma * self.lmbda * advantage + delta_t
                    advantage_lst.append(advantage)
                advantage_lst.reverse()
                if i == 0:

                    v_s_old_list.append(v_s)
                    v_s_next_old_list.append(v_next)


                advantage = torch.stack(advantage_lst).to(device)

                for n in range(num_agent):
                    obs = node_embedding[:, n, :].unsqueeze(1).expand(time_step,  num_action, node_embedding.shape[2])
                    obs_cat_action = torch.concat([summarized_state, obs, action_embedding], dim=2)
                    obs_cat = torch.concat([summarized_state, obs], dim=2)
                    h = self.network_no_attack(obs_cat[:, 0, :]).squeeze(1)
                    h = h.masked_fill(converted_no_attack_masks[:, n, :] == 0, -1e8)

                    prob_no_attack = torch.softmax(h, dim=-1)
                    u_no_attack = action_list[:, n, 0].unsqueeze(1)

                    pi_no_attack = prob_no_attack.gather(1, u_no_attack)


                    u_attack = action_list[:, n, 1].unsqueeze(1)
                    logit = self.network(obs_cat_action).squeeze(2)
                    #print(logit.shape, converted_attack_masks[:, n, :].shape)
                    logit = logit.masked_fill(converted_attack_masks[:, n, :] == 0, -1e8)
                    prob_attack = torch.softmax(logit, dim=-1)  # 에이전트 별 확률
                    pi_attack = prob_attack.gather(1, u_attack)
                    pi = pi_no_attack * pi_attack
                    pi_old = factorize_pi_old[:, n].unsqueeze(1)
                    advantage_i = advantage[:, n].unsqueeze(1)


                    ratio = torch.exp(torch.log(pi) - torch.log(pi_old).detach())  # a/b == exp(log(a)-log(b))
                    #print(pi.shape, pi_old.shape, advantage.shape, advantage_i.shape)
                    #print(ratio.shape, advantage_i.shape)
                    surr1 = ratio * (advantage_i.detach())
                    surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * (advantage_i.detach())
                    #print(entropy.shape)
                    entropy = -(pi * torch.log(pi)).squeeze(1)

                    if n == 0:
                        surr = torch.min(surr1, surr2).squeeze(1) / num_agent #+0.01*entropy/num_agent
                    else:
                        surr += torch.min(surr1, surr2).squeeze(1) / num_agent# +0.01*entropy/num_agent

                val_surr1 = v_s
                #v_ratio = v_s / v_s_old_list[l].detach()

                val_surr2 = torch.clamp(v_s, v_s_old_list[l].detach() - self.eps_clip, v_s_old_list[l].detach() + self.eps_clip)#*v_s_old_list[l].detach()
                #print(val_surr1.mean(1).shape, val_surr2.mean(1).shape)
                value_loss = (val_surr1-td_target.detach()),

                                      # F.huber_loss(val_surr2.mean(1), td_target.detach().mean(1), delta = 10)).mean()

                val1 = huber_loss(val_surr1, td_target.detach(), delta = 10)
                val2 = huber_loss(val_surr2, td_target.detach(), delta = 10)
                value_loss = torch.max(val1, val2).mean()
                #print(val.shape)


                loss = -surr.mean() +0.5* value_loss
                #print(loss.shape)
            #print(np.array([np.linalg.eigh(L[t, :, :].cpu().detach().numpy())[0][1] for t in range(time_step)]))

                if l == 0:
                    cum_loss = loss / self.n_data_parallelism
                else:
                    cum_loss = cum_loss + loss / self.n_data_parallelism

            grad_clip = float(os.environ.get("grad_clip", 10))
            cum_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.eval_params, grad_clip)
            self.optimizer1.step()
            self.optimizer1.zero_grad()
            # if l == 0:
            #     print(second_eigenvalue)

        self.batch_store = list()
        if cfg.given_edge== True:
            return cum_surr, cum_value_loss, 0, 0, 0  # second_eigenvalue
        else:
            return cum_surr, cum_value_loss, cum_lap_quad, cum_sec_eig_upperbound  # second_eigenvalue

    # def load_network(self, file_dir):
    #     print(file_dir)
    #     checkpoint = torch.load(file_dir)
    #     self.network.load_state_dict(checkpoint["network"])
    #     self.node_representation_ship_feature.load_state_dict(checkpoint["node_representation_ship_feature"])
    #     self.func_meta_path.load_state_dict(checkpoint["func_meta_path"])
    #     self.func_meta_path2.load_state_dict(checkpoint["func_meta_path2"])
    #     self.func_meta_path3.load_state_dict(checkpoint["func_meta_path3"])
    #     self.func_meta_path4.load_state_dict(checkpoint["func_meta_path4"])
    #     self.func_meta_path5.load_state_dict(checkpoint["func_meta_path5"])
    #     self.func_meta_path6.load_state_dict(checkpoint["func_meta_path6"])
    #     try:
    #         self.node_representation_wo_graph.load_state_dict(checkpoint["node_representation_wo_graph"])
    #     except KeyError:pass

    def eval_check(self, eval):
        if eval == True:
            self.network_no_attack.eval()
            self.network.eval()
            self.valuenetwork.eval()
            self.obs_net.eval()
            self.node_representation_sum_state.eval()
            self.node_representation_agent.eval()
            self.node_representation.eval()
            self.node_representation_comm.eval()
            self.action_representation.eval()
            self.func_obs.eval()
            self.func_action.eval()
            self.func_action2.eval()
            self.func_glcn.eval()

        else:
            self.network_no_attack.train()
            self.network.train()
            self.valuenetwork.train()
            self.obs_net.train()
            self.node_representation_sum_state.train()
            self.node_representation_agent.train()
            self.node_representation.train()
            self.node_representation_comm.train()
            self.action_representation.train()
            self.func_obs.train()
            self.func_action.train()
            self.func_action2.train()
            self.func_glcn.train()