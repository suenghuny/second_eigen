import numpy as np
import torch
import torch.nn as nn
import os
import torch.nn.functional as F

from collections import OrderedDict
import sys
sys.path.append("..")  # 상위 폴더를 import할 수 있도록 경로 추가
from cfg import get_cfg
from GTN.inits import glorot
cfg = get_cfg()
print(torch.cuda.device_count())
device =torch.device(cfg.cuda if torch.cuda.is_available() else "cpu")
print(device)


def sample_adjacency_matrix(weight_matrix):
    # weight_matrix는 n x n 텐서이며, 각 원소는 연결 확률을 나타냅니다.
    # 0과 1 사이의 uniform random matrix를 생성합니다.
    random_matrix = torch.rand(weight_matrix.size()).to(device)

    # weight_matrix의 확률 값을 사용하여 0 또는 1을 샘플링합니다.
    # weight_matrix의 각 원소가 해당 위치에서의 연결 확률을 나타내므로,
    # random_matrix가 그 확률 이하인 경우에는 연결(1)로, 그렇지 않으면 비연결(0)으로 판단합니다.
    adjacency_matrix = (random_matrix < weight_matrix).int()
    #print(adjacency_matrix)
    return adjacency_matrix

class StraightThroughEstimator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        sampled = (input > torch.rand(input.shape).to(input.device)).float()
        return sampled

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output을 그대로 통과시킴
        return grad_output


def gumbel_sigmoid_sample(logits, tau=float(os.environ.get("gumbel_tau",0.1)), eps=1e-10):
    # Gumbel(0, 1) 노이즈 생성
    U = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(U + eps) + eps)
    # 로짓에 Gumbel 노이즈 추가
    gumbel_logits = logits + gumbel_noise
    # Sigmoid 함수 적용
    y = torch.sigmoid(gumbel_logits / tau)
    return y


def weight_init_xavier_uniform(submodule):
    if isinstance(submodule, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(submodule.weight)
        submodule.bias.data.fill_(0.01)
    if isinstance(submodule, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(submodule.weight)
    elif isinstance(submodule, torch.nn.BatchNorm2d):
        submodule.weight.data.fill_(1.0)
        submodule.bias.data.zero_()

class NodeEmbedding(nn.Module):
    def __init__(self, feature_size, n_representation_obs, layers = [20, 30 ,40]):
        super(NodeEmbedding, self).__init__()
        self.feature_size = feature_size
        self.linears = OrderedDict()
        last_layer = self.feature_size
        for i in range(len(layers)):
            layer = layers[i]
            if i <= len(layers)-2:
                self.linears['linear{}'.format(i)]= nn.Linear(last_layer, layer)
                self.linears['batchnorm{}'.format(i)] = nn.BatchNorm1d(layer)
                self.linears['activation{}'.format(i)] = nn.ELU()
                last_layer = layer
            else:
                self.linears['linear{}'.format(i)] = nn.Linear(last_layer, n_representation_obs)
        self.node_embedding = nn.Sequential(self.linears)
        self.node_embedding.apply(weight_init_xavier_uniform)


    def forward(self, node_feature, missile=False):
        node_representation = self.node_embedding(node_feature)
        return node_representation


class GLCN(nn.Module):
    def __init__(self, feature_size, graph_embedding_size, link_prediction = True, feature_obs_size = None, skip_connection = False):
        super(GLCN, self).__init__()
        self.graph_embedding_size = graph_embedding_size
        self.link_prediction = link_prediction
        if self.link_prediction == True:
            self.feature_obs_size = feature_obs_size
            self.a_link = nn.Parameter(torch.empty(size=(self.feature_obs_size, 1)))
            nn.init.xavier_uniform_(self.a_link.data, gain=1.414)
            self.k_hop = int(os.environ.get("k_hop",2))
            self.sampling = bool(os.environ.get("sampling", True))
            self.skip_connection = skip_connection

            if self.skip_connection == True:
                graph_embedding_size = feature_size
                self.graph_embedding_size = feature_size

            if self.sampling == True:
                self.Ws = [nn.Parameter(torch.Tensor(feature_size, graph_embedding_size)) if k == 0 else nn.Parameter(torch.Tensor(size=(graph_embedding_size, graph_embedding_size))) for k in range(self.k_hop)]
                [glorot(W) for W in self.Ws]

                self.Wv = [nn.Parameter(torch.Tensor(feature_size, graph_embedding_size)) if k == 0 else nn.Parameter(torch.Tensor(size=(graph_embedding_size, graph_embedding_size))) for k in range(self.k_hop)]
                [glorot(W) for W in self.Wv]

                self.Wq = [nn.Parameter(torch.Tensor(feature_size, graph_embedding_size)) if k == 0 else nn.Parameter(torch.Tensor(size=(graph_embedding_size, graph_embedding_size))) for k in range(self.k_hop)]
                [glorot(W) for W in self.Wq]

                self.a = [nn.Parameter(torch.empty(size=(2 * graph_embedding_size, 1))) if k == 0 else nn.Parameter(torch.empty(size=(2 * graph_embedding_size, 1))) for k in range(self.k_hop)]
                [nn.init.xavier_uniform_(self.a[k].data, gain=1.414) for k in range(self.k_hop)]

                self.Ws = nn.ParameterList(self.Ws)
                self.Wv = nn.ParameterList(self.Wv)
                self.Wq = nn.ParameterList(self.Wq)
                self.a = nn.ParameterList(self.a)
            else:
                self.W = [nn.Parameter(torch.Tensor(size=(feature_size, graph_embedding_size))) if k == 0 else nn.Parameter(torch.Tensor(size=(graph_embedding_size, graph_embedding_size))) for k in range(self.k_hop)]
                [glorot(W) for W in self.W]
                self.W = nn.ParameterList(self.W)
        else:
            self.Ws = nn.Parameter(torch.Tensor(feature_size, graph_embedding_size))
            glorot(self.Ws)

            self.Wv = nn.Parameter(torch.Tensor(feature_size, graph_embedding_size))
            glorot(self.Wv)

            self.Wq = nn.Parameter(torch.Tensor(feature_size, graph_embedding_size))
            glorot(self.Wq)
            self.a = nn.Parameter(torch.empty(size=(2 * graph_embedding_size, 1)))
            nn.init.xavier_uniform_(self.a.data, gain=1.414)





    def _link_prediction(self, h, dead_masking, mini_batch = False):
        if cfg.softmax == True:
            h = h.detach()
            h = torch.einsum("ijk,kl->ijl", torch.abs(h.unsqueeze(1) - h.unsqueeze(0)), self.a_link)
            h = h.squeeze(2)
            A = F.softmax(h, dim = 1)
            A = (A+A.T)/2
        else:
            h = h.detach()
            h = h[:, :self.feature_obs_size]
            h = torch.einsum("ijk,kl->ijl", torch.abs(h.unsqueeze(1) - h.unsqueeze(0)), self.a_link)
            h = h.squeeze(2)
            if self.sampling == True:
                A = gumbel_sigmoid_sample(h)
            else:
                A = F.sigmoid(h)
            D = torch.diag(torch.diag(A))
            A = A-D
            if self.sampling ==True:
                I = torch.eye(A.size(0)).to(device)
                A = A+I
            else:
                A = A
        return A





    def _prepare_attentional_mechanism_input(self, Wq, Wv, k = None):
        if k == None:
            Wh1 = Wq
            Wh1 = torch.matmul(Wh1, self.a[:self.graph_embedding_size, : ])
            Wh2 = Wv
            Wh2 = torch.matmul(Wh2, self.a[self.graph_embedding_size:, :])
            e = Wh1 + Wh2.T
        else:
            Wh1 = Wq
            Wh1 = torch.matmul(Wh1, self.a[k][:self.graph_embedding_size, : ])
            Wh2 = Wv
            Wh2 = torch.matmul(Wh2, self.a[k][self.graph_embedding_size:, :])
            e = Wh1 @ Wh2.T
        return F.leaky_relu(e, negative_slope=cfg.negativeslope)



    def forward(self, A, X, dead_masking = False, mini_batch = False):
        if self.link_prediction == False:
            if mini_batch == False:
                E = A.to(device)
                num_nodes = X.shape[0]
                E = torch.sparse_coo_tensor(E, torch.ones(torch.tensor(E).shape[1]).to(device), (num_nodes, num_nodes)).long().to(device).to_dense()
                Wh = X @ self.Ws
                Wq = X @ self.Wq
                Wv = X @ self.Wv
                a = self._prepare_attentional_mechanism_input(Wq, Wv)
                zero_vec = -9e15 * torch.ones_like(E)
                a = torch.where(E > 0, a, zero_vec)
                a = F.softmax(a, dim = 1)
                H = F.relu(torch.matmul(a, Wh))
                #print("뒹벳",H) #
            else:
                batch_size = X.shape[0]
                num_nodes = X.shape[1]
                H_placeholder = list()
                for b in range(batch_size):
                    X_t = X[b,:,:]
                    E = torch.tensor(A[b]).long().to(device)
                    E = torch.sparse_coo_tensor(E, torch.ones(torch.tensor(E).shape[1]).to(device), (num_nodes, num_nodes)).long().to(device).to_dense()
                    Wh = X_t @ self.Ws
                    Wq = X_t @ self.Wq
                    Wv = X_t @ self.Wv
                    a = self._prepare_attentional_mechanism_input(Wq, Wv)
                    zero_vec = -9e15 * torch.ones_like(E)
                    a = torch.where(E > 0, a, zero_vec)
                    a = F.softmax(a, dim = 1)
                    H = F.relu(torch.matmul(a, Wh))
                    H_placeholder.append(H)
                H = torch.stack(H_placeholder)

            return H
        else:
            if mini_batch == False:
                A = self._link_prediction(X, dead_masking, mini_batch = mini_batch)
                if self.sampling == True:
                    H = X
                    for k in range(self.k_hop):
                        X_past = H
                        Wh = H @ self.Ws[k]
                        Wq = H @ self.Wq[k]
                        Wv = H @ self.Wv[k]
                        a = self._prepare_attentional_mechanism_input(Wq, Wv, k=k)
                        zero_vec = -9e15 * torch.ones_like(A)
                        a = torch.where(A > 0.01, a, zero_vec)
                        a = F.softmax(a, dim=1)
                        H = F.relu(torch.matmul(a, Wh))
                        if self.skip_connection == True:
                            H = H + X_past
                else:
                    I = torch.eye(A.size(0)).to(device)
                    A_hat = A + I
                    D_hat_diag = torch.sum(A_hat, dim=0)
                    D_hat_inv_sqrt_diag = torch.pow(D_hat_diag, -0.5)
                    D_hat_inv_sqrt = torch.diag(D_hat_inv_sqrt_diag)
                    for k in range(self.k_hop):
                        if k == 0:
                            support = torch.mm(X, self.W[k])
                            output = torch.mm(torch.mm(D_hat_inv_sqrt, A_hat), D_hat_inv_sqrt)
                        else:
                            support = torch.mm(H, self.W[k])
                            output = torch.mm(torch.mm(D_hat_inv_sqrt, A_hat), D_hat_inv_sqrt)
                        H = F.relu(torch.mm(output, support))

                return H, A, X
            else:
                num_nodes = X.shape[1]
                batch_size = X.shape[0]
                I = torch.eye(num_nodes).to(device)
                H_placeholder = list()
                A_placeholder = list()
                D_placeholder = list()
                for b in range(batch_size):
                    A = self._link_prediction(X[b], dead_masking[b], mini_batch = mini_batch)
                    A_placeholder.append(A)
                    D = torch.diag(torch.diag(A))
                    D_placeholder.append(D)
                    if self.sampling == True:
                        H = X[b, :, :]
                        for k in range(self.k_hop):
                            if k != 0:
                                A = A.detach()
                            X_past = H
                            Wh = H @ self.Ws[k]
                            Wq = H @ self.Wq[k]
                            Wv = H @ self.Wv[k]
                            a = self._prepare_attentional_mechanism_input(Wq, Wv, k = k)
                            zero_vec = -9e15 * torch.ones_like(A)
                            a = torch.where(A > 0.01, a, zero_vec)
                            a = F.softmax(a, dim=1)
                            H = F.relu(torch.matmul(a, Wh))
                            if self.skip_connection == True:
                                H = H + X_past
                            if k+1 == self.k_hop:
                                H_placeholder.append(H)
                    else:
                        A_hat = A + I
                        D_hat_diag = torch.sum(A_hat, dim=1)
                        D_hat_inv_sqrt_diag = torch.pow(D_hat_diag, -0.5)
                        D_hat_inv_sqrt = torch.diag(D_hat_inv_sqrt_diag)
                        for k in range(self.k_hop):
                            if k == 0:
                                support = torch.mm(X[b], self.W[k])
                                output = torch.mm(torch.mm(D_hat_inv_sqrt, A_hat), D_hat_inv_sqrt)
                                H = F.relu(torch.mm(output, support))
                            else:
                                support = torch.mm(H, self.W[k])
                                output = torch.mm(torch.mm(D_hat_inv_sqrt, A_hat), D_hat_inv_sqrt)
                                H = F.relu(torch.mm(output.detach(), support))
                            if k+1 == self.k_hop:
                                H_placeholder.append(H)
                H = torch.stack(H_placeholder)
                A = torch.stack(A_placeholder)
                D = torch.stack(D_placeholder)
                return H, A, X, D

