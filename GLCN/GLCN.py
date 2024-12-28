import numpy as np
import torch
import torch.nn as nn
import os
import torch.nn.functional as F
from torch import Tensor
from collections import OrderedDict
import sys
sys.path.append("..")  # 상위 폴더를 import할 수 있도록 경로 추가
from cfg import get_cfg
from GTN.inits import glorot
cfg = get_cfg()
print(torch.cuda.device_count())
device =torch.device(cfg.cuda if torch.cuda.is_available() else "cpu")
print(device)




def gumbel_sigmoid(logits: Tensor, tau: float = 1, hard: bool = False, threshold: float = 0.5) -> Tensor:
    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )

    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits, tau)
    y_soft = gumbels.sigmoid()


    if hard:
        # Straight through.

        indices = (y_soft > threshold).nonzero(as_tuple=True)
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format)
        y_hard[indices[0], indices[1]] = 1.0
        ret = y_hard - y_soft.detach() + y_soft

    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


def weight_init_xavier_uniform(submodule):
    if isinstance(submodule, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(submodule.weight)
        submodule.bias.data.fill_(0.01)
    if isinstance(submodule, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(submodule.weight)
    elif isinstance(submodule, torch.nn.BatchNorm2d):
        submodule.weight.data.fill_(1.0)
        submodule.bias.data.zero_()


class GLCN(nn.Module):
    def __init__(self, feature_size, graph_embedding_size, link_prediction = True, feature_obs_size = None, skip_connection = False):
        super(GLCN, self).__init__()
        self.graph_embedding_size = graph_embedding_size
        self.link_prediction = link_prediction

        self.Ws = nn.Parameter(torch.Tensor(feature_size, graph_embedding_size))
        glorot(self.Ws)
        self.a = nn.Parameter(torch.empty(size=(2 * graph_embedding_size, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)





    def _link_prediction(self, h):

        h = h.detach()
        #h = h[:, :self.feature_obs_size]
        h = torch.einsum("ijk,kl->ijl", torch.abs(h.unsqueeze(1) - h.unsqueeze(0)), self.a_link)
        h = h.squeeze(2)
        A = gumbel_sigmoid(h, tau = float(os.environ.get("gumbel_tau",1)), hard = True, threshold = 0.5)
        D = torch.diag(torch.diag(A))
        A = A-D
        I = torch.eye(A.size(0)).to(device)
        A = A+I

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
            e = Wh1 + Wh2.T

        return F.leaky_relu(e, negative_slope=cfg.negativeslope)

    def forward(self, A, X, mini_batch = False, dead_masking = None):
        if self.link_prediction == False:
            if mini_batch == False:
                E = A.to(device)
                num_nodes = X.shape[0]
                #print(num_nodes)
                E = torch.sparse_coo_tensor(E, torch.ones(torch.tensor(E).shape[1]).to(device), (num_nodes, num_nodes)).long().to(device).to_dense()
                Wh = X @ self.Ws
                a = self._prepare_attentional_mechanism_input(Wh, Wh)
                zero_vec = -9e15 * torch.ones_like(E)
                a = torch.where(E > 0, a, zero_vec)
                a = F.softmax(a, dim = 1)

                H = F.elu(torch.matmul(a, Wh))
            else:
                batch_size = X.shape[0]
                num_nodes = X.shape[1]
                Hs = torch.zeros([batch_size, num_nodes, self.graph_embedding_size]).to(device)

                for b in range(batch_size):
                    X_t = X[b,:,:]
                    E = torch.tensor(A[b]).long().to(device)
                    E = torch.sparse_coo_tensor(E, torch.ones(torch.tensor(E).shape[1]).to(device), (num_nodes, num_nodes)).long().to(device).to_dense()
                    Wh = X_t @ self.Ws
                    a = self._prepare_attentional_mechanism_input(Wh, Wh)
                    zero_vec = -9e15 * torch.ones_like(E)
                    a = torch.where(E > 0, a, zero_vec)
                    a = F.softmax(a, dim = 1)
                    H = F.elu(torch.matmul(a, Wh))
                    Hs[b, :, :] = H
                H = Hs

            return H
        else:
            if mini_batch == False:
                A = self._link_prediction(X)
                H = X
                for k in range(self.k_hop):
                    Wh = H @ self.Ws[k]
                    a = self._prepare_attentional_mechanism_input(Wh, Wh, k=k)
                    zero_vec = -9e15 * torch.ones_like(A)
                    a = torch.where(A > 0, A * a, zero_vec)
                    a = F.softmax(a, dim=1)
                    H = F.elu(torch.matmul(a, Wh))
                return H, A, X
            else:
                num_nodes = X.shape[1]
                batch_size = X.shape[0]
                Hs = torch.zeros([batch_size, num_nodes, self.graph_embedding_size]).to(device)
                As = torch.zeros([batch_size, num_nodes, num_nodes]).to(device)
                for b in range(batch_size):
                    A = self._link_prediction(X[b])
                    As[b, :, :] = A
                    H = X[b, :, :]
                    for k in range(self.k_hop):
                        if k != 0:
                            A = A.detach()
                        Wh = H @ self.Ws[k]
                        a = self._prepare_attentional_mechanism_input(Wh, Wh, k = k)
                        zero_vec = -9e15 * torch.ones_like(A)
                        a = torch.where(A > 0, A*a, zero_vec)
                        a = F.softmax(a, dim=1)
                        H = F.elu(torch.matmul(a, Wh))
                        if k+1 == self.k_hop:
                            Hs[b,:, :] = H
                return Hs, As, X, 1