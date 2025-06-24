import torch
import torch.nn as nn
import dgl.nn as dglnn
import dgl.function as fn
from torch.nn import Parameter
import torch.nn.functional as F
from utils.efficient_kan import KAN


class MLP_Encoder(nn.Module):
    def __init__(self, n_hidden, n_layers, activation, dropout):
        super(MLP_Encoder, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        for i in range(n_layers):
            self.layers.append(nn.Linear(n_hidden, n_hidden))
            self.layers.append(nn.ReLU())
        # output layer
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features):
        h = features
        for i in range(int(len(self.layers) / 2)):
            if (2 * i) != 0:
                h = self.dropout(h)
            h = self.layers[2 * i + 1](self.layers[2 * i](h)) + features  #
            # h = layer(g, h) # 0817-GCN
        return h + features


class GCN_Encoder(nn.Module):
    def __init__(self, n_hidden, n_layers, activation, dropout):
        super(GCN_Encoder, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        for i in range(n_layers):
            self.layers.append(
                dglnn.GraphConv(n_hidden, n_hidden, activation=activation)
            )
        # output layer
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h) + features  # 0816-GCN
            # h = layer(g, h) # 0817-GCN
        return h + features


class DAGNN_Encoder(nn.Module):
    def __init__(self, n_hidden, n_layers):
        super(DAGNN_Encoder, self).__init__()
        self.s = Parameter(torch.FloatTensor(n_hidden, 1))
        self.k = n_layers

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("sigmoid")
        nn.init.xavier_uniform_(self.s, gain=gain)

    def forward(self, graph, feats):
        with graph.local_scope():
            results = [feats]

            degs = graph.in_degrees().float()
            norm = torch.pow(degs, -0.5)
            norm = norm.to(feats.device).unsqueeze(1)

            for _ in range(self.k):
                feats = feats * norm
                graph.ndata["h"] = feats
                graph.update_all(fn.copy_u("h", "m"), fn.sum("m", "h"))
                feats = graph.ndata["h"]
                feats = feats * norm
                results.append(feats)

            H = torch.stack(results, dim=1)
            S = F.sigmoid(torch.matmul(H, self.s))
            S = S.permute(0, 2, 1)
            H = torch.matmul(S, H).squeeze()

            return H + feats


class KANEncoder(nn.Module):
    def __init__(self, n_hidden, n_layers, activation, dropout):
        super(KANEncoder, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        for i in range(n_layers):
            self.layers.append(KAN([n_hidden, n_hidden]))
            self.layers.append(nn.ReLU())
        # output layer
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features):
        h = features
        for i in range(int(len(self.layers) / 2)):
            if (2 * i) != 0:
                h = self.dropout(h)
            h = self.layers[2 * i + 1](self.layers[2 * i](h)) + features  #
            # h = layer(g, h) # 0817-GCN
        return h + features
