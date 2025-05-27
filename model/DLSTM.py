import torch
from torch_geometric.nn import GCNConv
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
#from utils_all import get_parents,get_gene_ontology,get_function_node,get_node_name
#from utils_all import *
import pandas as pd

# 定义Deep LSTM模型
class DeepLSTM(nn.Module):
    def __init__(self, args):
        super(DeepLSTM, self).__init__()

        self.lstm = nn.LSTM(args["MAXLEN"], args["hidden_size1"], args["num_layers"], batch_first=True,bidirectional=True).to(torch.float64)
        self.emb = nn.Linear(args["hidden_size2"], 1).to(torch.float64)
        self.fc = nn.Linear(args["hidden_size1"]*2, args["nb_classes"]).to(torch.float64)

    def forward(self, x):
        # x shape (batch, seq_len, input_size)
        x = F.one_hot(x.to(torch.int64), num_classes=21).to(torch.float64)  # (batch,1000,21)
        x = x.permute(0,2,1) #(batch,1000,21)->(batch,21,1000)
        x, _ = self.lstm(x) #(batch,21,1000)->(batch,21,512*2)
        x = self.emb(x.permute(0,2,1)).squeeze() #(batch,21,512*2)->(batch,512*2,21)->(batch,512*2,1)
        x = self.fc(x)
        return x

