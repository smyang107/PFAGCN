import torch
from torch_geometric.nn import GCNConv
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from utils_all import get_parents,get_gene_ontology,get_function_node,get_node_name
import pandas as pd

####### 对比实验
class ProteInfer(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=21, out_channels=32,  kernel_size=33,dilation=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=33, dilation=4)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=1, kernel_size=33, dilation=8)
        self.fc = nn.Linear(in_features=512, out_features=args["nb_classes"])
        #self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        '''
        输入：蛋白质序列 (batch,seq)
        输出：类别 (batch,classes)
        '''
        x = F.one_hot(x.to(torch.int64),num_classes=21).permute(0,2,1).float() #torch.Size([64, 2003])
        # 20中氨基酸的one-hot编码, torch.Size([64, 1002,20])->torch.Size([64,20,1002])
        #x = PackedSequence(x, batch_sizes=seq_len) # 维度为batch_size x seq_len x 20
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.squeeze(1)
        #x, _ = nn.utils.rnn.pad_packed_sequence(x)
        #由于蛋白质序列长度差异巨大,需要使用能处理可变长输入的池化方法,而不是简单的平均池化。# 替换平均池化为AdaptiveAvgPool1d
        x = F.adaptive_avg_pool1d(x, output_size=512) # 可变长池化
        x = self.fc(x)
        #x = self.sigmoid(x)
        return x

# ####### 对比实验
# class ProteInfer(nn.Module):
#     def __init__(self,args):
#         super().__init__()
#
#         self.residual_blocks = nn.ModuleList()
#         for _ in range(args["num_blocks"]):
#             self.residual_blocks.append(self.make_residual_block(channels=32))
#         self.fc = nn.Linear(in_features=512, out_features=args["nb_classes"])
#         #self.Embed = nn.Embedding(embedding_dim=args["nb_classes"])
#     def make_residual_block(self, channels):
#         return nn.Sequential(
#             nn.Conv1d(in_channels=21, out_channels=32,  kernel_size=33,dilation=2),
#             nn.ReLU(inplace=True),
#             nn.Conv1d(in_channels=32, out_channels=32, kernel_size=33, dilation=4),
#             nn.ReLU(inplace=True),
#             nn.Conv1d(in_channels=32, out_channels=1, kernel_size=33, dilation=8),
#             nn.ReLU(inplace=True),
#         )
#     def forward(self, x):
#         '''
#         输入：蛋白质序列 (batch,seq)
#         输出：类别 (batch,classes)
#         '''
#         x = F.one_hot(x.to(torch.int64),num_classes=21).permute(0,2,1).float() #torch.Size([64, 2003])
#         # 20中氨基酸的one-hot编码, torch.Size([64, 1002,20])->torch.Size([64,20,1002])
#         residual = x
#         out = x
#         i = 0
#         for block in self.residual_blocks:
#             out = block(out)
#             if i > 0 :
#                 out += residual
#             out = torch.relu(out)
#             residual = out
#             i = i + 1
#         out = out.squeeze(1)
#
#         x = F.adaptive_avg_pool1d(out, output_size=512) # 可变长池化
#         x = self.fc(x)
#         #x = self.sigmoid(x)
#         return x