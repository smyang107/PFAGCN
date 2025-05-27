import torch
from torch_geometric.nn import GCNConv
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from utils_all import scaled_Laplacian
import pandas as pd
from model.FuncGCNgo import Attention_layer as FunctionAttention
from model.FuncGCNgo import cheb_conv_K as FunctionConv
from model.FuncGCNgo import *
import math
from torch.autograd import Variable
from model.GCNDeepgo import Hierarchical_classifer_new
from model.PFGCN import PositionalEncoding

###### 学习序列信息得到图结构 ######
class ProteinAttention(nn.Module):
    def  __init__(self, in_channels, hid_channels, out_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_channels, hid_channels),
            nn.ReLU(),
            nn.Linear(hid_channels, out_channels),
        )
    def forward(self, x):
        ### 输入：(batch,len/batch,feature)
        ### 输出：每个蛋白质对每个蛋白质的邻接矩阵(batch,batch)
        ####方法一
        if x.dtype == torch.float64:
            x = x.float()
        Z = self.model(x)#torch.Size([64, nb_class,feature])-->torch.Size([64, nb_class,feature])
        score = torch.bmm(Z, Z.transpose(1, 2))#torch.Size([64, nb_class,feature])->torch.Size([64, nb_class,nb_class])
        score = torch.sum(score,dim=2) # torch.Size([64, nb_class,nb_class])->torch.Size([64, nb_class])
        score = torch.mm(score, score.t())#torch.Size([64, nb_class])->torch.Size([64, 64])
        #W = torch.sigmoid(score)  # 归一化为0-1权重
        W = F.softmax(score, dim=1).to(torch.float64) # 归一化为0-1权重
        return W

class ProteinConv(nn.Module):
    '''
    K-order chebyshev graph convolution
    时间维度上的图卷积
    '''

    def __init__(self,K,in_channels, out_channels,device):
        '''
        :param K: int
        :param in_channles: int, num of channels in the input sequence
        :param out_channels: int, num of channels in the output sequence
        '''
        super(ProteinConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.DEVICE = device
        self.K =K
        #self.Theta = nn.ParameterList([nn.Parameter(torch.FloatTensor(in_channels, out_channels).to(self.DEVICE)) for _ in range(K)])
        self.Theta=nn.Parameter(torch.FloatTensor(K,in_channels, out_channels).to(self.DEVICE)).to(torch.float64)

    def forward(self, x, spatial_attention):
        '''
        Chebyshev graph convolution operation
        切比雪夫图卷积运算
        :param x: (batch_size,nb_classes, F_in)
        :return: (batch_size,nb_classes, F_out)
        '''
        ################## 写法1 ##################
        graph_signal = x  # (b, nb_classes, F_in)

        T_k_with_at = spatial_attention  # torch.Size([64, 64])
        output = torch.zeros_like(graph_signal).to(self.DEVICE)  # (batch_size, class, F_out)
        for k in range(self.K):
            theta_k = self.Theta[k]  # (in_channels, out_channels)
            graph_signal_1 = torch.reshape(graph_signal, shape=(graph_signal.shape[0], -1))#(b, nb_classes, F_in)->(b, nb_classes*F_in)
            rhs = T_k_with_at.matmul(graph_signal_1)  # (b , b)(b, nb_classes*F_in) = (b,nb_classes*F_in) 因为是左乘，所以多行和为1变为多列和为1，即一行之和为1，进行左乘
            rhs = torch.reshape(rhs,shape=(graph_signal.shape[0],graph_signal.shape[1],-1))#(b,nb_classes*F_in)->(b,nb_classes,F_in)
            rhs = torch.reshape(rhs, shape=(-1, graph_signal.shape[2]))
            # rhs=rhs.to(torch.float64)
            output += torch.reshape(rhs.matmul(theta_k),shape=(graph_signal.shape[0],graph_signal.shape[1],-1))#(b*nb_classes, F_in)(F_in, F_out) = (b,nb_classes, F_out)
        return output

####### 点嵌入模型 #######
class GoFeature(nn.Module):
    def __init__(self, in_features,classes,out_channels):
        super(GoFeature, self).__init__()

        self.atten = SelfAttention(in_features=in_features, out_features=classes).to(torch.float64)
        #######
        #self.conv = nn.Conv1d(in_channels=1, out_channels=out_channels, kernel_size=1).to(torch.float64)
        #self.conv = nn.Conv1d(in_channels=1, out_channels=out_channels, kernel_size=3, stride=1, padding=1).to(torch.float64)
        self.conv1 = nn.Conv1d(in_channels= 1, out_channels=32, kernel_size=3, padding=1).to(torch.float64)
        self.conv2 = nn.Conv1d(in_channels= 32, out_channels = out_channels, kernel_size=3, padding=1).to(torch.float64)
    def forward(self, x):
        ### (batch,feature) -> (batch,class,new_feature)
        x = torch.relu(self.atten(x))# torch.Size([64, 512])->torch.Size([64, nb_class])
        x = x.unsqueeze(dim=-1).permute(0,2,1) #torch.Size([64, nb_class])-->torch.Size([64, nb_class,1])-->torch.Size([64,1,nb_class])
        #x = self.conv(x).permute(0, 2, 1) #torch.Size([64,1,nb_class])->torch.Size([64,new_feature,nb_class])->torch.Size([64,nb_class,new_feature])
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x)).permute(0, 2, 1)
        return x

###### 引入go-go网络和protein-protein网络，间接构建蛋白质到蛋白质的网络
class PFGCNgo_new(torch.nn.Module):
    def __init__(self,args):
        super(PFGCNgo_new, self).__init__()
        self.training = True
        self.dropout = 0.3
        self.node_names = args["node_names"]
        self.go = args["go"]
        self.functions=args["functions"]
        self.func_set = set(self.functions)
        self.Go_id = args["Go_id"]
        self.MAXLEN = args["MAXLEN"]
        self.args = args
        #### 词嵌入层 ####
        #self.position1 = PositionalEncoding(outfea=1 , len = args["MAXLEN"],args=args)
        self.position1 = PositionalEncoding(outfea=1, len=1000, args=args)
        self.position2 = PositionalEncoding(outfea=1, len=256, args=args)
        # self.feature = FeatureModel(max_features=args["Feature"]["max_features"],
        #                             embedding_dims=args["Feature"]["embedding_dims"])
        self.goFeature = GoFeature(in_features=args["GoFeature"]["in_features"],
                                      classes=args["nb_classes"],
                                      out_channels=args["GoFeature"]["out_channels"])
        ######## BatchNorm ########
        self.norm1 = nn.BatchNorm1d(args["nb_classes"]).to(torch.float64)
        self.norm2 = nn.BatchNorm1d(args["nb_classes"]).to(torch.float64)
        self.norm3 = nn.BatchNorm1d(args["nb_classes"]).to(torch.float64)
        self.norm4 = nn.BatchNorm1d(args["GoFeature"]["in_features"]).to(torch.float64)

        self.FunctionAttention = FunctionAttention(args["FunctionAttention"]["in_channels"],
                                         args["FunctionAttention"]["hid_channels"],
                                         args["FunctionAttention"]["out_channels"])

        self.ProteinAttention = ProteinAttention(args["ProteinAttention"]["in_channels"],
                                                 args["ProteinAttention"]["hid_channels"],
                                                 args["ProteinAttention"]["out_channels"])
        self.scaled_adj = scaled_Laplacian
        self.FunctionConv = FunctionConv(args["FunctionConv"]["K"],
                                        args["FunctionConv"]["in_channels"],
                                         args["FunctionConv"]["out_channels"],
                                         args["device"])
        self.ProteinConv = ProteinConv(args["FunctionConv"]["K"],
                                        args["ProteinConv"]["in_channels"],
                                       args["ProteinConv"]["out_channels"],
                                       args["device"])
        # self.hie_class = Hierarchical_classifer_new(args)
        #### 输出层
        self.fuse = nn.Parameter(data=torch.tensor(0.5),requires_grad=True)
        ### 第一种输出层
        self.outlayer1 = nn.Linear(in_features=args["FunctionConv"]["out_channels"]*2, out_features=1).to(torch.float64)
        self.outlayer2 = nn.Linear(in_features=args["nb_classes"], out_features=args["nb_classes"]).to(torch.float64)
        ### 第二种输出层
        #self.outlayer1 = nn.Conv1d(in_channels=args["nb_classes"], out_channels=args["nb_classes"], kernel_size=17).to(torch.float64)
        #self.outlayer2 = nn.Linear(in_features=16, out_features=1).to(torch.float64)

    def forward(self, data,node_features,go_adj, batch_adj):
        # ############################## 串行 +修改gofeature格式 ##############################
        # 输入：(batch,feature) torch.Size([64, 1256])
        # 输出: (batch,class)
        ####### 先进行词嵌入 #########
        # go_adj = torch.tensor(go_adj).to(self.args["device"])
        # inputs1 = data[:, :self.MAXLEN]#torch.Size([64, 1000]) ##input1加入位置编码
        # inputs2 =  data[:, self.MAXLEN:]#torch.Size([64, 256])
        # #feature = self.feature(inputs1)##torch.Size([64, 1000])-->#torch.Size([64, 256])
        # feature = self.position(inputs1)  # 加入位置编码
        # inputs2 = self.position(inputs2) # 加入位置编码
        # merged_1 = torch.concatenate([feature, inputs2],dim=1)  # torch.Size([64, 256])+torch.Size([64, 256])-->torch.Size([64, 512])
        # merged_1 = self.goFeature(merged_1)  # torch.Size([64, 512])-->torch.Size([64, nbclass, feature])
        # #merged_1 = self.norm1(merged_1)
        # ######### 加入图嵌入特征
        # node_features = node_features.unsqueeze(0).expand(data.shape[0], -1, -1) #torch.Size([64, nbclass, feature]s
        # #print(node_features.requires_grad)
        # merged = torch.concat((merged_1 ,node_features),dim=2) #torch.Size([64, nbclass, feature] 32+16=48
        # merged = self.norm1(merged)
        #
        # ############## 分支一：Protein-protein GCN ##############
        # # adj = self.ProteinAttention(merged)  # (batch,batch)-->(batch,batch) torch.Size([64, 64])
        # # batch_adj = batch_adj + torch.softmax(adj, dim=1)
        # #batch_adj = torch.softmax(batch_adj, dim=1)  # 归一化
        # #batch_adj = torch.sigmoid(batch_adj)  # 归一化
        #
        # ######### 图卷积  #########
        # x1 = self.ProteinConv(merged,batch_adj) +merged # ((batch,feature )(batch,batch)->(batch,len,feature) torch.Size([64, 512])->torch.Size([64, 512])
        # x1 = self.norm2(x1) #torch.Size([64, 512])
        # x1 = F.dropout(F.relu(x1), p=self.dropout, training=self.training) #torch.Size([64, 512])
        # #x1 = self.outlayer1(x1) # torch.Size([64, 512])->torch.Size([64, 512])
        #
        # ###########################  串行连接 #############################
        # ############## 分支二：Function-Function GCN ############
        # ## 特征向量拼接 代替 残差连接，这是跳跃连接
        # #new_feature = torch.concat((x1 , merged_1),dim=1) #torch.Size([64, 512])-->torch.Size([64, 1024])
        #
        # ########### 加入图嵌入的节点特征（加入先验知识） ###########
        # # node_features =  self.args["node_features"].to(self.args["device"])
        # # print(node_features)
        # ######### 自适应构建图 #########
        # atten_adj = self.FunctionAttention(x1)#torch.Size([64, nbclass, feature])torch.Size([64, 589, 64])-->得到矩阵 torch.Size([batch, nbclass, nbclass])
        # ### go_ad将大小为torch.Size([nbclass, nbclass])的矩阵扩展为大小为torch.Size([batch, nbclass, nbclass])的矩阵
        # go_adj = go_adj.unsqueeze(0).expand(data.shape[0], -1, -1)
        # go_adj = go_adj + torch.softmax(atten_adj,dim=1) #####torch.Size([batch, nbclass, nbclass])
        # #go_adj = torch.softmax(go_adj,dim=1) ##### torch.Size([batch, nbclass, nbclass]) 归一化
        #
        # ######### 图卷积：汉堡包结构,引入了go-go图卷积  #########
        # #CONV/FC -> BatchNorm -> ReLu(or other activation) -> Dropout -> CONV/FC ->
        # x2 =self.FunctionConv(merged, go_adj)+ merged#torch.Size([64, nbclass, feature])torch.Size([nbclass, nbclass])-> torch.Size([64, nbclass, new_feature])
        # x2 = self.norm3(x2)
        # x2 = F.dropout(F.relu(x2), p=self.dropout, training=self.training)
        #
        # ############## 输出层 ##############
        # ### 结合输出：
        # #output = self.fuse * x1 + (1-self.fuse) * x2
        # output = self.outlayer2(x2).squeeze(dim=-1)  # torch.Size([64, nbclass, new_feature])->torch.Size([64, nbclass, 1])->torch.Size([64, nbclass])
        # #output = self.outlayer2(self.outlayer1(x2)).squeeze(dim=-1)
        # return output
        # #return F.softmax(output, dim=1)
        # #return torch.sigmoid(output)
        # #return F.log_softmax(output, dim=1)

        ############################## 并行 +删除feature网络 + 修改gofeature格式 ##############################
        # 输入：(batch,feature) torch.Size([64, 1256])
        # 输出: (batch,class)
        ####### 先进行词嵌入 #########
        go_adj = torch.tensor(go_adj).to(self.args["device"])
        inputs1 = data[:, :self.MAXLEN]  # torch.Size([64, 1000]) ##input1加入位置编码
        inputs2 = data[:, self.MAXLEN:]  # torch.Size([64, 256])
        #feature = self.feature(inputs1)  ##torch.Size([64, 1000])-->#torch.Size([64, 256])
        feature = self.position1(inputs1)  # 加入位置编码
        inputs2 = self.position2(inputs2)  # 加入位置编码
        merged_1 = torch.concatenate([feature, inputs2],dim=1)  # torch.Size([64, 256])+torch.Size([64, 256])-->torch.Size([64, 512])
        merged = self.goFeature(merged_1)  # torch.Size([64, 512])-->torch.Size([64, nbclass, feature])
        ##################  加入图嵌入特征  ##################
        node_features = node_features.unsqueeze(0).expand(data.shape[0], -1, -1)  # torch.Size([64, nbclass, feature]s
        # print(node_features.requires_grad)
        merged = torch.concat((merged, node_features), dim=2)  # torch.Size([64, nbclass, feature] 32+16=48
        merged = self.norm1(merged)

        ############## 分支一：Protein-protein GCN ##############

        # merged_1 = self.norm1(merged_1)
        adj = self.ProteinAttention(merged)  # (batch,batch)-->(batch,batch) torch.Size([64, 64])
        batch_adj = batch_adj + torch.softmax(adj, dim=1)
        batch_adj = torch.softmax(batch_adj, dim=1)  # 归一化
        batch_adj = torch.sigmoid(batch_adj)  # 归一化

        ######### 图卷积  #########
        x1 = self.ProteinConv(merged,batch_adj)# ((batch,feature )(batch,batch)->(batch,len,feature) torch.Size([64, 512])->torch.Size([64, 512])
        x1 = self.norm2(x1)  # torch.Size([64, 512])
        x1 = F.dropout(F.relu(x1), p = self.dropout, training=self.training)  # torch.Size([64, 512])
        # x1 = self.outlayer1(x1) # torch.Size([64, 512])->torch.Size([64, 512])

        ############## 分支二：Function-Function GCN ############
        ## 特征向量拼接 代替 残差连接，这是跳跃连接
        # new_feature = torch.concat((x1 , merged_1),dim=1) #torch.Size([64, 512])-->torch.Size([64, 1024])

        ########### 加入图嵌入的节点特征（加入先验知识） ###########
        # node_features =  self.args["node_features"].to(self.args["device"])
        # print(node_features)
        ######### 自适应构建图 #########
        atten_adj = self.FunctionAttention(merged)  # torch.Size([64, nbclass, feature])torch.Size([64, 589, 64])-->得到矩阵 torch.Size([batch, nbclass, nbclass])
        ### go_ad将大小为torch.Size([nbclass, nbclass])的矩阵扩展为大小为torch.Size([batch, nbclass, nbclass])的矩阵
        go_adj = go_adj.unsqueeze(0).expand(data.shape[0], -1, -1)
        go_adj = go_adj + torch.softmax(atten_adj, dim=1)  #####torch.Size([batch, nbclass, nbclass])
        # go_adj = torch.softmax(go_adj,dim=1) ##### torch.Size([batch, nbclass, nbclass]) 归一化

        ######### 图卷积：汉堡包结构,引入了go-go图卷积  #########
        # CONV/FC -> BatchNorm -> ReLu(or other activation) -> Dropout -> CONV/FC ->
        x2 = self.FunctionConv(merged,go_adj) # torch.Size([64, nbclass, feature])torch.Size([nbclass, nbclass])-> torch.Size([64, nbclass, new_feature])
        x2 = self.norm3(x2)
        x2 = F.dropout(F.relu(x2), p=self.dropout, training=self.training)

        ############## 分支三：分层分类 ############
        # merged_1 = self.norm4(merged_1)
        # x3 = self.hie_class(merged_1)

        # ############## 输出层 ##############
        # ### 结合输出：
        #output = self.fuse * x1 + (1-self.fuse) * x2
        output = torch.concat((x1,x2),dim=2)
        output = F.relu(self.outlayer1(output).squeeze(dim=-1))  # torch.Size([64, nbclass, new_feature])->torch.Size([64, nbclass, 1])->torch.Size([64, nbclass])
        output = self.outlayer2(output)
        return output
        # return torch.sigmoid(output)
        # return F.log_softmax(output, dim=1)
        # return F.softmax(output, dim=1)