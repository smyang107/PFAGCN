import torch
from torch_geometric.nn import GCNConv
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from utils_all import scaled_Laplacian
import pandas as pd
from model.FuncGCNgo import Attention_layer as FunctionAttention
from model.GCNDeepgo import Attention_layer as ProteinAttention
from model.GCNDeepgo import cheb_conv_K as ProteinConv
from model.FuncGCNgo import cheb_conv_K as FunctionConv
from model.FuncGCNgo import *
import math
from torch.autograd import Variable
from model.GCNDeepgo import Hierarchical_classifer_new


####位置编码
class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, outfea, len, args):
        super(PositionalEncoding, self).__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(len, outfea).to(args["device"])
        position = torch.arange(0, len).unsqueeze(1)  # 这里表示i=[0,1,2,3..n-1]
        div_term = torch.exp(torch.arange(0, outfea, 2) *-(math.log(10000.0) / outfea))
        pe[:, 0::2] = torch.sin(position * div_term)  # t为偶数，0，2，4，6.。。
        pe[:, 1::2] = torch.cos(position * div_term)  # t为奇数，1，3，5
        pe = pe.unsqueeze(0)  # [1,T,F]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 我们通过将Ht[i，：]与位置编码组合在一起，生成新的表示H't[i：]
        # 位置编码：Variable(self.pe,requires_grad=False)
        # 相同点：Variable就是 变量 的意思，Tensor 是张量的意思（也就是多维矩阵）。两者都是用于存放数据的。
        # tensor不能反向传播，variable可以反向传播（可学习参数）
        x = x + Variable(self.pe, requires_grad=False).squeeze(-1)
        # Variable(self.pe,requires_grad=True)
        return x

###### 引入go-go网络和protein-protein网络，间接构建蛋白质到蛋白质的网络
class PFGCNgo(torch.nn.Module):
    def __init__(self,args):
        super(PFGCNgo, self).__init__()
        self.training = True
        self.dropout = 0.2
        self.node_names = args["node_names"]
        self.go = args["go"]
        self.functions=args["functions"]
        self.func_set = set(self.functions)
        self.Go_id = args["Go_id"]
        self.MAXLEN = args["MAXLEN"]
        self.args = args
        # #这里加入自适应调整
        # if args["addaptadj"]:
        #     if args["supports"] is None:
        #         self.supports = []
        #     self.nodevec1 = nn.Parameter(torch.randn(args["batch_size"], 10).to(args["device"]), requires_grad=True).to(args["device"])
        #     self.nodevec2 = nn.Parameter(torch.randn(10, args["batch_size"]).to(args["device"]), requires_grad=True).to(args["device"])
        #### 词嵌入层 ####
        self.position1 = PositionalEncoding(outfea=1 , len=1000,args=args)
        self.position2 = PositionalEncoding(outfea=1, len=256, args=args)
        self.feature = FeatureModel(max_features=args["Feature"]["max_features"],
                                    embedding_dims=args["Feature"]["embedding_dims"])
        self.goFeature = GoFeature(in_features=args["GoFeature"]["in_features"],
                                      classes=args["nb_classes"],
                                      out_channels=args["GoFeature"]["out_channels"])
        #### BatchNorm
        self.norm1 = nn.BatchNorm1d(args["GoFeature"]["in_features"]).to(torch.float64)
        self.norm2 = nn.BatchNorm1d(args["ProteinConv"]["in_channels"]).to(torch.float64)
        self.norm3 = nn.BatchNorm1d(args["nb_classes"]).to(torch.float64)
        self.norm4 = nn.BatchNorm1d(args["nb_classes"]).to(torch.float64)

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
        #self.hie_class = Hierarchical_classifer_new(args)
        #### 输出层
        self.fuse = nn.Parameter(data=torch.tensor(0.5),requires_grad=True)
        ### 第一种输出层
        #self.outlayer1 = nn.Linear(in_features=args["ProteinConv"]["out_channels"], out_features=args["nb_classes"]).to(torch.float64)
        self.outlayer2 = nn.Linear(in_features=args["FunctionConv"]["out_channels"], out_features=1).to(torch.float64)
        #self.outlayer3 = nn.Linear(in_features=args["nb_classes"], out_features=args["nb_classes"]).to(torch.float64)
        ### 第二种输出层
        #self.outlayer1 = nn.Conv1d(in_channels=args["nb_classes"], out_channels=args["nb_classes"], kernel_size=17).to(torch.float64)
        #self.outlayer2 = nn.Linear(in_features=16, out_features=1).to(torch.float64)

    def forward(self, data,node_features,go_adj, batch_adj):
        ############################## 串行 ##############################
        # 输入：(batch,feature) torch.Size([64, 1256])
        # 输出: (batch,class)
        # ####### 先进行词嵌入 #########
        go_adj = torch.tensor(go_adj).to(self.args["device"])
        inputs1 = data[:, :self.MAXLEN]#torch.Size([64, 1000]) ##input1加入位置编码
        inputs1 = self.position1(inputs1)  # 加入位置编码
        inputs2 =  data[:, self.MAXLEN:]#torch.Size([64, 256])
        inputs2 = self.position2(inputs2)  # 加入位置编码
        feature = self.feature(inputs1)##torch.Size([64, 1000])-->#torch.Size([64, 256])
        merged_1 = torch.concatenate([feature, inputs2],dim=1)  # torch.Size([64, 256])+torch.Size([64, 256])-->torch.Size([64, 512])
        merged_1 = self.norm1(merged_1)

        ############## 分支一：Protein-protein GCN ##############
        adj = self.ProteinAttention(merged_1)  # (batch,batch)-->(batch,batch) torch.Size([64, 64])
        batch_adj = batch_adj + torch.softmax(adj, dim=1)
        #batch_adj = torch.softmax(batch_adj, dim=1)  # 归一化
        #batch_adj = torch.sigmoid(batch_adj)  # 归一化

        ######### 图卷积  #########
        x1 = self.ProteinConv(merged_1,batch_adj)  # ((batch,feature )(batch,batch)->(batch,len,feature) torch.Size([64, 512])->torch.Size([64, 512])
        x1 = self.norm2(x1) #torch.Size([64, 512])
        x1 = F.dropout(F.relu(x1), p=self.dropout, training=self.training) #torch.Size([64, 512])
        #x1 = self.outlayer1(x1) # torch.Size([64, 512])->torch.Size([64, 512])

        ###########################  串行连接 #############################
        ############## 分支二：Function-Function GCN ############
        ## 特征向量拼接 代替 残差连接，这是跳跃连接
        #new_feature = torch.concat((x1 , merged_1),dim=1) #torch.Size([64, 512])-->torch.Size([64, 1024])
        new_feature = x1 + merged_1 #残存连接
        merged = self.goFeature(new_feature)  # torch.Size([64, 512])-->torch.Size([64, nbclass, feature])
        ########### 加入图嵌入的节点特征（加入先验知识） ###########
        # node_features =  self.args["node_features"].to(self.args["device"])
        # print(node_features)
        node_features = node_features.unsqueeze(0).expand(data.shape[0], -1, -1)
        #print(node_features.requires_grad)
        merged = torch.concat((merged ,node_features),dim=2) #torch.Size([64, nbclass, feature]-->torch.Size([64, nbclass, feature+node_feature]- 32+16=48
        ##特征融合
        merged = self.norm3(merged)

        ######### 自适应构建图 #########
        atten_adj = self.FunctionAttention(merged)#torch.Size([64, nbclass, feature])torch.Size([64, 589, 64])-->得到矩阵 torch.Size([batch, nbclass, nbclass])
        ### go_ad将大小为torch.Size([nbclass, nbclass])的矩阵扩展为大小为torch.Size([batch, nbclass, nbclass])的矩阵
        go_adj = go_adj.unsqueeze(0).expand(data.shape[0], -1, -1)
        go_adj = go_adj + torch.softmax(atten_adj,dim=1) #####torch.Size([batch, nbclass, nbclass])
        #go_adj = torch.softmax(go_adj,dim=1) ##### torch.Size([batch, nbclass, nbclass]) 归一化

        ######### 图卷积：汉堡包结构,引入了go-go图卷积  #########
        #CONV/FC -> BatchNorm -> ReLu(or other activation) -> Dropout -> CONV/FC ->
        x2 =self.FunctionConv(merged, go_adj) + merged#torch.Size([64, nbclass, feature])torch.Size([nbclass, nbclass])-> torch.Size([64, nbclass, new_feature])
        x2 = self.norm4(x2)
        x2 = F.dropout(F.relu(x2), p=self.dropout, training=self.training)

        ############## 输出层 ##############
        ### 结合输出：
        #output = self.fuse * x1 + (1-self.fuse) * x2
        output = self.outlayer2(x2).squeeze(dim=-1)  # torch.Size([64, nbclass, new_feature])->torch.Size([64, nbclass, 1])->torch.Size([64, nbclass])
        #output = self.outlayer2(self.outlayer1(x2)).squeeze(dim=-1)
        return output
        #return F.softmax(output, dim=1)
        #return torch.sigmoid(output)
        #return F.log_softmax(output, dim=1)

        ####################### 并行：结构2 #######################
        ### 输入：(batch,feature) torch.Size([64, 1256])
        ### 输出: (batch,class)
        ######### 先进行词嵌入 #########
        # go_adj = torch.tensor(go_adj).to(self.args["device"])
        # inputs1 = data[:, :self.MAXLEN]  # torch.Size([64, 1000])
        # inputs2 = data[:, self.MAXLEN:]  # torch.Size([64, 256])
        # feature = self.feature(inputs1)  ##torch.Size([64, 1000])-->#torch.Size([64, 256])
        # merged = torch.concatenate([feature, inputs2],dim=1)  # torch.Size([64, 256])+torch.Size([64, 256])-->torch.Size([64, 512])
        # merged = self.norm1(merged)
        # ############## 分支一：Protein-protein GCN ##############
        # adj = self.ProteinAttention(merged)  # (batch,batch)-->(batch,batch) torch.Size([64, 64])
        # #batch_adj = torch.softmax(batch_adj + adj, dim=1)  # 归一化
        # batch_adj = torch.mm(batch_adj,adj)
        # #batch_adj = self.scaled_adj(batch_adj + adj,self.args)
        # batch_adj = torch.softmax(batch_adj, dim=1)  # 归一化
        #
        # ######### 图卷积  #########
        # x1 = F.relu(self.ProteinConv(merged,batch_adj))  # ((batch,feature )(batch,batch)->(batch,feature) torch.Size([64, 512])->torch.Size([64, 512])
        # x1 = F.dropout(x1, p=self.dropout, training=self.training)  # torch.Size([64, 512])
        # x1 = self.norm2(x1)  # torch.Size([64, 512])
        # x1 = self.outlayer1(x1) # torch.Size([64, 512])->torch.Size([64, 512])
        #
        # ############## 分支二：Function-Function GCN ############
        # merged = self.goFeature(merged)  # torch.Size([64, 512])-->torch.Size([64, nbclass, feature])
        # node_features = node_features.unsqueeze(0).expand(data.shape[0], -1, -1)
        # merged = torch.concat((merged, node_features), dim=2) #加入图卷积特征
        # ######### 自适应构建图 #########
        # atten_adj = self.FunctionAttention(merged)  # torch.Size([64, nbclass, feature])torch.Size([64, 589, 64])-->得到矩阵 torch.Size([batch, nbclass, nbclass])
        # go_adj = torch.matmul(atten_adj,go_adj) # [nbclass, nbclass]* torch.Size([batch, nbclass, nbclass])->
        # # go_adj = self.scaled_adj(go_adj,self.args) ## 归一化方法1
        #
        # ### go_ad将大小为torch.Size([nbclass, nbclass])的矩阵扩展为大小为torch.Size([batch, nbclass, nbclass])的矩阵
        # # go_adj = go_adj.unsqueeze(0).expand(data.shape[0], -1, -1)
        # # go_adj = go_adj + atten_adj  #####torch.Size([batch, nbclass, nbclass])
        # go_adj = torch.softmax(go_adj, dim=1)  ##### torch.Size([batch, nbclass, nbclass]) #归一化方法2
        #
        # ######### 图卷积：引入了go-go图卷积  #########
        # x2 = F.relu(self.FunctionConv(merged,go_adj)+merged)  # torch.Size([64, nbclass, feature])torch.Size([nbclass, nbclass])-> torch.Size([64, nbclass, new_feature])
        # x2 = F.dropout(x2, p=self.dropout, training=self.training)  # torch.Size([64, nbclass, new_feature])
        # x2 = self.norm3(x2)
        # x2 = self.outlayer2(x2).squeeze(dim=-1)  # torch.Size([64, nbclass, new_feature])->torch.Size([64, nbclass, 1])->torch.Size([64, nbclass])
        #
        # ############## 输出层 ##############
        # ### 结合输出：
        # output = self.fuse * x1 + (1-self.fuse) * x2
        # output = self.outlayer3(output)
        # return output
        # return F.softmax(output, dim=1)
        # return torch.sigmoid(output)
        # return F.log_softmax(output, dim=1)
        ########################################## 串行 +分层分类 ##########################################
        ## 输入：(batch,feature) torch.Size([64, 1256])
        ## 输出: (batch,class)
        ######## 先进行词嵌入 #########
        # go_adj = torch.tensor(go_adj).to(self.args["device"])
        # inputs1 = data[:, :self.MAXLEN]  # torch.Size([64, 1000])
        # inputs2 = data[:, self.MAXLEN:]  # torch.Size([64, 256])
        # feature = self.feature(inputs1)  ##torch.Size([64, 1000])-->#torch.Size([64, 256])
        # merged = torch.concatenate([feature, inputs2],dim=1)  # torch.Size([64, 256])+torch.Size([64, 256])-->torch.Size([64, 512])
        # merged_1 = self.norm1(merged)
        #
        # ############## 分支一：Protein-protein GCN ##############
        # adj = self.ProteinAttention(merged_1)  # (batch,batch)-->(batch,batch) torch.Size([64, 64])
        # batch_adj = torch.softmax(batch_adj + adj, dim=1)  # 归一化
        # # batch_adj = torch.sigmoid(batch_adj)  # 归一化
        #
        # ######### 图卷积  #########
        # x1 = F.relu(self.ProteinConv(merged_1,batch_adj) + merged_1)  # ((batch,feature )(batch,batch)->(batch,len,feature) torch.Size([64, 512])->torch.Size([64, 512])
        # x1 = F.dropout(x1, p=self.dropout, training=self.training)  # torch.Size([64, 512])
        # x1 = self.norm2(x1)  # torch.Size([64, 512])
        # # x1 = self.outlayer1(x1) # torch.Size([64, 512])->torch.Size([64, 512])
        #
        # ###########################  串行连接 #############################
        #
        # ############## 分支二：Function-Function GCN ############
        # merged = self.goFeature(x1)  # torch.Size([64, 512])-->torch.Size([64, nbclass, feature])
        # ######### 自适应构建图 #########
        # atten_adj = self.FunctionAttention(merged)  # torch.Size([64, nbclass, feature])torch.Size([64, 589, 64])-->得到矩阵 torch.Size([batch, nbclass, nbclass])
        # ### go_ad将大小为torch.Size([nbclass, nbclass])的矩阵扩展为大小为torch.Size([batch, nbclass, nbclass])的矩阵
        # go_adj = go_adj.unsqueeze(0).expand(data.shape[0], -1, -1)
        # go_adj = go_adj + atten_adj  #####torch.Size([batch, nbclass, nbclass])
        # go_adj = torch.softmax(go_adj, dim=1)  ##### torch.Size([batch, nbclass, nbclass]) 归一化
        #
        # ######### 图卷积：汉堡包结构,引入了go-go图卷积  #########
        # x2 = F.relu(self.FunctionConv(merged,go_adj) + merged)  # torch.Size([64, nbclass, feature])torch.Size([nbclass, nbclass])-> torch.Size([64, nbclass, new_feature])
        # x2 = F.dropout(x2, p=self.dropout, training = self.training)  # torch.Size([64, nbclass, new_feature])
        # x2 = self.norm3(x2)
        #
        # ############ 分支3 ：分层分类 ############
        # x3 = self.hie_class(merged_1)
        # ############## 输出层 ##############
        # ### 结合输出：
        # # output = self.fuse * x1 + (1-self.fuse) * x2
        # output = self.outlayer1(x2).squeeze(dim=-1)  # torch.Size([64, nbclass, new_feature])->torch.Size([64, nbclass, 1])->torch.Size([64, nbclass])
        # output = self.outlayer2(self.fuse * output + (1 - self.fuse) * x3)
        # return output
    def getFeature(self, data,node_features,go_adj, batch_adj):
        ############################## 串行 ##############################
        # 输入：(batch,feature) torch.Size([64, 1256])
        # 输出: (batch,class)
        # ####### 先进行词嵌入 #########
        go_adj = torch.tensor(go_adj).to(self.args["device"])
        inputs1 = data[:, :self.MAXLEN]#torch.Size([64, 1000]) ##input1加入位置编码
        inputs1 = self.position1(inputs1)  # 加入位置编码
        inputs2 =  data[:, self.MAXLEN:]#torch.Size([64, 256])
        feature = self.feature(inputs1)##torch.Size([64, 1000])-->#torch.Size([64, 256])

        inputs2 = self.position2(inputs2) # 加入位置编码
        merged_1 = torch.concatenate([feature, inputs2],dim=1)  # torch.Size([64, 256])+torch.Size([64, 256])-->torch.Size([64, 512])
        merged_1 = self.norm1(merged_1)

        ############## 分支一：Protein-protein GCN ##############
        adj = self.ProteinAttention(merged_1)  # (batch,batch)-->(batch,batch) torch.Size([64, 64])
        batch_adj = batch_adj + torch.softmax(adj, dim=1)
        #batch_adj = torch.softmax(batch_adj, dim=1)  # 归一化
        #batch_adj = torch.sigmoid(batch_adj)  # 归一化

        ######### 图卷积  #########
        x1 = self.ProteinConv(merged_1,batch_adj)  # ((batch,feature )(batch,batch)->(batch,len,feature) torch.Size([64, 512])->torch.Size([64, 512])
        x1 = self.norm2(x1) #torch.Size([64, 512])
        x1 = F.dropout(F.relu(x1), p=self.dropout, training=self.training)  # torch.Size([64, 512])
        # x1 = self.outlayer1(x1) # torch.Size([64, 512])->torch.Size([64, 512])

        ###########################  串行连接 #############################
        ############## 分支二：Function-Function GCN ############
        ## 特征向量拼接 代替 残差连接，这是跳跃连接
        # new_feature = torch.concat((x1 , merged_1),dim=1) #torch.Size([64, 512])-->torch.Size([64, 1024])
        new_feature = x1 + merged_1  # 残存连接
        merged = self.goFeature(new_feature)  # torch.Size([64, 512])-->torch.Size([64, nbclass, feature])
        ########### 加入图嵌入的节点特征（加入先验知识） ###########
        # node_features =  self.args["node_features"].to(self.args["device"])
        # print(node_features)
        node_features = node_features.unsqueeze(0).expand(data.shape[0], -1, -1)
        # print(node_features.requires_grad)
        merged = torch.concat((merged, node_features),dim=2)  # torch.Size([64, nbclass, feature]-->torch.Size([64, nbclass, feature+node_feature]- 32+16=48
        ##特征融合
        merged = self.norm3(merged)

        ######### 自适应构建图 #########
        atten_adj = self.FunctionAttention(merged)  # torch.Size([64, nbclass, feature])torch.Size([64, 589, 64])-->得到矩阵 torch.Size([batch, nbclass, nbclass])
        ### go_ad将大小为torch.Size([nbclass, nbclass])的矩阵扩展为大小为torch.Size([batch, nbclass, nbclass])的矩阵
        go_adj = go_adj.unsqueeze(0).expand(data.shape[0], -1, -1)
        go_adj = go_adj + torch.softmax(atten_adj, dim=1)  #####torch.Size([batch, nbclass, nbclass])
        # go_adj = torch.softmax(go_adj,dim=1) ##### torch.Size([batch, nbclass, nbclass]) 归一化

        ######### 图卷积：汉堡包结构,引入了go-go图卷积  #########
        # CONV/FC -> BatchNorm -> ReLu(or other activation) -> Dropout -> CONV/FC ->
        x2 = self.FunctionConv(merged,go_adj) + merged  # torch.Size([64, nbclass, feature])torch.Size([nbclass, nbclass])-> torch.Size([64, nbclass, new_feature])
        x2 = self.norm4(x2)
        #x2 = F.dropout(F.relu(x2), p=self.dropout, training=self.training)

        return x2# torch.Size([64, nbclass, new_feature])--> torch.Size([64*nbclass, new_feature])

class PGCN(torch.nn.Module):
    def __init__(self,args):
        super(PGCN, self).__init__()
        self.training = True
        self.dropout = 0.2
        self.node_names = args["node_names"]
        self.go = args["go"]
        self.functions=args["functions"]
        self.func_set = set(self.functions)
        self.Go_id = args["Go_id"]
        self.MAXLEN = args["MAXLEN"]
        self.args = args
        # #这里加入自适应调整
        # if args["addaptadj"]:
        #     if args["supports"] is None:
        #         self.supports = []
        #     self.nodevec1 = nn.Parameter(torch.randn(args["batch_size"], 10).to(args["device"]), requires_grad=True).to(args["device"])
        #     self.nodevec2 = nn.Parameter(torch.randn(10, args["batch_size"]).to(args["device"]), requires_grad=True).to(args["device"])
        #### 词嵌入层 ####
        self.position1 = PositionalEncoding(outfea=1 , len=256,args=args)
        self.position2 = PositionalEncoding(outfea=1, len=256, args=args)
        self.feature = FeatureModel(max_features=args["Feature"]["max_features"],
                                    embedding_dims=args["Feature"]["embedding_dims"])
        self.goFeature = GoFeature(in_features=args["GoFeature"]["in_features"],
                                      classes=args["nb_classes"],
                                      out_channels=args["GoFeature"]["out_channels"])
        #### BatchNorm
        self.norm1 = nn.BatchNorm1d(args["GoFeature"]["in_features"]).to(torch.float64)
        self.norm2 = nn.BatchNorm1d(args["ProteinConv"]["in_channels"]).to(torch.float64)
        self.norm3 = nn.BatchNorm1d(args["nb_classes"]).to(torch.float64)
        self.norm4 = nn.BatchNorm1d(args["nb_classes"]).to(torch.float64)


        self.ProteinAttention = ProteinAttention(args["ProteinAttention"]["in_channels"],
                                                 args["ProteinAttention"]["hid_channels"],
                                                 args["ProteinAttention"]["out_channels"])
        self.scaled_adj = scaled_Laplacian
        self.ProteinConv = ProteinConv(args["ProteinConv"]["K"],
                                        args["ProteinConv"]["in_channels"],
                                       args["ProteinConv"]["out_channels"],
                                       args["device"])
        #self.hie_class = Hierarchical_classifer_new(args)
        #### 输出层
        self.fuse = nn.Parameter(data=torch.tensor(0.5),requires_grad=True)
        ### 第一种输出层
        self.outlayer1 = nn.Linear(in_features=args["ProteinConv"]["out_channels"], out_features=args["nb_classes"]).to(torch.float64)
        self.outlayer2 = nn.Linear(in_features=args["nb_classes"], out_features=args["nb_classes"]).to(torch.float64)

        ### 第二种输出层
        #self.outlayer1 = nn.Conv1d(in_channels=args["nb_classes"], out_channels=args["nb_classes"], kernel_size=17).to(torch.float64)
        #self.outlayer2 = nn.Linear(in_features=16, out_features=1).to(torch.float64)

    def forward(self, data,node_features,go_adj, batch_adj):
        ############################## 串行 ##############################
        # 输入：(batch,feature) torch.Size([64, 1256])
        # 输出: (batch,class)
        # ####### 先进行词嵌入 #########
        go_adj = torch.tensor(go_adj).to(self.args["device"])
        inputs1 = data[:, :self.MAXLEN]#torch.Size([64, 1000]) ##input1加入位置编码
        inputs2 =  data[:, self.MAXLEN:]#torch.Size([64, 256])
        feature = self.feature(inputs1)##torch.Size([64, 1000])-->#torch.Size([64, 256])
        feature = self.position1(feature)  # 加入位置编码
        inputs2 = self.position2(inputs2) # 加入位置编码
        # merged_1 = torch.concatenate([feature, inputs2],dim=1)  # torch.Size([64, 256])+torch.Size([64, 256])-->torch.Size([64, 512])
        # merged_1 = self.norm1(merged_1)
        merged_1 = inputs2
        ############## 分支一：Protein-protein GCN ##############
        adj = self.ProteinAttention(merged_1)  # (batch,batch)-->(batch,batch) torch.Size([64, 64])
        batch_adj = batch_adj + torch.softmax(adj, dim=1)
        #batch_adj = torch.softmax(batch_adj, dim=1)  # 归一化
        #batch_adj = torch.sigmoid(batch_adj)  # 归一化

        ######### 图卷积  #########
        x1 = self.ProteinConv(merged_1,batch_adj)  # ((batch,feature )(batch,batch)->(batch,len,feature) torch.Size([64, 512])->torch.Size([64, 512])
        x1 = self.norm2(x1) #torch.Size([64, 512])
        x1 = F.dropout(F.relu(x1), p=self.dropout, training=self.training) #torch.Size([64, 512])
        #x1 = self.outlayer1(x1) # torch.Size([64, 512])->torch.Size([64, 512])

        ############## 输出层 ##############
        ### 结合输出：
        #output = self.fuse * x1 + (1-self.fuse) * x2
        output = self.outlayer1(x1)# torch.Size([64, nbclass, new_feature])->torch.Size([64, nbclass, 1])->torch.Size([64, nbclass])
        output = self.outlayer2(output)
        #output = self.outlayer2(self.outlayer1(x2)).squeeze(dim=-1)
        return output


###### 引入go-go网络和protein-protein网络，间接构建蛋白质到蛋白质的网络
class FGCN(torch.nn.Module):
    def __init__(self,args):
        super(FGCN, self).__init__()
        self.training = True
        self.dropout = 0.2
        self.node_names = args["node_names"]
        self.go = args["go"]
        self.functions=args["functions"]
        self.func_set = set(self.functions)
        self.Go_id = args["Go_id"]
        self.MAXLEN = args["MAXLEN"]
        self.args = args
        # #这里加入自适应调整
        # if args["addaptadj"]:
        #     if args["supports"] is None:
        #         self.supports = []
        #     self.nodevec1 = nn.Parameter(torch.randn(args["batch_size"], 10).to(args["device"]), requires_grad=True).to(args["device"])
        #     self.nodevec2 = nn.Parameter(torch.randn(10, args["batch_size"]).to(args["device"]), requires_grad=True).to(args["device"])
        #### 词嵌入层 ####
        self.position1 = PositionalEncoding(outfea=1 , len=256,args=args)
        self.position2 = PositionalEncoding(outfea=1, len=256, args=args)
        self.feature = FeatureModel(max_features=args["Feature"]["max_features"],
                                    embedding_dims=args["Feature"]["embedding_dims"])
        self.goFeature = GoFeature(in_features=args["GoFeature"]["in_features"],
                                      classes=args["nb_classes"],
                                      out_channels=args["GoFeature"]["out_channels"])
        #### BatchNorm
        self.norm1 = nn.BatchNorm1d(args["GoFeature"]["in_features"]).to(torch.float64)
        self.norm2 = nn.BatchNorm1d(args["ProteinConv"]["in_channels"]).to(torch.float64)
        self.norm3 = nn.BatchNorm1d(args["nb_classes"]).to(torch.float64)
        self.norm4 = nn.BatchNorm1d(args["nb_classes"]).to(torch.float64)

        self.FunctionAttention = FunctionAttention(args["FunctionAttention"]["in_channels"],
                                         args["FunctionAttention"]["hid_channels"],
                                         args["FunctionAttention"]["out_channels"])

        self.scaled_adj = scaled_Laplacian
        self.FunctionConv = FunctionConv(args["FunctionConv"]["K"],
                                        args["FunctionConv"]["in_channels"],
                                         args["FunctionConv"]["out_channels"],
                                         args["device"])
        #self.hie_class = Hierarchical_classifer_new(args)
        #### 输出层
        self.fuse = nn.Parameter(data=torch.tensor(0.5),requires_grad=True)
        ### 第一种输出层

        self.outlayer1 = nn.Linear(in_features=args["FunctionConv"]["out_channels"], out_features=1).to(torch.float64)
        #self.outlayer2 = nn.Linear(in_features=args["nb_classes"], out_features=args["nb_classes"]).to(torch.float64)
        ### 第二种输出层
        #self.outlayer1 = nn.Conv1d(in_channels=args["nb_classes"], out_channels=args["nb_classes"], kernel_size=17).to(torch.float64)
        #self.outlayer2 = nn.Linear(in_features=16, out_features=1).to(torch.float64)

    def forward(self, data,node_features,go_adj, batch_adj):
        ############################## 串行 ##############################
        # 输入：(batch,feature) torch.Size([64, 1256])
        # 输出: (batch,class)
        # ####### 先进行词嵌入 #########
        go_adj = torch.tensor(go_adj).to(self.args["device"])
        inputs1 = data[:, :self.MAXLEN]#torch.Size([64, 1000]) ##input1加入位置编码
        inputs2 =  data[:, self.MAXLEN:]#torch.Size([64, 256])
        feature = self.feature(inputs1)##torch.Size([64, 1000])-->#torch.Size([64, 256])
        feature = self.position1(feature)  # 加入位置编码
        inputs2 = self.position2(inputs2) # 加入位置编码
        merged_1 = torch.concatenate([feature, inputs2],dim=1)  # torch.Size([64, 256])+torch.Size([64, 256])-->torch.Size([64, 512])
        merged_1 = self.norm1(merged_1)
        ###########################  串行连接 #############################
        ############## 分支二：Function-Function GCN ############
        ## 特征向量拼接 代替 残差连接，这是跳跃连接
        #new_feature = torch.concat((x1 , merged_1),dim=1) #torch.Size([64, 512])-->torch.Size([64, 1024])
        new_feature =  merged_1 #残存连接
        merged = self.goFeature(new_feature)  # torch.Size([64, 512])-->torch.Size([64, nbclass, feature])
        ########### 加入图嵌入的节点特征（加入先验知识） ###########
        # node_features =  self.args["node_features"].to(self.args["device"])
        # print(node_features)
        node_features = node_features.unsqueeze(0).expand(data.shape[0], -1, -1)
        #print(node_features.requires_grad)
        merged = torch.concat((merged ,node_features),dim=2) #torch.Size([64, nbclass, feature] 32+16=48
        merged = self.norm3(merged)

        ######### 自适应构建图 #########
        atten_adj = self.FunctionAttention(merged)#torch.Size([64, nbclass, feature])torch.Size([64, 589, 64])-->得到矩阵 torch.Size([batch, nbclass, nbclass])
        ### go_ad将大小为torch.Size([nbclass, nbclass])的矩阵扩展为大小为torch.Size([batch, nbclass, nbclass])的矩阵
        go_adj = go_adj.unsqueeze(0).expand(data.shape[0], -1, -1)
        go_adj = go_adj + torch.softmax(atten_adj,dim=1) #####torch.Size([batch, nbclass, nbclass])
        #go_adj = torch.softmax(go_adj,dim=1) ##### torch.Size([batch, nbclass, nbclass]) 归一化

        ######### 图卷积：汉堡包结构,引入了go-go图卷积  #########
        #CONV/FC -> BatchNorm -> ReLu(or other activation) -> Dropout -> CONV/FC ->
        x2 =self.FunctionConv(merged, go_adj) + merged#torch.Size([64, nbclass, feature])torch.Size([nbclass, nbclass])-> torch.Size([64, nbclass, new_feature])
        x2 = self.norm4(x2)
        x2 = F.dropout(F.relu(x2), p=self.dropout, training=self.training)

        ############## 输出层 ##############
        ### 结合输出：
        #output = self.fuse * x1 + (1-self.fuse) * x2
        output = self.outlayer1(x2).squeeze(dim=-1)  # torch.Size([64, nbclass, new_feature])->torch.Size([64, nbclass, 1])->torch.Size([64, nbclass])
        #output = self.outlayer2(output)
        #output = self.outlayer2(self.outlayer1(x2)).squeeze(dim=-1)
        return output
        #return F.softmax(output, dim=1)
        #return torch.sigmoid(output)
        #return F.log_softmax(output, dim=1)


###### 引入go-go网络和protein-protein网络，间接构建蛋白质到蛋白质的网络
class PFGCNgo_AA(torch.nn.Module):
    def __init__(self,args):
        super( PFGCNgo_AA, self).__init__()
        self.training = True
        self.dropout = 0.2
        self.node_names = args["node_names"]
        self.go = args["go"]
        self.functions=args["functions"]
        self.func_set = set(self.functions)
        self.Go_id = args["Go_id"]
        self.MAXLEN = args["MAXLEN"]
        self.args = args
        # #这里加入自适应调整
        # if args["addaptadj"]:
        #     if args["supports"] is None:
        #         self.supports = []
        #     self.nodevec1 = nn.Parameter(torch.randn(args["batch_size"], 10).to(args["device"]), requires_grad=True).to(args["device"])
        #     self.nodevec2 = nn.Parameter(torch.randn(10, args["batch_size"]).to(args["device"]), requires_grad=True).to(args["device"])
        #### 词嵌入层 ####
        # self.position1 = PositionalEncoding(outfea=1 , len=1000,args=args)
        self.position2 = PositionalEncoding(outfea=1, len=256, args=args)
        # self.feature = FeatureModel(max_features=args["Feature"]["max_features"],
        #                             embedding_dims=args["Feature"]["embedding_dims"])
        self.goFeature = GoFeature(in_features=args["GoFeature"]["in_features"],
                                      classes=args["nb_classes"],
                                      out_channels=args["GoFeature"]["out_channels"])
        #### BatchNorm
        self.norm1 = nn.BatchNorm1d(256).to(torch.float64)
        self.norm2 = nn.BatchNorm1d(args["ProteinConv"]["in_channels"]).to(torch.float64)
        self.norm3 = nn.BatchNorm1d(args["nb_classes"]).to(torch.float64)
        self.norm4 = nn.BatchNorm1d(args["nb_classes"]).to(torch.float64)

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
        #self.hie_class = Hierarchical_classifer_new(args)
        #### 输出层
        # self.fuse = nn.Parameter(data=torch.tensor(0.5),requires_grad=True)
        ### 第一种输出层
        #self.outlayer1 = nn.Linear(in_features=args["ProteinConv"]["out_channels"], out_features=args["nb_classes"]).to(torch.float64)
        self.outlayer2 = nn.Linear(in_features=args["FunctionConv"]["out_channels"], out_features=1).to(torch.float64)
        #self.outlayer3 = nn.Linear(in_features=args["nb_classes"], out_features=args["nb_classes"]).to(torch.float64)
        ### 第二种输出层
        #self.outlayer1 = nn.Conv1d(in_channels=args["nb_classes"], out_channels=args["nb_classes"], kernel_size=17).to(torch.float64)
        #self.outlayer2 = nn.Linear(in_features=16, out_features=1).to(torch.float64)

    def forward(self, data,node_features,go_adj, batch_adj):
        ############################## 串行 ##############################
        # 输入：(batch,feature) torch.Size([64, 1256])
        # 输出: (batch,class)
        # ####### 先进行词嵌入 #########
        go_adj = torch.tensor(go_adj).to(self.args["device"])
        # inputs1 = data[:, :self.MAXLEN]#torch.Size([64, 1000]) ##input1加入位置编码
        # inputs1 = self.position1(inputs1)  # 加入位置编码
        inputs2 =  data[:, self.MAXLEN:]#torch.Size([64, 256])
        inputs2 = self.position2(inputs2)  # 加入位置编码
        #feature = self.feature(inputs1)##torch.Size([64, 1000])-->#torch.Size([64, 256])
        # merged_1 = torch.concatenate([feature, inputs2],dim=1)  # torch.Size([64, 256])+torch.Size([64, 256])-->torch.Size([64, 512])
        merged_1 = inputs2
        merged_1 = self.norm1(merged_1)

        ############## 分支一：Protein-protein GCN ##############
        adj = self.ProteinAttention(merged_1)  # (batch,batch)-->(batch,batch) torch.Size([64, 64])
        batch_adj = batch_adj + torch.softmax(adj, dim=1)
        #batch_adj = torch.softmax(batch_adj, dim=1)  # 归一化
        #batch_adj = torch.sigmoid(batch_adj)  # 归一化

        ######### 图卷积  #########
        x1 = self.ProteinConv(merged_1,batch_adj)  # ((batch,feature )(batch,batch)->(batch,len,feature) torch.Size([64, 512])->torch.Size([64, 512])
        x1 = self.norm2(x1) #torch.Size([64, 512])
        x1 = F.dropout(F.relu(x1), p=self.dropout, training=self.training) #torch.Size([64, 512])
        #x1 = self.outlayer1(x1) # torch.Size([64, 512])->torch.Size([64, 512])

        ###########################  串行连接 #############################
        ############## 分支二：Function-Function GCN ############
        ## 特征向量拼接 代替 残差连接，这是跳跃连接
        #new_feature = torch.concat((x1 , merged_1),dim=1) #torch.Size([64, 512])-->torch.Size([64, 1024])
        new_feature = x1 + merged_1 #残存连接
        merged = self.goFeature(new_feature)  # torch.Size([64, 512])-->torch.Size([64, nbclass, feature])
        ########### 加入图嵌入的节点特征（加入先验知识） ###########
        # node_features =  self.args["node_features"].to(self.args["device"])
        # print(node_features)
        node_features = node_features.unsqueeze(0).expand(data.shape[0], -1, -1)
        #print(node_features.requires_grad)
        merged = torch.concat((merged ,node_features),dim=2) #torch.Size([64, nbclass, feature]-->torch.Size([64, nbclass, feature+node_feature]- 32+16=48
        ##特征融合
        merged = self.norm3(merged)

        ######### 自适应构建图 #########
        atten_adj = self.FunctionAttention(merged)#torch.Size([64, nbclass, feature])torch.Size([64, 589, 64])-->得到矩阵 torch.Size([batch, nbclass, nbclass])
        ### go_ad将大小为torch.Size([nbclass, nbclass])的矩阵扩展为大小为torch.Size([batch, nbclass, nbclass])的矩阵
        go_adj = go_adj.unsqueeze(0).expand(data.shape[0], -1, -1)
        go_adj = go_adj + torch.softmax(atten_adj,dim=1) #####torch.Size([batch, nbclass, nbclass])
        #go_adj = torch.softmax(go_adj,dim=1) ##### torch.Size([batch, nbclass, nbclass]) 归一化

        ######### 图卷积：汉堡包结构,引入了go-go图卷积  #########
        #CONV/FC -> BatchNorm -> ReLu(or other activation) -> Dropout -> CONV/FC ->
        x2 =self.FunctionConv(merged, go_adj) + merged#torch.Size([64, nbclass, feature])torch.Size([nbclass, nbclass])-> torch.Size([64, nbclass, new_feature])
        x2 = self.norm4(x2)
        x2 = F.dropout(F.relu(x2), p=self.dropout, training=self.training)

        ############## 输出层 ##############
        ### 结合输出：
        #output = self.fuse * x1 + (1-self.fuse) * x2
        output = self.outlayer2(x2).squeeze(dim=-1)  # torch.Size([64, nbclass, new_feature])->torch.Size([64, nbclass, 1])->torch.Size([64, nbclass])
        #output = self.outlayer2(self.outlayer1(x2)).squeeze(dim=-1)
        return output