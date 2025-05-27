from model.GCNDeepgo import FeatureModel,Hierarchical_classifer_new
import torch
from torch_geometric.nn import GCNConv
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
#from utils_all import get_parents,get_gene_ontology,get_function_node,get_node_name
#from utils_all import *
import pandas as pd


### 先静态构建邻接矩阵，再自适应构调整
### 静态邻接矩阵(len,len),自适应调整（batch,batch)
class Deepgo(torch.nn.Module):
    def __init__(self,args):
        super(Deepgo, self).__init__()

        self.training = True
        self.dropout = 0.2
        self.node_names = args["node_names"]
        self.go = args["go"]
        self.functions=args["functions"]
        self.func_set = set(self.functions)
        self.Go_id = args["Go_id"]
        self.MAXLEN = args["MAXLEN"]

        self.feature = FeatureModel(max_features=args["Feature"]["max_features"],
                                    embedding_dims=args["Feature"]["embedding_dims"])
        #### BatchNorm
        self.norm1 = nn.BatchNorm1d(args["cheb_conv"]["out_channels"]).to(torch.float64)
        #self.norm2 = nn.BatchNorm1d(args["nb_classes"]).to(torch.float64)
        ###### 分层分类
        self.hie_class = Hierarchical_classifer_new(args)
        #### 输出层
        self.outlayer = nn.Linear(in_features=args["nb_classes"],out_features=args["nb_classes"]).to(torch.float64)

    def forward(self, data):
        ### 输入：(batch,feature) torch.Size([64, 1256])
        ### 输出: (batch,class)
        ######### 先进行词嵌入 #########
        inputs1 = data[:, :self.MAXLEN]#torch.Size([64, 1000])
        inputs2 =  data[:, self.MAXLEN:]#torch.Size([64, 256])
        feature = self.feature(inputs1)##torch.Size([64, 1000])-->#torch.Size([64, 256])
        merged = torch.concatenate([feature, inputs2],dim=1)  # torch.Size([64, 256])+torch.Size([64, 256])-->torch.Size([64, 512])
        merged = self.norm1(merged)

        ############## 知识图谱 ##############
        ######################## 第二层：构建了多层感知机结构,每个GO term用一层表示。通过这个结构,可以学习父子GO term之间的关系,反映蛋白质间的拓扑互作用信息 ########################
        #layers = get_layers(inputs=merged,node_names=self.node_names,GO_ID=self.Go_id,go=self.go,func_set=self.func_set,functions=self.functions)  # TensorShape([Dimension(None), Dimension(None)])-->
        x2 = self.hie_class(merged)#torch.Size([64, 512])--> torch.Size([64, nb_classes])
        #x2 = F.dropout(x2, p=self.dropout, training=self.training)

        ############## 输出层 ##############
        ### 结合输出：
        output =  self.outlayer(x2)
        #output = x2
        return output
        #return F.softmax(output, dim=1)
        #return torch.sigmoid(output)
        #return F.log_softmax(output, dim=1)
