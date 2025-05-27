import torch
from torch_geometric.nn import GCNConv
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from utils_all import get_parents,get_gene_ontology,get_function_node,get_node_name
import pandas as pd

####### 点嵌入模型 #######
class FeatureModel(nn.Module):
    def __init__(self, max_features=8001, embedding_dims=128):
        super(FeatureModel, self).__init__()

        self.embedding = nn.Embedding(max_features, embedding_dims)
        #self.conv1d = nn.Conv1d(in_channels=1000, out_channels=256, kernel_size=128)
        self.norm = nn.BatchNorm1d(1000)  # 数据标准化模块
        #self.norm = nn.BatchNorm2d(1000)
        # self.conv1 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=125, dilation=2, padding=0)
        # self.conv2 = nn.Conv1d(in_channels=64, out_channels=1, kernel_size=125, dilation=4, padding=0)### 256

        self.conv1 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=55, dilation=2, padding=0)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=54, dilation=4, padding=0)  ### 256
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=1, kernel_size=54, dilation=8, padding=0)

        # self.conv1 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=115, dilation=2, padding=0)
        # self.conv2 = nn.Conv1d(in_channels=64, out_channels=1, kernel_size=114, dilation=4, padding=0)### 320
        # self.conv3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=9, dilation=3, padding=0)
        # self.conv4 = nn.Conv1d(in_channels=256, out_channels=1, kernel_size=9, dilation=3, padding=0)
        #self.max_pool1d = nn.MaxPool1d(kernel_size=64, stride=32)

        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.embedding(x.long())#torch.Size([64, 1000])--->torch.Size([64, 1000, 128])

        #x = x.permute(0,2,1) #torch.Size([64, 1000, 128])->torch.Size([64, 128, 1000])
        ######## 方案1 ########
        # x = self.conv1d(x)#torch.Size([64, 1000, 128])->torch.Size([64, 512, 1])
        # x = F.relu(x)
        ######## 方案2 ########
        # x = self.norm(x)  # 数据标准化 torch.Size([64, 128, 1000])->torch.Size([64, 128, 1000])
        # x = self.conv1d(x)#torch.Size([64, 1000, 128])->torch.Size([64, 512, 1])
        # x = F.relu(x)
        ######## 方案3 #######
        x = self.norm(x)
        x = x.permute(0, 2, 1)
        x = torch.relu(self.conv1(x))# torch.Size([64, 128, 1000])->torch.Size([64, 256, 996])
        x = torch.relu(self.conv2(x))#torch.Size([64, 256, 1000])->torch.Size([64, 1, 256])
        x = torch.relu(self.conv3(x))
        # x = F.relu(self.conv3(x))
        # x = self.conv4(x)
        #x = self.max_pool1d(x)
        x = self.flatten(x) # torch.Size([64, 1, 256])->torch.Size([64, 256])
        return x



###### 学习序列信息得到图结构 ######
class Attention_layer(nn.Module):
    def  __init__(self, in_channels, hid_channels, out_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_channels, hid_channels),
            nn.ReLU(),
            nn.Linear(hid_channels, out_channels),
            #nn.Linear(in_channels, out_channels),
        )
    def forward(self, x):
        ### 输入：(batch,class,feature)
        ### 输出：go到go关系的邻接矩阵 (classes，classes)
        ####方法一
        # if x.dtype == torch.float64:
        #     x = x.float()
        # Z = self.model(x)#(batch,class,feature)-->(batch,class,feature)
        # score = torch.mm(Z, Z.t()) #(batch,class,feature) * (batch,class,feature) --> (batch,class,class)
        # W = torch.sigmoid(score)  # 归一化为0-1权重
        # return W

        ####方法二
        if x.dtype == torch.float64:
            x = x.float()
        Z = self.model(x)  # (batch,class,feature)-->(batch,class,feature)
        score = torch.bmm(Z, Z.transpose(1, 2))  # (batch,class,feature) * (batch,class,feature) --> (batch,class,class)
        # 批量求和得到邻接矩阵
        #score = torch.sum(score, dim=0) #批量求和 (batch,class,class)-> (class,class)
        #W = torch.sigmoid(score)  # 归一化为0-1权重 (batch,class,class)
        W = F.softmax(score,dim=1).to(torch.float64)# 归一化为0-1权重 (batch,class,class)
        return W


#### 自定义的图卷积
class cheb_conv_withSAt(nn.Module):

    def __init__(self,in_channels, out_channels,device):
        '''
        :param K: int
        :param in_channles: int, num of channels in the input sequence
        :param out_channels: int, num of channels in the output sequence
        '''
        super(cheb_conv_withSAt, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.DEVICE = device
        #self.Theta = nn.ParameterList([nn.Parameter(torch.FloatTensor(in_channels, out_channels).to(self.DEVICE)) for _ in range(K)])
        self.Theta = nn.Parameter(torch.FloatTensor(in_channels, out_channels).to(self.DEVICE)).to(torch.float64)

    def forward(self, x, adj):
        '''
        Chebyshev graph convolution operation
        切比雪夫图卷积运算
        :param x: (batch_size, class, F_in),邻接矩阵 (batch_size, class, class)
        :return: (batch_size, class, F_out)
        '''
        ############### 写法1 ###############
        graph_signal = x  # (batch_size, class, F_in)

        T_k_with_at = adj  # torch.Size([ batch_size, class,  class])

        theta_k = self.Theta  # (in_channel, out_channel)

        rhs = torch.bmm(T_k_with_at,graph_signal)# (batch_size, class, F_in) * [ batch_size, class,  class]->(batch_size, class, F_in)

        #output = torch.bmm(rhs,theta_k) #(batch_size, class, F_in)(F_in, F_out) = (batch_size, class, F_in)
        output = rhs.matmul(theta_k)  # (batch_size, class, F_in)(F_in, F_out) = (batch_size, class, F_in)
        return F.relu(output)  # (b,  F_out)
        ############### 写法2 ，不正确###############
        # batch_size, num_of_vertices, in_channels = x.shape
        # output = torch.zeros(batch_size, num_of_vertices, self.out_channels).to(self.DEVICE)  # (b, N, F_out)
        # for k in range(self.K):#batch
        #     T_k = self.cheb_polynomials[k]  # (N,N)
        #     T_k_with_at = T_k.mul(adj)   # (N,N)*(N,N) = (N,N) 多行和为1, 按着列进行归一化
        #     theta_k = self.Theta[k]  # (in_channel, out_channel)
        #     rhs = T_k_with_at.permute(0, 2, 1).matmul(graph_signal)  # (N, N)(b, N, F_in) = (b, N, F_in) 因为是左乘，所以多行和为1变为多列和为1，即一行之和为1，进行左乘
        #     output = output + rhs.matmul(theta_k)  # (b, N, F_in)(F_in, F_out) = (b, N, F_out)
        # outputs.append(output.unsqueeze(-1))  # (b, N, F_out, 1)

class cheb_conv_K(nn.Module):
    def __init__(self, K, in_channels, out_channels, device):
        super(cheb_conv_K, self).__init__()
        self.K = K
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.DEVICE = device
        self.Theta = nn.Parameter(torch.FloatTensor(K, in_channels, out_channels).to(self.DEVICE)).to(torch.float64)

    def forward(self, x, adj):
        '''
        Chebyshev graph convolution operation
        切比雪夫图卷积运算
        :param x: (batch_size, class, F_in),邻接矩阵 (batch_size, ... class, ... class)
        :return: (batch_size, class, F_out)
        '''
        graph_signal = x  # (batch_size, class, F_in)
        T_k_with_at = adj  # (batch_size, class, class)
        output = torch.zeros_like(graph_signal).to(self.DEVICE)  # (batch_size, class, F_out)
        for k in range(self.K):
            theta_k = self.Theta[k]  # (in_channels, out_channels)
            rhs = torch.bmm(T_k_with_at, graph_signal)  # (batch_size, class, F_in) * (batch_size, class, class) -> (batch_size, class, F_in)
            output += torch.matmul(rhs, theta_k)  # (batch_size, class, F_in) * (F_in, F_out) -> (batch_size, class, F_out)

        return F.relu(output)  # (batch_size, class, F_out)

######## 自注意力机制 ########
class SelfAttention(nn.Module):
    def __init__(self, in_features, out_features):
        super(SelfAttention, self).__init__()

        self.W = nn.Linear(in_features, out_features).to(torch.float64)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x shape (len, in_features)
        h = self.W(x)  # (len, out_features)
        attention = torch.matmul(h, h.transpose(0, 1))
        attention = self.softmax(self.tanh(attention))
        attention = torch.matmul(attention, h)
        return self.tanh(attention)

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

## 引入go-go网络，间接构建蛋白质到蛋白质的网络
class FuncGCNgo(torch.nn.Module):
    def __init__(self,args):
        super(FuncGCNgo, self).__init__()
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
        self.feature = FeatureModel(max_features=args["Feature"]["max_features"],
                                    embedding_dims=args["Feature"]["embedding_dims"])
        self.goFeature = GoFeature(in_features=args["GoFeature"]["in_features"],
                                      classes=args["nb_classes"],
                                      out_channels=args["GoFeature"]["out_channels"])
        #### BatchNorm
        self.norm1 = nn.BatchNorm1d(args["GoFeature"]["in_features"]).to(torch.float64)
        self.norm2 = nn.BatchNorm1d(args["nb_classes"]).to(torch.float64)
        #self.norm3 = nn.BatchNorm1d(args["nb_classes"]).to(torch.float64)

        self.attention = Attention_layer(args["Attention_layer"]["in_channels"],
                                         args["Attention_layer"]["hid_channels"],
                                         args["Attention_layer"]["out_channels"])

        self.conv1 = cheb_conv_withSAt(in_channels=args["cheb_conv"]["in_channels"],
                                        out_channels=args["cheb_conv"]["out_channels"],
                                        device=args["device"])

        #### 输出层
        # self.fuse = nn.Parameter(data=torch.tensor(0.5),requires_grad=True)
        self.outlayer1 = nn.Linear(in_features=args["cheb_conv"]["out_channels"],out_features=1).to(torch.float64)
        #self.outlayer2 = nn.Linear(in_features=args["nb_classes"], out_features=args["nb_classes"]).to(torch.float64)

    def forward(self, data,go_adj):
        ### 输入：(batch,feature) torch.Size([64, 1256])
        ### 输出: (batch,class)
        ######### 先进行词嵌入 #########
        go_adj = torch.tensor(go_adj).to(self.args["device"])
        inputs1 = data[:, :self.MAXLEN]#torch.Size([64, 1000])
        inputs2 =  data[:, self.MAXLEN:]#torch.Size([64, 256])
        feature = self.feature(inputs1)##torch.Size([64, 1000])-->#torch.Size([64, 256])
        merged = torch.concatenate([feature, inputs2],dim=1)  # torch.Size([64, 256])+torch.Size([64, 256])-->torch.Size([64, 512])
        merged = self.norm1(merged)
        merged = self.goFeature(merged)# torch.Size([64, 512])-->torch.Size([64, nbclass, feature])

        ############## 自建图，然后图卷积提起特征 ##############
        ######### 自适应构建图 #########

        atten_adj = self.attention(merged)#torch.Size([64, nbclass, feature])-->得到矩阵 torch.Size([batch, nbclass, nbclass])
        # go_ad将大小为torch.Size([nbclass, nbclass])的矩阵扩展为大小为torch.Size([batch, nbclass, nbclass])的矩阵
        go_adj = go_adj.unsqueeze(0).expand(data.shape[0], -1, -1)
        go_adj = go_adj + atten_adj #####torch.Size([batch, nbclass, nbclass])
        go_adj = torch.softmax(go_adj,dim=2) ##### torch.Size([batch, nbclass, nbclass]) 归一化

        ######### 图卷积：汉堡包结构,引入了go-go图卷积  #########
        x = self.conv1(merged,go_adj)+ merged#torch.Size([64, nbclass, feature])torch.Size([nbclass, nbclass])-> torch.Size([64, nbclass, new_feature])
        x1 = F.dropout(x, p = self.dropout ,training=self.training)#torch.Size([64, 512])
        x1 = self.norm2(F.relu(x1))

        # x2 = self.conv2(x1,batch_adj)+ merged
        # x2 = F.dropout(x2, p=self.dropout, training=self.training)
        # x2 = F.relu(x2)

        ############## 输出层 ##############
        ### 结合输出：
        #all_x = self.fuse * x2 + (1-self.fuse) * x3
        #output =  self.norm3(self.outlayer1(x1).squeeze(dim=-1))
        #output = self.outlayer2(output)
        output = self.outlayer1(x1).squeeze(dim=-1)
        return output
        #return F.softmax(output, dim=1)
        #return torch.sigmoid(output)
        #return F.log_softmax(output, dim=1)