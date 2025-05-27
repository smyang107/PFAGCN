import torch
from torch_geometric.nn import GCNConv
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from utils_all import get_parents,get_gene_ontology,get_function_node,get_node_name
#from utils_all import *
import pandas as pd
import numpy as np

###### 学习序列信息得到图结构 ######
class Attention_layer(nn.Module):
    def  __init__(self, in_channels, hid_channels, out_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_channels, hid_channels),
            nn.ReLU(),
            nn.Linear(hid_channels, out_channels),
        )
    def forward(self, x):
        ### 输入：(batch,len/batch,feature )或者 (batch,feature )
        ### 输出：每个蛋白质对每个蛋白质的邻接矩阵(batch, seq_len, seq_len) 或者 (batch)
        ####方法一
        if x.dtype == torch.float64:
            x = x.float()
        Z = self.model(x)#torch.Size([64, 512])-->torch.Size([64, 512])
        score = torch.mm(Z, Z.t())
        #W = torch.sigmoid(score)  # 归一化为0-1权重
        W = F.softmax(score, dim=1).to(torch.float64) # 归一化为0-1权重
        return W

        ####方法二
        # z = self.model(x)
        # W = torch.bmm(z, z.transpose(1, 2))  # (batch, seq_len, seq_len)


#### 自定义的图卷积
class cheb_conv_withSAt(nn.Module):
    '''
    K-order chebyshev graph convolution
    时间维度上的图卷积
    '''

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
        self.Theta=nn.Parameter(torch.FloatTensor(in_channels, out_channels).to(self.DEVICE)).to(torch.float64)

    def forward(self, x, spatial_attention):
        '''
        Chebyshev graph convolution operation
        切比雪夫图卷积运算
        :param x: (batch_size, F_in) torch.Size([64, 512])
        :return: (batch_size, F_out)
        '''
        ################## 写法1 ##################
        graph_signal = x  # (b, F_in)

        T_k_with_at = spatial_attention  # torch.Size([64, 64])

        theta_k = self.Theta  # (in_channel, out_channel)

        rhs = T_k_with_at.matmul(graph_signal)  # (b , b)(b, F_in) = (b,F_in) 因为是左乘，所以多行和为1变为多列和为1，即一行之和为1，进行左乘
        #rhs=rhs.to(torch.float64)
        output = rhs.matmul(theta_k)  # (b,F_in)(F_in, F_out) = (b, F_out)

        ################## 写法2 ##################
        # graph_signal = x  # (b, F_in)
        #
        # T_k_with_at = spatial_attention  # torch.Size([64, 64])
        #
        # output = T_k_with_at.matmul(graph_signal)  # (b,F_in)(F_in, F_out) = (b, F_out)

        return F.relu(output)  # (b,  F_out)

class cheb_conv_K(nn.Module):
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
        super(cheb_conv_K, self).__init__()
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
        :param x: (batch_size, F_in) torch.Size([64, 512])
        :return: (batch_size, F_out)
        '''
        ################## 写法1 ##################
        graph_signal = x  # (b, F_in)

        T_k_with_at = spatial_attention  # torch.Size([64, 64])
        output = torch.zeros_like(graph_signal).to(self.DEVICE)  # (batch_size, class, F_out)
        for k in range(self.K):
            theta_k = self.Theta[k]  # (in_channels, out_channels)
            rhs = T_k_with_at.matmul(graph_signal)  # (b , b)(b, F_in) = (b,F_in) 因为是左乘，所以多行和为1变为多列和为1，即一行之和为1，进行左乘
            # rhs=rhs.to(torch.float64)
            output += rhs.matmul(theta_k)  # (b,F_in)(F_in, F_out) = (b, F_out)
        return output

####### 点嵌入模型 #######
class FeatureModel(nn.Module):
    def __init__(self, max_features=8001, embedding_dims=128):
        super(FeatureModel, self).__init__()

        self.embedding = nn.Embedding(max_features, embedding_dims)
        #self.conv1d = nn.Conv1d(in_channels=1000, out_channels=256, kernel_size=128)
        self.norm = nn.BatchNorm1d(1000)  # 数据标准化模块
        #self.norm = nn.BatchNorm2d(1000)
        self.conv1 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=125, dilation=2, padding=0)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=1, kernel_size=125, dilation=4, padding=0)
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
        # x = F.relu(self.conv3(x))
        # x = self.conv4(x)
        #x = self.max_pool1d(x)
        x = self.flatten(x) # torch.Size([64, 1, 256])->torch.Size([64, 256])
        return x


######分层分类模型
def get_layers(inputs,node_names,GO_ID,go,func_set,functions):
    layers = {}
    q = deque()

    name = get_node_name(GO_ID,node_names)
    layers[GO_ID] = {'net': inputs}

    for node_id in go[GO_ID]['children']:
        if node_id in func_set:
            q.append((node_id, inputs))

    while len(q) > 0:
        node_id, net = q.popleft()
        parent_nets = [inputs]

        for p_id in get_parents(go, node_id):
            if p_id in func_set:
                parent_nets.append(layers[p_id]['net'])

        if len(parent_nets) > 1:
            name = get_node_name(node_id) + '_parents'
            net = torch.cat(parent_nets, dim=1)

        name = get_node_name(node_id,node_names)
        net, output = get_function_node(name, net) #在这里获得网络和输出

        if node_id not in layers:
            layers[node_id] = {'net': net, 'output': output}
            for n_id in go[node_id]['children']:
                if n_id in func_set and n_id not in layers:
                    q.append((n_id, net))

    for node_id in functions:
        childs = set(go[node_id]['children']).intersection(func_set)
        if len(childs) > 0:
            outputs = [layers[node_id]['output']]
            for ch_id in childs:
                outputs.append(layers[ch_id]['output'])

            name = get_node_name(node_id) + '_max'
            layers[node_id]['output'] = torch.max(torch.stack(outputs), dim=0)[0]

    return layers


class ProteinFunctionNode(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.dense1 = nn.Linear(in_features, 256)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        x = self.sigmoid(x)
        return x

class Hierarchical_classifer(torch.nn.Module):
    def __init__(self,args):
        """
            在蛋白质的GO结构上构建功能预测模型
            通过父子GO term连接反映拓扑依赖性
        """
        super(Hierarchical_classifer, self).__init__()
        self.node_names = set()
        self.go = get_gene_ontology(args["go_path"])
        func_df = pd.read_pickle(args["DATA_ROOT"] + args["function"] + '.pkl')
        functions = func_df['functions'].values
        self.functions = functions
        self.func_set = set(functions)
        BIOLOGICAL_PROCESS = 'GO:0008150'
        MOLECULAR_FUNCTION = 'GO:0003674'
        CELLULAR_COMPONENT = 'GO:0005575'
        FUNC_DICT = {
            'cc': CELLULAR_COMPONENT,
            'mf': MOLECULAR_FUNCTION,
            'bp': BIOLOGICAL_PROCESS}
        self.GO_ID = FUNC_DICT[args["function"]]
        self.layers = {}
        self.name = get_node_name(self.GO_ID, self.node_names)

        self.node_names = {go_id: self._get_node_name(go_id) for go_id in self.func_set}

        # 定义线性层
        self.func_layers = nn.ModuleList()
        for go_id in self.func_set:
            self.func_layers.append(
                nn.Linear(512, 512).to(torch.float64))
            # self.func_layers.append(
            #     nn.Linear(256, 1).to(torch.float64))
        self.relu = nn.ReLU()

    def _get_node_name(self, go_id):
        return go_id.split(":")[1] + "_layer"

    def forward(self, inputs):
        layers = {}
        q = deque()

        go_id = list(self.go.keys())[0]  # 获取GO顶层id
        layers[go_id] = self.func_layers[0](inputs)

        q.append((go_id, layers[go_id]))

        # BFS建立层次
        while len(q) > 0:
            cur_id, prev_layer = q.popleft()
            for child_id in self.go[cur_id]['children']:
                if child_id in self.func_set:
                    inp = prev_layer
                    if child_id in layers:
                        inp = torch.cat([prev_layer, layers[child_id]], dim=1)
                    layers[child_id] = self.relu(self.func_layers[self.node_names[child_id]](inp))
                    q.append((child_id, layers[child_id]))

        outputs = [l for l in layers.values()] #torch.Size([64, 512])
        return torch.cat(outputs, dim=1)

## 实现：不固定输入维度的线性层
class MyLinear(nn.Module):
    def __init__(self, output_dim,args):
        super(MyLinear, self).__init__()
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.randn(output_dim),requires_grad=True).to(torch.float64).to(args["device"])
        self.bias = torch.zeros(self.output_dim).to(torch.float64).to(args["device"])
    def forward(self, inputs):
        input_dim = inputs.shape[-1]
        weight = self.weight.unsqueeze(0).expand(input_dim, self.output_dim)
        outputs = torch.matmul(inputs, weight) + self.bias
        return outputs

class Hierarchical_classifer_new(torch.nn.Module):
    def __init__(self,args):
        """
            在蛋白质的GO结构上构建功能预测模型
            通过父子GO term连接反映拓扑依赖性
        """
        super(Hierarchical_classifer_new, self).__init__()
        self.node_names = set()
        self.go = get_gene_ontology(args["go_path"])
        func_df = pd.read_pickle(args["DATA_ROOT"] + args["function"] + '.pkl')
        functions = func_df['functions'].values
        self.functions = functions
        self.func_set = set(functions)
        BIOLOGICAL_PROCESS = 'GO:0008150'
        MOLECULAR_FUNCTION = 'GO:0003674'
        CELLULAR_COMPONENT = 'GO:0005575'
        FUNC_DICT = {
            'cc': CELLULAR_COMPONENT,
            'mf': MOLECULAR_FUNCTION,
            'bp': BIOLOGICAL_PROCESS}
        self.GO_ID = FUNC_DICT[args["function"]]
        self.layers = {}
        self.name = get_node_name(self.GO_ID, self.node_names)

        self.node_names = {go_id: self._get_node_name(go_id) for go_id in self.func_set}
        self.args = args
        # 定义线性层
        self.func_layers = nn.ModuleDict()
        for i, go_id in enumerate(self.node_names):
            ### 一个node的结构
            linear = nn.ModuleList()
            #linear.append(nn.Linear(None, 128).to(torch.float64))
            linear.append(MyLinear(output_dim=128,args=args).to(torch.float64))
            linear.append(nn.ReLU())
            linear.append(nn.Linear(128, 1).to(torch.float64))
            #self.func_layers.append(linear)
            # 设置名称
            self.func_layers[self.node_names[go_id]] = linear.to(args["device"])
        self.relu = nn.ReLU()


    def _get_node_name(self, go_id):
        return go_id.split(":")[1] + "_layer"

    def get_parents(self,gos, go_id):
        go_set = set()
        for parent_id in gos[go_id]['is_a']:
            if parent_id in gos:
                go_set.add(parent_id)
        return go_set

    def forward(self, inputs):
        q = deque()
        layers = {}
        name = self._get_node_name(self.GO_ID)
        noneOuput = torch.empty((64,0)).to(self.args["device"])
        layers[self.GO_ID] = {'net': inputs,'output':noneOuput}
        for node_id in self.go[self.GO_ID]['children']:  ##获得gon类别的孩子（更具体的类别）
            if node_id in self.func_set:
                q.append((node_id, inputs))
        while len(q) > 0:  ###安装go功能一个一个处理
            node_id, net = q.popleft()
            parent_nets = [inputs]
            for p_id in self.get_parents(self.go, node_id):
                if p_id in self.func_set:
                    parent_nets.append(layers[p_id]['net'])
            if len(parent_nets) > 1:  # 1
                name = self._get_node_name(node_id) + '_parents'
                # TypeError: expected Tensor as element 1 in argument 0, but got ModuleList
                net = torch.concatenate(parent_nets, dim=1)
            name = self._get_node_name(node_id)
            net = self.func_layers[name][0](net) #获得输出 torch.Size([64, 512])->torch.Size([64, 512])
            net = self.func_layers[name][1](net)
            output = self.func_layers[name][2](net) #torch.Size([64, 512])->torch.Size([64, 1])
            if node_id not in layers:
                layers[node_id] = {'net': net, 'output': output}
                for n_id in self.go[node_id]['children']:
                    if n_id in self.func_set and n_id not in layers:
                        ok = True
                        for p_id in self.get_parents(self.go, n_id):
                            if p_id in self.func_set and p_id not in layers:
                                ok = False
                        if ok:
                            q.append((n_id, net))

        for node_id in self.functions:
            childs = set(self.go[node_id]['children']).intersection(self.func_set)
            if len(childs) > 0:
                #outputs = [layers[node_id]['output']]#这里设为list
                outputs = torch.reshape(layers[node_id]['output'],shape=(-1,1))
                for ch_id in childs:
                    #outputs.append(layers[ch_id]['output'])
                    outputs = torch.concat((outputs,torch.reshape(layers[ch_id]['output'],shape=(-1,1))),dim=1)
                name = self._get_node_name(node_id) + '_max'
                # layers[node_id]['output'] = merge(outputs, mode='max', name=name)
                layers[node_id]['output'] = torch.reshape(torch.max(outputs,dim=1).values,shape=(-1,1))

        #outputs = [layers[l]['output'] for l in layers.keys()]
        #outputs = [np.where(self.functions==l)[0][0] for l in layers.keys()]
        sort_outputs = []
        for i in range(len(self.functions)):
            sort_outputs.append(layers[self.functions[i]]['output'])
        return torch.cat(sort_outputs, dim=1)

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

### 先静态构建邻接矩阵，再自适应构调整
### 静态邻接矩阵(len,len),自适应调整（batch,batch)
class GCN_static(torch.nn.Module):
    def __init__(self,args):
        super(GCN_static, self).__init__()
        self.attention = Attention_layer(args["Attention_layer"]["in_channels"], args["Attention_layer"]["hid_channels"], args["Attention_layer"]["out_channels"])
        self.feature = FeatureModel(max_features=args["Feature"]["max_features"],embedding_dims=args["Feature"]["embedding_dims"])
        self.conv1 = cheb_conv_withSAt(args["cheb_conv"]["in_channels"], args["cheb_conv"]["out_channels"],args["device"])
        #self.conv1d = nn.Conv1d(in_channels=1, out_channels=args["cheb_conv"]["out_channels"], kernel_size=args["cheb_conv"]["out_channels"]).to(torch.float64)
        self.transformer = SelfAttention(in_features=args["transformer"]["in_channels"],
                                        out_features=args["transformer"]["out_channels"])
        self.conv2 = cheb_conv_withSAt(args["cheb_conv"]["in_channels"], args["cheb_conv"]["out_channels"],args["device"])
        self.training = True
        self.dropout = 0.2
        self.node_names = args["node_names"]
        self.go = args["go"]
        self.functions=args["functions"]
        self.func_set = set(self.functions)
        self.Go_id = args["Go_id"]
        self.MAXLEN = args["MAXLEN"]
        # #这里加入自适应调整
        # if args["addaptadj"]:
        #     if args["supports"] is None:
        #         self.supports = []
        #     self.nodevec1 = nn.Parameter(torch.randn(args["batch_size"], 10).to(args["device"]), requires_grad=True).to(args["device"])
        #     self.nodevec2 = nn.Parameter(torch.randn(10, args["batch_size"]).to(args["device"]), requires_grad=True).to(args["device"])
        #### BatchNorm
        self.norm1 = nn.BatchNorm1d(args["cheb_conv"]["out_channels"]).to(torch.float64)
        self.norm2 = nn.BatchNorm1d(args["cheb_conv"]["out_channels"]).to(torch.float64)
        ###分层分类
        self.hie_class = Hierarchical_classifer(args)
        #输出层
        self.fuse = nn.Parameter(data=torch.tensor(0.5),requires_grad=True)
        self.outlayer=nn.Linear(in_features=args["cheb_conv"]["out_channels"],out_features=args["nb_classes"]).to(torch.float64)

    def forward(self, data,batch_adj):
        ### 输入：(batch,feature) torch.Size([64, 1256])
        ### 输出: (batch,class)
        ######### 先进行词嵌入 #########
        inputs1 = data[:, :self.MAXLEN]#torch.Size([64, 1000])
        inputs2 =  data[:, self.MAXLEN:]#torch.Size([64, 256])
        feature=self.feature(inputs1)##torch.Size([64, 1000])-->#torch.Size([64, 256])
        merged = torch.concatenate([feature, inputs2],dim=1)  # torch.Size([64, 256])+torch.Size([64, 256])-->torch.Size([64, 512])
        merged = self.norm1(merged)
        ############## 分支一：自建图，然后图卷积提起特征 ##############
        ######### 自适应构建图 #########

        adj = self.attention(merged)#(batch,batch)-->(batch,batch) torch.Size([64, 64])
        batch_adj = torch.sigmoid(batch_adj + adj) #归一化
        #batch_adj = torch.sigmoid(batch_adj)  # 归一化

        ######### 图卷积：汉堡包结构 #########
        x = self.conv1(merged,batch_adj)#((batch,feature )(batch,batch)->(batch,len,feature ) torch.Size([64, 512])->torch.Size([64, 512])
        x1 = F.dropout(x, p=self.dropout ,training=self.training)#torch.Size([64, 512])
        x1 = self.norm2(F.relu(x1) + merged)

        # x1 = x1.unsqueeze(dim=-1).permute(0,2,1)#torch.Size([64, 512])->torch.Size([64, 512, 1])->torch.Size([64, 1, 512])
        # x1 = self.conv1d(x1) #torch.Size([64, 1, 512])->torch.Size([64, 512, 1])
        # x1 = x1.squeeze()
        x1 = self.transformer(x1) ##加入注意力层来提取序列信息  torch.Size([64, 512])->torch.Size([64, 512])

        x2 = self.conv2(x1,batch_adj)+ merged
        x2 = F.dropout(x2, p=self.dropout, training=self.training)
        x2 = F.relu(x2)
        ############## 分支二：知识图谱 ##############
        ######################## 第二层：构建了多层感知机结构,每个GO term用一层表示。通过这个结构,可以学习父子GO term之间的关系,反映蛋白质间的拓扑互作用信息 ########################
        #layers = get_layers(inputs=merged,node_names=self.node_names,GO_ID=self.Go_id,go=self.go,func_set=self.func_set,functions=self.functions)  # TensorShape([Dimension(None), Dimension(None)])-->
        x3 = F.relu(self.hie_class(merged))+ merged#torch.Size([64, 512])--> torch.Size([64, 512])
        x3 = F.dropout(x3, p=self.dropout, training=self.training)
        # output_models = []
        # for i in range(len(self.functions)):
        #     output_models.append(x3[self.functions[i]]['output'])
        # net = merge(output_models, mode='concat', concat_axis=1)
        # net = tf.concat(output_models, axis=1)
        # x3 = torch.concatenate((output_models),dim=1)

        ############## 输出层 ##############
        ### 结合输出：
        all_x = self.fuse * x2 + (1-self.fuse) * x3
        output =  self.outlayer(all_x)
        return output
        #return F.softmax(output, dim=1)
        #return torch.sigmoid(output)
        #return F.log_softmax(output, dim=1)


### 先静态构建邻接矩阵，再自适应构调整
### 静态邻接矩阵(len,len),自适应调整（batch,batch)
class GCN_static_new(torch.nn.Module):
    def __init__(self,args):
        super(GCN_static_new, self).__init__()
        self.training = True
        self.dropout = 0.2
        self.node_names = args["node_names"]
        self.go = args["go"]
        self.functions=args["functions"]
        self.func_set = set(self.functions)
        self.Go_id = args["Go_id"]
        self.MAXLEN = args["MAXLEN"]
        self.attention = Attention_layer(args["Attention_layer"]["in_channels"], args["Attention_layer"]["hid_channels"], args["Attention_layer"]["out_channels"])
        self.feature = FeatureModel(max_features=args["Feature"]["max_features"],embedding_dims=args["Feature"]["embedding_dims"])
        self.conv1 = cheb_conv_withSAt(args["cheb_conv"]["in_channels"], args["cheb_conv"]["out_channels"],args["device"])
        #self.conv1d = nn.Conv1d(in_channels=1, out_channels=args["cheb_conv"]["out_channels"], kernel_size=args["cheb_conv"]["out_channels"]).to(torch.float64)
        # self.transformer = SelfAttention(in_features=args["transformer"]["in_channels"],
        #                                 out_features=args["transformer"]["out_channels"])
        self.conv2 = cheb_conv_withSAt(args["cheb_conv"]["in_channels"], args["cheb_conv"]["out_channels"],args["device"])
        #### BatchNorm
        self.norm1 = nn.BatchNorm1d(args["cheb_conv"]["out_channels"]).to(torch.float64)
        self.norm2 = nn.BatchNorm1d(args["cheb_conv"]["out_channels"]).to(torch.float64)
        ###### 分层分类
        # self.hie_class = Hierarchical_classifer_new(args)
        #### 输出层
        #self.fuse = nn.Parameter(data=torch.tensor(0.5),requires_grad=True)
        self.outlayer=nn.Linear(in_features=args["cheb_conv"]["out_channels"],out_features=args["nb_classes"]).to(torch.float64)

    def forward(self, data,batch_adj):
        ### 输入：(batch,feature) torch.Size([64, 1256])
        ### 输出: (batch,class)
        ######### 先进行词嵌入 #########
        inputs1 = data[:, :self.MAXLEN]#torch.Size([64, 1000])
        inputs2 =  data[:, self.MAXLEN:]#torch.Size([64, 256])
        feature=self.feature(inputs1)##torch.Size([64, 1000])-->#torch.Size([64, 256])
        merged = torch.concatenate([feature, inputs2],dim=1)  # torch.Size([64, 256])+torch.Size([64, 256])-->torch.Size([64, 512])
        merged = self.norm1(merged)
        ############## 分支一：自建图，然后图卷积提起特征 ##############
        ######### 自适应构建图 #########

        adj = self.attention(merged)#(batch,batch)-->(batch,batch) torch.Size([64, 64])
        batch_adj = torch.sigmoid(batch_adj + adj) #归一化
        #batch_adj = torch.sigmoid(batch_adj)  # 归一化

        ######### 图卷积：汉堡包结构 #########
        x = self.conv1(merged,batch_adj)#((batch,feature )(batch,batch)->(batch,len,feature ) torch.Size([64, 512])->torch.Size([64, 512])
        x1 = F.dropout(x, p=self.dropout ,training=self.training)#torch.Size([64, 512])
        x1 = self.norm2(F.relu(x1) + merged)

        # x1 = x1.unsqueeze(dim=-1).permute(0,2,1)#torch.Size([64, 512])->torch.Size([64, 512, 1])->torch.Size([64, 1, 512])
        # x1 = self.conv1d(x1) #torch.Size([64, 1, 512])->torch.Size([64, 512, 1])
        # x1 = x1.squeeze()
        #x1 = self.transformer(x1) ##加入注意力层来提取序列信息  torch.Size([64, 512])->torch.Size([64, 512])

        x2 = self.conv2(x1,batch_adj)+ merged
        x2 = F.dropout(x2, p=self.dropout, training=self.training)
        x2 = F.relu(x2)
        ############## 分支二：知识图谱 ##############
        ######################## 第二层：构建了多层感知机结构,每个GO term用一层表示。通过这个结构,可以学习父子GO term之间的关系,反映蛋白质间的拓扑互作用信息 ########################
        #layers = get_layers(inputs=merged,node_names=self.node_names,GO_ID=self.Go_id,go=self.go,func_set=self.func_set,functions=self.functions)  # TensorShape([Dimension(None), Dimension(None)])-->
        # x3 = F.relu(self.hie_class(merged))#torch.Size([64, 512])--> torch.Size([64, 512])
        # x3 = F.dropout(x3, p=self.dropout, training=self.training)

        ############## 输出层 ##############
        ### 结合输出：
        #all_x = self.fuse * x2 + (1-self.fuse) * x3
        #output =  self.fuse * self.outlayer(x2) + (1 - self.fuse) * x3
        output = self.outlayer(x2)
        return output
        #return F.softmax(output, dim=1)
        #return torch.sigmoid(output)
        #return F.log_softmax(output, dim=1)

