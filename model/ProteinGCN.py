from model.GCNDeepgo import *

### 先静态构建邻接矩阵，再自适应构调整
### 静态邻接矩阵(len,len),自适应调整（batch,batch)
class ProteinGCNgo(torch.nn.Module):
    def __init__(self,args):
        super(ProteinGCNgo, self).__init__()
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
        ######### 自建图，然后图卷积提起特征 #########

        adj = self.attention(merged)#(batch,batch)-->(batch,batch) torch.Size([64, 64])
        batch_adj = torch.softmax(batch_adj + adj,dim=1) #归一化
        #batch_adj = torch.sigmoid(batch_adj)  # 归一化

        ######### 图卷积：汉堡包结构 #########
        x = self.conv1(merged,batch_adj)#((batch,feature )(batch,batch)->(batch,len,feature ) torch.Size([64, 512])->torch.Size([64, 512])
        x1 = F.dropout(x, p=self.dropout ,training=self.training)#torch.Size([64, 512])
        x1 = self.norm2(F.relu(x1) + merged)

        # x1 = x1.unsqueeze(dim=-1).permute(0,2,1)#torch.Size([64, 512])->torch.Size([64, 512, 1])->torch.Size([64, 1, 512])
        # x1 = self.conv1d(x1) #torch.Size([64, 1, 512])->torch.Size([64, 512, 1])
        # x1 = x1.squeeze()
        #x1 = self.transformer(x1) ##加入注意力层来提取序列信息  torch.Size([64, 512])->torch.Size([64, 512])

        # x2 = self.conv2(x1,batch_adj)+ merged
        # x2 = F.dropout(x2, p=self.dropout, training=self.training)
        # x2 = F.relu(x2)

        output = self.outlayer(x1)
        return output
        #return F.softmax(output, dim=1)
        #return torch.sigmoid(output)
        #return F.log_softmax(output, dim=1)