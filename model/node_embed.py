import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import json
def reshape(values):
    #values = np.hstack(values).reshape(len(values), len(values[0]))
    values = np.hstack(values).reshape(len(values), len(values[0]))
    return values
def get_values(data_frame,MAXLEN):
    print((data_frame['labels'].values.shape))
    labels = reshape(data_frame['labels'].values)
    # arg_labels=np.argmax(labels,axis=1)
    # arg_labels=np.reshape(arg_labels,newshape=(-1,1))
    # ngrams = sequence.pad_sequences(
    #     data_frame['ngrams'].values, maxlen=MAXLEN)### 进行0填充
    # 将序列 padded到最大长度MAXLEN
    # padded_ngrams = pad_sequence(data_frame['ngrams'].values, batch_first=True, padding_value=0)
    # ngrams = padded_ngrams[:, :MAXLEN]
    #ngrams = torch.nn.ConstantPad2d(padding=MAXLEN, value=0)
    #ngrams=np.pad(data_frame['ngrams'].values, ((0, 0), (0, MAXLEN-)), 'constant', constant_values=(0, 0))
    ngrams=data_frame['ngrams'].values
    ngrams = np.array([
        np.pad(ngrams[i], (0, MAXLEN- len(ngrams[i])), 'constant') for i in range(len(ngrams))
    ])
    ngrams = reshape(ngrams)
    rep = reshape(data_frame['embeddings'].values)
    # data = (ngrams, rep)
    ####在这里进行拼接
    data = np.concatenate((ngrams, rep), axis=-1)
    return data, labels

def load_data_mask(DATA_ROOT,FUNCTION,ORG,MAXLEN):
    ###num_train = 1000
    #df = pd.read_pickle(DATA_ROOT +FUNCTION + '.pkl')
    df1 = pd.read_pickle(DATA_ROOT + 'train' + '-' + FUNCTION + '.pkl')
    df2 = pd.read_pickle(DATA_ROOT + 'test' + '-' + FUNCTION + '.pkl')
    df=pd.concat([df1,df2],axis=0)
    gos = df["gos"].values
    sequences = df["sequences"].values
    proteins = df["proteins"].values
    seed=1
    n = len(df)
    index = df.index.values
    np.random.seed(1)
    shuffled_idx = np.random.permutation(np.array(range(n)))  # 已经被随机打乱
    train_idx = shuffled_idx[:int(0.90 * n)]
    val_idx = shuffled_idx[int(0.9 * n): int(0.901 * n)]
    test_idx = shuffled_idx[int(0.90 * n):]
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[train_idx] = 1
    val_mask[val_idx]=1
    test_mask[test_idx]=1

    train_df = df.loc[train_idx]
    valid_df = df.loc[val_idx]
    test_df= df.loc[test_idx]

    # Filter by type
    # org_df = pd.read_pickle('data/prokaryotes.pkl')
    # orgs = org_df['orgs']
    # test_df = test_df[test_df['orgs'].isin(orgs)]

    # train = get_values(train_df)
    # valid = get_values(valid_df)
    # test = get_values(test_df)
    all_values = get_values(df,MAXLEN)
    ## 数据预处理 ：在网络前进行处理
    #all_values = (standardization(all_values[0]),all_values[1])
    return all_values,train_mask,val_mask,test_mask,shuffled_idx,gos,sequences,proteins

def get_gene_ontology(filename):
    # Reading Gene Ontology from OBO Formatted file
    go = dict()
    obj = None
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line == '[Term]':
                if obj is not None:
                    go[obj['id']] = obj
                obj = dict()
                obj['is_a'] = list()
                obj['part_of'] = list()
                obj['regulates'] = list()
                obj['is_obsolete'] = False
                continue
            elif line == '[Typedef]':
                obj = None
            else:
                if obj is None:
                    continue
                l = line.split(": ")
                if l[0] == 'id':
                    obj['id'] = l[1]
                elif l[0] == 'is_a':
                    obj['is_a'].append(l[1].split(' ! ')[0])
                elif l[0] == 'name':
                    obj['name'] = l[1]
                elif l[0] == 'is_obsolete' and l[1] == 'true':
                    obj['is_obsolete'] = True
    if obj is not None:
        go[obj['id']] = obj
    for go_id in list(go.keys()):
        if go[go_id]['is_obsolete']:
            del go[go_id]
    for go_id, val in go.items():
        if 'children' not in val:
            val['children'] = set()
        for p_id in val['is_a']:
            if p_id in go:
                if 'children' not in go[p_id]:
                    go[p_id]['children'] = set()
                go[p_id]['children'].add(go_id)
    return go

### 构建 go-go之间的邻接矩阵
def build_GoAdj(args):
    adj_mx = np.zeros(shape=(args["nb_classes"],args["nb_classes"]))
    triples = []
    ######### 读取关系 triplet ########
    for node_id in args["functions"]:
        childs = set(args["go"][node_id]['children']).intersection(args["func_set"])
        if len(childs) > 0:
            for ch_id in childs:
                triples.append((node_id, 'is_a', ch_id))
    Functions = args["functions"]
    ########## 构建邻接矩阵 ########
    for s, r, o in triples:
        s_index = np.where(Functions==s)[0][0]
        o_index = np.where(Functions == o)[0][0]
        adj_mx[s_index, o_index] = 1

    return adj_mx

conjName=r"..\conj\GCNDeepgo.json"
def read_config():
    #超参数：配置信息
    with open(conjName,encoding='UTF-8') as json_file:
        config = json.load(json_file)
    config["org"]= None
    config["supports"] = None
    config["node_names"] = set()
    config["go"] = get_gene_ontology(config["go_path"])
    func_df = pd.read_pickle(config["DATA_ROOT"] + config["function"] + '.pkl')
    functions = func_df['functions'].values
    config["functions"] = functions
    config["func_set"] = set(functions)
    config["nb_classes"] = len(config["func_set"])
    BIOLOGICAL_PROCESS = 'GO:0008150'
    MOLECULAR_FUNCTION = 'GO:0003674'
    CELLULAR_COMPONENT = 'GO:0005575'
    FUNC_DICT = {
        'cc': CELLULAR_COMPONENT,
        'mf': MOLECULAR_FUNCTION,
        'bp': BIOLOGICAL_PROCESS}
    config["Go_id"] = FUNC_DICT[config["function"]]
    config["MAXLEN"] = config["MAXLEN"]
    return config


class Tree2Vec(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, output_dim=16):
        super(Tree2Vec, self).__init__()
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.fc1 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=output_dim)

    def forward(self, adjacency_matrix,args):#
        # Embedding layer
        adjacency_matrix = torch.tensor(adjacency_matrix)#torch.Size([589, 589])
        embedded = self.embedding(adjacency_matrix.long())#torch.Size([589, 589, 32]),node i 到nodej之间的特征

        # Aggregation layer
        aggregated = torch.sum(embedded, dim=1)#torch.Size([589, 32])，node i的特征

        # Fully connected layers
        hidden = F.relu(self.fc1(aggregated))
        output = self.fc2(hidden)#torch.Size([589, 16])
        return output

# 定义自监督学习任务
def self_supervised_task(embedded):
    # 在这里定义你的自监督学习任务
    # 例如，可以使用节点的上下文信息来预测节点的标签
    # 这里只是一个示例，你可以根据自己的需求来设计任务

    # 随机选择一个节点作为正样本
    positive_node = torch.randint(0, embedded.size(0), (1,))#589
    positive_embedding = embedded[positive_node]

    # 随机选择一个节点作为负样本
    negative_node = torch.randint(0, embedded.size(0), (1,))
    negative_embedding = embedded[negative_node]

    # 计算正样本和负样本之间的相似度
    similarity = torch.cosine_similarity(positive_embedding, negative_embedding)

    return similarity

if __name__ == '__main__':

    # 创建邻接矩阵
    args = read_config()
    all_values, train_mask, val_mask, test_mask, shuffled_idx, gos, sequences, _ = load_data_mask(
        DATA_ROOT=args["DATA_ROOT"], FUNCTION=args["function"],
        ORG=args["org"], MAXLEN=args["MAXLEN"])
    ########################## 加载数据为batch ########################
    x = all_values[0]
    y = all_values[1]
    args["functions"] = args["functions"][:y.shape[1]]
    args["nb_classes"] = y.shape[1]
    adj = build_GoAdj(args=args)
    adj = torch.tensor(adj).to(torch.float64)#1表示i和j存在，

    input_dim = adj.shape[0]  # 输入维度，即节点的数量
    hidden_dim = 32  # 隐藏层维度
    output_dim = 16  # 输出维度，即节点特征的维度

    # 创建模型
    model = Tree2Vec(input_dim, hidden_dim, output_dim)
    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    min_loss = float('inf')
    best_epoch = 0
    function = args["function"]
    # 自监督学习训练
    for epoch in range(100):
        # 前向传播
        embedded = model(adj,args)
        similarity = self_supervised_task(embedded)

        # 计算损失
        loss = criterion(similarity, torch.ones_like(similarity))#loss最小化

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 打印训练进度
        print(f"Epoch: {epoch+1}, Loss: {loss.item()}")
        if loss <= min_loss:
            min_loss = loss
            best_epoch = epoch
            torch.save(model.state_dict(), f"..\\experiments\\best_embed_{function}.pth")


    # 加载最佳模型
    print("the best epoch is", best_epoch)
    best_model = Tree2Vec(input_dim, hidden_dim, output_dim)
    best_model.load_state_dict(torch.load(f"..\experiments\\best_embed_{function}.pth"))

    # 使用训练好的模型进行预测
    node_features = best_model(adj,args)
    print(node_features) #torch.Size([16])

