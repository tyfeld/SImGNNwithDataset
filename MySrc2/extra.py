#!/usr/bin/env python
# coding: utf-8

# In[36]:


import torch
import numpy as np
import torch.nn as nn
from torch_geometric.nn import GCNConv
#from models import Dense, FC


# In[3]:


"""用来将图的节点标号随机交换，但不改变图的结构"""
class Dense(nn.Module):
    
    def __init__(self, in_size, out_size, bias = True, act = "relu"):
        
        super(Dense, self).__init__()
        self.conv = nn.Linear(in_size, out_size, bias)
        
        self.act = nn.ReLU()
        if(act == "sigmoid"):
            self.act = nn.Sigmoid()
            
        """预设权重"""
        nn.init.kaiming_normal_(self.conv.weight)
        if bias is True:
            self.conv.bias.data.zero_()
            
    def forward(self, x):
        return self.act(self.conv(x))

class FC(nn.Module):
    
    def __init__(self, input_size):
        
        super(FC, self).__init__()
        self.conv_1 = Dense(input_size, 16, act = "relu")
        self.conv_2 = Dense(16, 8, act = "relu")
        self.conv_3 = Dense(8, 4, act = "relu")
        self.conv_4 = Dense(4, 1, act = "sigmoid")
        
    def forward(self, x):
        ret = self.conv_2(self.conv_1(x))
        ret = self.conv_4(self.conv_3(ret))
        return ret

def random_id(graph, label):
    
    tmp_graph = []
    
    for edge in graph:
        tmp_graph.append(edge.copy())
    
    n = np.shape(label)[0]
    iid = [i for i in range(n)]    
    tmp_label = [0 for i in range(n)]
    
    np.random.shuffle(iid) #换称号,i的称号换成iid[i]
    
    for edge in tmp_graph:
        edge[0], edge[1] = iid[edge[0]], iid[edge[1]]
    for i in range(n):
        tmp_label[iid[i]] = label[i] 
    
    return tmp_graph, tmp_label


# In[4]:


"""封装 预设 权重 和激活函数的 nn.Conv2d"""
class Conv(nn.Module):
    
    def __init__(self, in_channel, out_channel, kernel_size = 3, stride = 1, padding = 1, bias = True, act="relu"):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, bias = bias)
        
        self.act = nn.ReLU()
        if (act == "lrelu"):
            self.act = nn.LeakyReLU(0.2)
        nn.init.kaiming_normal_(self.conv.weight)
        self.conv.bias.data.zero_()
        
    def forward(self, x):
        return self.act(self.conv(x))


# In[5]:


"""用多个卷积层来对 内积矩阵 进行打分"""
class Conv_module(nn.Module):
    
    def __init__(self, act = "relu"):
        
        super(Conv_module, self).__init__()
        self.conv1 = Conv(1, 8, act = act)
        self.conv2 = Conv(8, 32, act = act)
        self.conv3 = Conv(32, 16, act = act)
        self.act = nn.Sigmoid()
        
    def forward(self, x):
        
        tmp = x.view(-1, 1, x.shape[0], x.shape[1])
        
        ret = self.conv1(tmp)
        ret = self.conv2(ret)
        ret = self.conv3(ret)
        
        tmp = torch.mean(ret, dim = 2)
        ttmp = torch.mean(tmp, dim = 2)
        
        return self.act(ttmp)


# In[22]:


class Dense_GCN(nn.Module):
    
    def __init__(self, number_of_labels):
        
        super(Dense_GCN, self).__init__()
        
        self.labels = number_of_labels
        self.conv_1 = GCNConv(self.labels, 64)
        self.conv_2 = GCNConv(64, 32)
        self.conv_3 = GCNConv(32, 16)
        self.fc = Dense(64+32+16, 16, act = "relu")
    
    def forward(self, edges, features):
        
        features = self.conv_1(features, edges)
        features = nn.functional.relu(features) #注意不能使用nn.ReLU()
        features1 = nn.functional.dropout(features, p = 0.3, training=self.training) #不能使用nn.dropout(),同时应该注意training的设置

        features2 = self.conv_2(features1, edges)
        features2 = nn.functional.relu(features2)
        features2 = nn.functional.dropout(features2, p = 0.4, training=self.training) #p表示丢弃的概率

        features3 = self.conv_3(features2, edges)
        features3 = nn.functional.relu(features3)
        ret = torch.cat((features1, features2, features3), 1)
        
        ret = self.fc(ret)
        
        return ret


# In[29]:




