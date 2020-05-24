#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn


"""可学习的注意模块"""
class Att(nn.Module):
    
    def __init__(self):
        
        super(Att, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(16, 16))#将张量变为可训练的
        torch.nn.init.xavier_uniform_(self.weight)
        
    def forward(self, embedding):
        
        #mean = torch.mean(embedding, dim = 0) #需要划分batch,但是使用for循环使得一次只有一张图通过该模块，故dim=0
        #global_context = torch.tanh(torch.mm(mean, self.weight))
        global_context = torch.mean(torch.matmul(embedding, self.weight),dim=0)
        transformed_global = torch.tanh(global_context)
        att_scores = torch.sigmoid(torch.mm(embedding, global_context.view(-1, 1))) #结果为长为n的得分序列,是列向量
        ret = torch.mm(torch.t(embedding), att_scores)
        return ret


"""对两个向量的相似度进行打分"""
class NTN(nn.Module):
    
    def __init__(self):
        
        super(NTN, self).__init__()
        self.W = torch.nn.Parameter(torch.Tensor(16, 16, 16)) #最后一维是K
        self.V = torch.nn.Parameter(torch.Tensor(16, 32))
        self.bias =  torch.nn.Parameter(torch.Tensor(16, 1))
        torch.nn.init.xavier_uniform_(self.W)
        torch.nn.init.xavier_uniform_(self.V)
        torch.nn.init.xavier_uniform_(self.bias)
        
    def forward(self, embedding_1, embedding_2): #注意两个向量都是列向量
        
        A = torch.mm(torch.t(embedding_1), self.W.view(16,-1)) 
        A = A.view(16, 16) 
        A = torch.mm(torch.t(A), embedding_2)
        
        B = torch.cat((embedding_1, embedding_2))
        B = torch.mm(self.V, B)
        
        ret = nn.functional.relu(A + B + self.bias)
        
        return ret