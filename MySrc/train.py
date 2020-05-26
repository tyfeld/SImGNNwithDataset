#!/usr/bin/env python
# coding: utf-8

# In[1]:


import glob
import torch
import pickle
import random
import math
import numpy as np
from tqdm import tqdm, trange
from model import Att, NTN, FC, SimGNN


# In[1]:


class Trainer(object):
    
    def prepare_for_train(self, batch_size = 128, epoch_num = 10, val = 0.2):
        
        self.batch_size = batch_size
        self.epoch_num = epoch_num
        self.record = [] #用来记录(train_loss, val_loss)
        
        """载入数据并划分出 验证集"""
        print("\nEnumerating unique labels.\n")
        self.training_graphs = pickle.load(open("./dataset/train_data50.pickle",'rb'))
        self.testing_graphs = pickle.load(open("./dataset/test_data50.pickle",'rb'))
        random.shuffle(self.training_graphs)
        L = len(self.training_graphs)
        div = int((1 - val) * L)
        self.val_graphs = self.training_graphs[div:L]
        self.training_graphs = self.training_graphs[0:div]
        
        """求出一共的特征数量"""
        graph_pairs = self.training_graphs + self.testing_graphs
        self.global_labels = set()
        for data in tqdm(graph_pairs):
            self.global_labels = self.global_labels.union(set(data["labels_1"]))
            self.global_labels = self.global_labels.union(set(data["labels_2"]))
        self.global_labels = list(self.global_labels)
        self.global_labels = {val:index  for index, val in enumerate(self.global_labels)}
        self.labels = len(self.global_labels)
        
        self.model = SimGNN(self.labels)
    
    def load_model(self, file):
        self.testing_graphs = pickle.load(open("./dataset/test_data.pickle",'rb'))
        self.model = torch.load(file)
        
    def save_model(self, file):
        torch.save(self.model, file)

    def save_record(self,fliename):
        pickle.dump(self.record,open(fliename,'wb'))
        
    """将读入的文件转化成网络能接受的形式"""
    def transfer_to_torch(self, data):

        new_data = dict()
        edges_1 = data["graph_1"] + [[y, x] for x, y in data["graph_1"]]
        edges_2 = data["graph_2"] + [[y, x] for x, y in data["graph_2"]]

        edges_1 = torch.from_numpy(np.array(edges_1, dtype=np.int64).T).type(torch.long)
        edges_2 = torch.from_numpy(np.array(edges_2, dtype=np.int64).T).type(torch.long)

        features_1, features_2 = [], []

        for n in data["labels_1"]:
            features_1.append([1.0 if self.global_labels[n] == i else 0.0 for i in self.global_labels.values()])

        for n in data["labels_2"]:
            features_2.append([1.0 if self.global_labels[n] == i else 0.0 for i in self.global_labels.values()])
            
        features_1 = torch.FloatTensor(np.array(features_1))
        features_2 = torch.FloatTensor(np.array(features_2))

        new_data["edge_index_1"] = edges_1
        new_data["edge_index_2"] = edges_2
        new_data["features_1"] = features_1
        new_data["features_2"] = features_2
        norm_ged = data["ged"]/(0.5*(len(data["labels_1"])+len(data["labels_2"])))
        new_data["target"] = torch.from_numpy(np.exp(-norm_ged).reshape(1, 1)).view(-1).float()

        return new_data
    
    """每个batch调用一次"""
    def process_batch(self, batch):
        
        self.optimizer.zero_grad() #梯度清0
        losses = 0
        for data in batch:
            data = self.transfer_to_torch(data)
            prediction = self.model(data)
            tmp = torch.nn.functional.mse_loss(data["target"], prediction[0])
            losses = losses + tmp
        losses.backward(retain_graph=True)
        self.optimizer.step()
        loss = losses.item()
        return loss
    
    """每个epoch调用一次，处理所有batch"""
    def train(self, epochs):
        
        random.shuffle(self.training_graphs)
        batches = []
        L = len(self.training_graphs)
        for graph in range(0, L, self.batch_size):
            batches.append(self.training_graphs[graph:graph+self.batch_size])
            
        loss_sum = 0
        L = 0
        
        self.model.train() #进入train状态
        for index, batch in tqdm(enumerate(batches), total=len(batches)):
            loss_score = self.process_batch(batch)
            L = L + len(batch)
            #loss_sum = loss_sum + loss_score * len(batch)
            loss_sum = loss_sum +loss_score
            loss = loss_sum/L
            epochs.set_description("Epoch (Loss=%g)" % round(loss, 5))
            
        val_loss = self.validate()
        self.record.append((loss, val_loss)) #记录(train_loss, val_loss)
        epochs.set_description("Epoch train_loss:[%g] val_loss:[%g]" % (round(loss, 5), round(val_loss, 5)))
    
    """每个epoch验证一次"""
    def validate(self):
        
        self.model.eval()
        losses = 0
        for data in self.val_graphs:
            data = self.transfer_to_torch(data)
            prediction = self.model(data)
            losses = losses + torch.nn.functional.mse_loss(data["target"], prediction[0])
        return losses.item()
    
    def fit(self):
        
        print("\nModel training.\n")
        
        self.optimizer = torch.optim.Adam(self.model.parameters())
        #self.model.train()
        epochs = trange(self.epoch_num, leave=True)

        for epoch in epochs:
            self.train(epochs)
            
    def calculate_loss(self,prediction, target):
    
        prediction = -math.log(prediction)
        target = -math.log(target)
        score = (prediction-target)**2
        return score

    def score(self):

        print("\n\nModel evaluation.\n")

        self.model.eval()
        losses = 0
        
        for data in tqdm(self.testing_graphs):
            data = self.transfer_to_torch(data)
            target = data["target"]
            prediction = self.model(data)
            losses = losses + torch.nn.functional.mse_loss(data["target"], prediction[0])
            #losses = losses + self.calculate_loss(data["target"], prediction[0])

        losses = losses / len(self.testing_graphs)
        print(len(self.testing_graphs))
        print("\nModel test error: " +str(round(losses.item(), 5))+".")

    
