#!/usr/bin/env python
# -*- coding:utf-8 -*-
############################
#File Name:
#Author: yanbin
#Mail: yanbin918@gmail.com
#Created Time: 2017-08-10 10:33:09
############################
import math ,operator,random
import sys
sys.path.append('../')
from ml_latest_small.data import *

class ItemCFModel(object):   
    def __init__(self,train,W,K=5,N=5):
        self.train = train
        self.W = W
        self.K = K
        self.N = N

    def recommendation(self,user_id,config=None):
        W = self.W
        K = self.K
        N = self.N
        if config is not None:
            try:
                K = config['K']
                N = config['N']
            except:
                print "config = {'K':nearest nums,'N':topN nums}"
                return
        train = self.train
        rank=dict()
        if user_id not in train:
            print 'The user is not in traindata'
            return
        ru = train[user_id]
        for i,rui in ru.items():
            for j,wij in sorted(W[i].items(),key=operator.itemgetter(1),reverse=True)[0:K]:
                if j in ru:
                    continue
                if j not in rank:
                    rank[j]=0
                rank[j]+=wij*rui
        topN = sorted(rank.items(),key=operator.itemgetter(1),reverse=True)[0:N]
        return dict(topN)





class ItemCF(object):
    def itemSimilarity(self,train):            
#calculate co-rated items between users
        C = dict()
        N = dict()
        for user,items in train.items():
            for i in items:
                if i not in C:
                    C[i]=dict()
                if i not in N:
                    N[i]=0
                N[i] +=1
                for j in items:
                    if i==j:
                        continue
                    if j not in C[i]:
                        C[i][j]=0
                    C[i][j]+=1
        #calculate finial similarity matrix W
        W = dict()
        for i,related_items in C.items():
            W[i]=dict()
            for j,cij in related_items.items():
                W[i][j] = cij/math.sqrt(N[i] * N[j])
        return W

    def itemSimilarity2(self,train):
        #calculate co-rated items between users
        C = dict()
        N = dict()
        for user,items in train.items():
            for i in items:
                if i not in C:
                    C[i]=dict()
                if i not in N:
                    N[i]=0
                N[i] +=1
                for j in items:
                    if i==j:
                        continue
                    if j not in C[i]:
                        C[i][j]=0
                    C[i][j]+=1/math.log(1+len(items)*1.0)
        #calculate finial similarity matrix W
        W = dict()
        for i,related_items in C.items():
            W[i]=dict()
            for j,cij in related_items.items():
                W[i][j] = cij/math.sqrt(N[i] * N[j])
        return W
    
   
    def train(self,train,K=5,Normalization=False):
        W = self.itemSimilarity2(train)
        if Normalization :
            W = self.Normalization(W)
        return ItemCFModel(train,W)
        
        
    def Normalization(self,W):
        for i,items in W.items():
            m = max(items.values())
            for j,wij in items.items():
                W[i][j]=wij/m
        return W

if __name__ == '__main__':
    itemCF = ItemCF()
    inputPath = "../ml_latest_small/ratings.csv"
    #data = itemCF.loadData(inputPath)
    train_dataset,test_dataset = read_data_sets(inputPath,with_split=True)
    # data = {'A':{'a':1,'b':1,'d':1},'B':{'b':1,'c':1,'e':1},'C':{'d':1,'d':1},'D':{'b':1,'d':1,'d':1},'E':{'a':1,'d':1}}
    train = train_dataset.user_item_dict
    ItemCF_model=itemCF.train(train)
    ItemCF_model.recommendation(user_id='405')
