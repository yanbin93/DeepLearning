#!/usr/bin/env python
#-*- coding:utf-8 -*-
############################
#File Name: userBased.py
#Author: yanbin
#Mail: yanbin918@gmail.com
#Created Time: 2017-08-17 10:12:02
############################
import math,operator
import sys
sys.path.append('../')
from ml_latest_small.data import *


class UserCFModel(object):
    def __init__(self,train,W,K=5,N=10):
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
        rank = dict()
        interacted_items = train[user_id]
        for v,wuv in sorted(W[user_id].items(),key=operator.itemgetter(1),
                            reverse=True)[0:K]:
            for i , rvi in train[v].items():
                if i in interacted_items:
                    continue
                if i not in rank:
                    rank[i] = 0
    #filter items user interacted
                rank[i] += wuv*rvi
        topN = sorted(rank.items(),key=operator.itemgetter(1),reverse=True)[0:N]
        return dict(topN)



class UserCF(object):
    def userSimilarity(self,train):
        W=dict()
        for u in train.keys():
            W[u]=dict()
            for v in train.keys():
                if u==v:
                    continue
                W[u][v]=len(set(train[u].keys())
		&set(train[v].keys()))/math.sqrt(len(train[u])*len(train[v])*1.0)
        return W

    def userSimilarity2(self,train):
        #build inverse table for item_users
        item_users = dict()
        for u,items in train.items():
            for i in items.keys():
                if i not in item_users:
                    item_users[i] = set()
                item_users[i].add(u)
        #calculate co-rated items between users
        C = dict()
        N = dict()
        for i, users in item_users.items():
            for u in users:
                if u not in C:
                    C[u]=dict()
                if u not in N:
                    N[u]=0
                N[u] +=1
                for v in users:
                    if u==v:
                        continue
                    if v not in C[u]:
                        C[u][v]=0
                    C[u][v]+=1
        #calculate finial similarity matrix W
        W = dict()
        for u,related_users in C.items():
            W[u]=dict()
            for v,cuv in related_users.items():
                W[u][v] = cuv/math.sqrt(N[u] * N[v])
        return W


    def userSimilarity3(self,train):
        #build inverse table for item_users
        item_users = dict()
        for u,items in train.items():
            for i in items.keys():
                if i not in item_users:
                    item_users[i] = set()
                item_users[i].add(u)

        #calculate co-rated items between users
        C = dict()
        N = dict()
        for i, users in item_users.items():
            for u in users:
                if u not in C:
                    C[u]=dict()
                if u not in N:
                    N[u]=0
                N[u] +=1
                for v in users:
                    if u==v:
                        continue
                    if v not in C[u]:
                        C[u][v]=0
                    C[u][v]+=1/math.log(1+len(users))
        #calculate finial similarity matrix W
        W = dict()
        for u,related_users in C.items():
            W[u]=dict()
            for v,cuv in related_users.items():
                W[u][v] = cuv/math.sqrt(N[u] * N[v])
        return W

    def train(self,train,Normalization=False):
        W = self.userSimilarity(train)
        if Normalization :
            W = self.Normalization(W)
        return UserCFModel(train,W)

    def Normalization(self,W):
        for i,items in W.items():
            m = max(items.values())
            for j,wij in items.items():
                W[i][j]=wij/m
        return W

if __name__ == '__main__':
    userCF = UserCF()
    inputPath = "../ml_latest_small/ratings.csv"
    train_dataset,test_dataset = read_data_sets(inputPath,with_split=True)
    train = train_dataset.user_item_dict
    UserCF_model=userCF.train(train)
    print UserCF_model.recommendation(user_id='405')
