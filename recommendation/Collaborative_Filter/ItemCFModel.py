#!/usr/bin/env python
#-*- coding:utf-8 -*-
############################
#File Name: ItemCFModel.py
#Author: yanbin
#Mail: yanbin918@gmail.com
#Created Time: 2017-08-16 17:53:37
############################
class ItemCFModel(object):   
    def __init__(self,train,W,K=5,N=5):
        self.train = train
        self.W = W
        self.K = K
        self.N = N

    def Recommendation(self,user_id,config=None):
        W = self.W
        K = self.K
        N = self.N
        if config is not None:
            try:
                K = config['K']
                N = config['N']
            except:
                print "config = {'K':nearest nums,'N':topN nums}"
                break
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
        return topN

