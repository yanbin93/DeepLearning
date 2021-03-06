#!/usr/bin/env python
#-*- coding:utf-8 -*-
############################
#File Name: lfmBased.py
#Author: yanbin
#Mail: yanbin918@gmail.com
#Created Time: 2017-08-17 16:04:18
############################
import sys
sys.path.append('../')
from ml_latest_small.data import *
import numpy as np
import operator
import math 
import random
import math
 
class lfm(object):
 
    def __init__(self, F, alpha=0.1,lmbd=0.1,ratio=1, max_iter=500):
        '''rating_data是list<(user,list<(position,rate)>)>类型
        '''
        self.F = F
        self.P = dict()  # R=PQ^T
        self.Q = dict()
        self.alpha = alpha
        self.lmbd = lmbd
        self.max_iter = max_iter
        self.item_pool = []
        self.ratio = 1
        
    def init_model(self,train_dict):
        '''随机初始化矩阵P和Q'''
        self.init_item_pool(train_dict)
        for user, rates in train_dict.items():
            self.P[user] = [random.random() / math.sqrt(self.F)
                            for x in xrange(self.F)]
            for item, _ in rates.items():
                if item not in self.Q:
                    self.Q[item] = [random.random() / math.sqrt(self.F)
                                    for x in xrange(self.F)]
 
    def randomSelectNegativeSamples(self,items):
        ret = dict()
        for i in items.keys():
            ret[i] = 1
        n =0
        item_pool = self.item_pool[:self.ratio*len(items)*2]
        for i in range(0,self.ratio*len(items)*3):
            item =item_pool[random.randint(0,len(item_pool)-1)]
            if item in ret:
                continue
            ret[item]=0
            n += 1
            if n > self.ratio*len(items):
                break
        return ret

    def init_item_pool(self,train_dict):
        item_list = reduce(lambda x,y:x+y,[x.keys() for x in train_dict.values()])
        item_count = map(lambda x:(x,1),item_list)
        item_pool = {}
        for item,count in item_count:
            if item not in item_pool:
                item_pool[item]=0
            item_pool[item] +=count
        self.user_nums = len(train_dict)
        self.item_nums = len(item_pool)
        item_pool = sorted(item_pool.items(),key=operator.itemgetter(1),reverse=True)
        self.item_pool = [x[0] for x in item_pool]
        
    def fit(self,train_dict):
        '''随机梯度下降法训练参数P和Q
        '''
        self.init_model(train_dict)
        for step in xrange(self.max_iter):
            for user, rates in train_dict.items():
                rates = self.randomSelectNegativeSamples(rates)
                for item, rui in rates.items():
                    hat_rui = self.predict(user, item)
                    err_ui = rui - hat_rui
                    for f in xrange(self.F):
                        self.P[user][f] += self.alpha * (err_ui * self.Q[item][f] - self.lmbd * self.P[user][f])
                        self.Q[item][f] += self.alpha * (err_ui * self.P[user][f] - self.lmbd * self.Q[item][f])
            self.alpha *= 0.9  # 每次迭代步长要逐步缩小
        return lfmModel(self.P,self.Q,self.F,train_dict)
    
    def predict(self, user, item):
        '''预测用户user对物品item的评分
        '''
        return sum(self.P[user][f] * self.Q[item][f] for f in xrange(self.F))

    
    
class lfmModel(object):
    def __init__(self,P,Q,F,train_dict=None,N=10):
        self.P = P
        self.Q = Q
        self.F = F
        self.N = N
        self.train_dict = train_dict

    def recommendation(self,user,config = None):
        N = self.N
        if config is not None:
            N = config['N']
        rank = dict()
        for item in self.Q.keys():
            if item in self.train_dict[user]:
                continue
            rank[item] = sum(self.P[user][f]*self.Q[item][f] for f in xrange(self.F))
        topN = sorted(rank.items(),key=operator.itemgetter(1),reverse=True)[0:N]
        return dict(topN)
    

if __name__ == '__main__':
    lfm = lfm(F=5)
    inputPath = "../ml_latest_small/ratings.csv"
    train_dataset,test_dataset = read_data_sets(inputPath,with_split=True)
    lfm_model= lfm.fit(train_dataset)
    print lfm_model.recommendation(user='405')
