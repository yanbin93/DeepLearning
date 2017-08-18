#!/usr/bin/env python
#-*- coding:utf-8 -*-
############################
#File Name: topnBased.py
#Author: yanbin
#Mail: yanbin918@gmail.com
#Created Time: 2017-08-17 15:30:42
############################
import operator
import sys
sys.path.append('../')
from ml_latest_small.data import read_data_sets

class topnModel(object):
    def __init__(self,items_times=dict(),N=10):
        self.items_times = items_times
        self.N = N

    def fit(self,train):
        for user,items in train.items():
            for item,times in items.items():
                if item not in self.items_times:
                    self.items_times[item]=0
                self.items_times[item]+=1

    def recommendation(self,user_id,config=None):
        N = self.N
        if config is not None:
            N = config['N']
        rank = sorted(self.items_times.items(),
			key=operator.itemgetter(1),reverse=True)[0:N]
        return dict(rank)

if __name__ == '__main__':
    inputPath = "../ml_latest_small/ratings.csv"
    train_dataset,test_dataset = read_data_sets(inputPath,with_split=True)
    train = train_dataset.user_item_dict
    topn_model = topnModel()
    topn_model.fit(train)
    print topn_model.recommendation(user_id='405')

