#!/usr/bin/env python
#-*- coding:utf-8 -*-
############################
#File Name: randomBased.py
#Author: yanbin
#Mail: yanbin918@gmail.com
#Created Time: 2017-08-17 11:46:46
############################
import random
import sys
sys.path.append('../')
from ml_latest_small.data import read_data_sets
class RandomModel(object):
    def __init__(self,train,N=10):
        self.train = train
        self.N = N

    def recommendation(self,user_id,config=None):
        N = self.N
        if config is not None:
            try:
                N = config['N']
            except:
                print "config = {N':topN nums}"
                return
        train =list(set(reduce(lambda x,y:x+y,[x.keys() for x in self.train.values()])))
        result=[]
        for i in range(N):
            rd = random.randint(0,len(train)-1)
            result.append((train[rd],0))
        return dict(result)

if __name__ == '__main__':
    inputPath = "../ml_latest_small/ratings.csv"
    train_dataset,test_dataset = read_data_sets(inputPath,with_split=True)
    train = train_dataset.user_item_dict
    randomModel = RandomModel(train)
    print randomModel.recommendation(user_id='405')

