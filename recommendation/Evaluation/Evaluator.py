#!/usr/bin/env python
#-*- coding:utf-8 -*-
############################
#File Name: Evaluator.py
#Author: yanbin
#Mail: yanbin918@gmail.com
#Created Time: 2017-08-16 22:52:14
############################
import math
class Evaluator(object):
    def __init__(self,train_dict,test_dict,prediction_dict,N):
        self.train_dict = train_dict
        self.test_dict = test_dict
        self.prediction_dict = prediction_dict
        self.N = N

    def getRecommendation(self,userId):
        N = self.N
        recomUser = self.prediction_dict[userId]
        return recomUser

    def recall(self):
        train = self.train_dict
        test = self.test_dict
        hit = 0
        all = 0
        for user in train.keys():
            tu = test[user] if test.get(user) else {}
            rank = self.getRecommendation(user)
            for item in rank:
                if item in tu:
                    hit +=1
            all +=len(tu)
        return hit/(all * 1.0)

    def precision (self):
        train = self.train_dict
        test = self.test_dict
        N = self.N
        hit = 0
        all = 0
        for user in train.keys():
            tu = test[user] if test.get(user) else {}
            rank = self.getRecommendation(user)
            for item in rank:
                if tu and item in tu:
                    hit +=1
            all += N
        return hit*1.0/(all*1.0)

    def coverage(self):
        train = self.train_dict
        recommend_items =set()
        all_items = set()
        for user in train.keys():
            for item in train[user].keys():
                all_items.add(item)
            rank = self.getRecommendation(user)
            for item in rank:
                recommend_items.add(item)
        return len(recommend_items)/(len(all_items)*1.0)

    def popularity(self):
        train = self.train_dict
        item_popularity = dict()
        for user, items in train.items():
            for item in items:
                if item not in item_popularity:
                    item_popularity[item] = 0
                item_popularity[item] +=1

        ret = 0
        n = 0
        for user,items in train.items():
            rank = self.getRecommendation(user)
            for item in rank:
                ret += math.log(1+item_popularity[item])
                n+=1
        ret /= n*1.0
        return ret
