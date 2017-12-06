#!/usr/bin/env python
#-*- coding:utf-8 -*-
############################
#File Name: Evaluator.py
#Author: yanbin
#Mail: yanbin918@gmail.com
#Created Time: 2017-08-16 22:52:14
############################
__author__ = "yan bin"
from pyspark.sql.functions import *

class Evaluator(object):
    def __init__(self,train,test,pre):
        self.train = train
        self.test = test
        self.pre = pre
    
    def recall(self):
        train = self.train 
        test = self.test
        pre = self.pre
        cond = [pre.user == test.user,pre.item == test.item]
        m = test.count()
        n = pre.join(test, cond, 'inner').count()
        return n*1.0/(m*1.0)

    def precision (self):
        train = self.train 
        test = self.test
        pre = self.pre
        cond = [pre.user == test.user,pre.item == test.item]
        m = pre.count()
        n = pre.join(test, cond, 'inner').count()
        return n*1.0/(m*1.0)

    def coverage(self):
        train = self.train 
        test = self.test
        pre = self.pre
        m = train.select('item').distinct().count()
        n = pre.select('item').distinct().count()
        return n*1.0/(m*1.0)

    def popularity(self):
        train = self.train 
        test = self.test
        pre = self.pre
        item_popularity = train.groupBy('item').agg(count('*').alias('pop'))\
                                .withColumn('logPop',log(col('pop')+1))\
                                .select('item','logPop')
        cond = [pre.item == item_popularity.item]
        ret = pre.join(item_popularity,cond,'inner').select(sum('logPop')).rdd.map(lambda x:x[0]).collect()[0]
        n = pre.count()
        ret /= n*1.0
        return ret