#!/usr/bin/env python
#-*- coding:utf-8 -*-
###########################
#File Name: data.py
#Author: yanbin
#Mail: yanbin918@gmail.com
#Created Time: 2017-08-09 18:35:28
############################
import numpy as np
import random
def stringIndexer(itemset):
    item_sorted = sorted(itemset,cmp=lambda x,y:cmp(int(x),int(y)))
    index = xrange(1,len(item_sorted)+1)
    Indexer = dict(zip(item_sorted,index))
    return Indexer

def indexToString(itemset):
    item_sorted = sorted(itemset,cmp=lambda x,y:cmp(int(x),int(y)))
    index = xrange(1,len(item_sorted)+1)
    Indexer = dict(zip(index,item_sorted))
    return Indexer


class DataSet(object):
    def __init__(self,
                 user_item,
                 fake_data=False,
                 one_hot=False,
                 reshape=True):
        self._num_examples = len(user_item)
        self._user_item = user_item
        self._epochs_completed = 0 
        self._index_in_epoch = 0
    
    @property
    def user_item(self):
        return self._user_item

    @property
    def num_examples(self):
        return self._num_examples
    
    @property
    def epochs_completed(self):
        return self._epochs_completed

    @property
    def user_item_dict(self):
        dataset = self.user_item
        datadict = dict()
        for user,item in dataset:
            if user not in datadict:
                datadict[user] = dict()
            if item not in datadict[user]:
                datadict[user][item] = 0
            datadict[user][item] += 1
        return  datadict

    @property
    def itemset(self):
        return set([x[1] for x in self.user_item])

    @property
    def itemIndexer(self):
        return stringIndexer(self.itemset)

    @property
    def indexItemer(self):
        return indexToString(self.itemset)
    
    @property
    def userset(self):
        return set([x[0] for x in self.user_item])

    @property
    def userIndexer(self):
        return stringIndexer(self.userset)
     
    @property
    def indexUser(self):
        return indexToString(self.userset)

    @property
    def user_item_matrix(self):
        item_nums = len(self.itemset)
        user_nums = len(self.userset)
        datamatrix = np.zeros((user_nums+1,item_nums+1))
        userIndexer = self.userIndexer 
        itemIndexer = self.itemIndexer
        datadict = self.user_item_dict
        for user,items in datadict.items():
            for item in  items:
                user_index = userIndexer[user]
                item_index = itemIndexer[item]
                datamatrix[user_index][item_index]=datadict[user][item]
        return datamatrix
    
    def imbd(self,local_file='./links.csv'):
        with open(local_file,'rb') as f:
            line = f.readline()
            line = f.readline()
            imBD = dict()
            while(line):
                movieId,imbdId = line.split(',')[:2]
                imBD[movieId] = imbdId
                line = f.readline()
        return imBD

    def next_batch(self, batch_size, fake_data=False, shuffle=True):
#"""Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
#             perm0 = np.arange(self._num_examples)
#             np.random.shuffle(perm0)
#             self._user_item = self._user_item[perm0]
            random.shuffle(self._user_item)
        # Go to the next epoch
        
        if start + batch_size > self._num_examples:
          # Finished epoch
            self._epochs_completed += 1
          # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            user_item_rest_part = self._user_item[start:self._num_examples]
          # Shuffle the data
            if shuffle:
#                 perm = np.arange(self._num_examples)
#                 np.random.shuffle(perm)
#                 self._user_item = self._user_item[perm]
                random.shuffle(self._user_item)
          # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            user_item_new_part = self._user_item[start:end]
            return np.concatenate(
              (user_item_rest_part, user_item_new_part),
              axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._user_item[start:end]


def read_data_sets(local_file,split_char=',',with_split = False,one_hot = False,index = False,reshape=True):
    dataset = []
    with open(local_file,'rb') as f:
        line = f.readline()
        line = f.readline()
        while(line):
            dataset.append(line.split(split_char)[:2])
            line = f.readline()
    if not with_split:
        return DataSet(dataset)
    train = []
    test = []
    random.seed(40)
    for u,m in dataset:
        if random.randint(0,5)==0:
            test.append([u,m])
        else:
            train.append([u,m])
    return DataSet(train),DataSet(test)

if  __name__ == '__main__':
    dataset = read_data_sets(
        "/home/yanbin/pythonProject/DeepLearning/recommendation/ml_latest_small/ratings.csv",
        with_split=True)[1]
    user_item =  dataset.user_item
    userset = dataset.userset
    itemset = dataset.itemset
    print dataset._num_examples
    print dataset.next_batch(10000)
## for i in xrange(1,10):
##     print i,datadict[str(i)]
#datamatrix,userIndexer,itemIndexer = read_data_sets(
#    "/home/yanbin/pythonProject/DeepLearning/recommendation/ml-latest-small/ratings.csv",
#    one_hot=True,index=True)
#for i in xrange(1,10):
#    print userIndexer[i]
#    print datamatrix[i,:]
