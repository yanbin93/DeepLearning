#!/usr/bin/env python
#-*- coding:utf-8 -*-
############################
#File Name: recommendator.py
#Author: yanbin
#Mail: yanbin918@gmail.com
#Created Time: 2017-08-17 10:
from ml_latest_small.data import *
from Collaborative_Filter.itemBased import ItemCFModel,ItemCF
from Collaborative_Filter.userBased import UserCFModel,UserCF
from Collaborative_Filter.lfmBased import lfmModel,lfm
from Comparision.randomBased import RandomModel
from Comparision.topnBased import topnModel
import pickle
import os
from Evaluation.Evaluator import Evaluator

def load_result(path):
    pkl_file = open(path, 'rb')
    prediction_dict = pickle.load(pkl_file)
    # pprint.pprint(data)
    pkl_file.close()
    return prediction_dict

def evaluate(evaluator):
    precise = evaluator.precision()
    coverage = evaluator.coverage()
    popularity = evaluator.popularity()
    recall = evaluator.recall()
    return precise,recall,coverage,popularity

def itemCF_predict(train_dataset,test_dataset,K):
    train_dict = train_dataset.user_item_dict
    test_dict = test_dataset.user_item_dict
    itemCF = ItemCF()
    #triain ItemCF_model
    print "训练基于物品协同过滤算法模型........"
    ItemCF_model=itemCF.train(train_dict)
    #recommendation for every user in train_Dataset
    users = train_dict.keys()
    user_nums = len(users)
    i = 0
    prediction_dict ={}
    for user in users:
        prediction_dict[user]=ItemCF_model.recommendation(user,config={'K':K,'N':10}).keys()
    #将每位用户预测结果存入本地
    output = open(result_pt+itemCF_prediction_result_pt %K, 'wb')
    # Pickle dictionary using protocol 0.
    pickle.dump(prediction_dict, output)
    print "物品协同过滤推荐算法预测结果存入本地!"
    # Pickle the list using the highest protocol available.
    output.close()
    return prediction_dict

def userCF_predict(train_dataset,test_dataset,K):
    train_dict = train_dataset.user_item_dict
    test_dict = test_dataset.user_item_dict    
    userCF = UserCF()
    #triain ItemCF_model
    print "用户基于物品协同过滤算法模型........"
    UserCF_model=userCF.train(train_dict)
    #recommendation for every user in train_Dataset
    users = train_dict.keys()
    user_nums = len(users)
    i = 0
    prediction_dict ={}
    for user in users:
        prediction_dict[user]=UserCF_model.recommendation(user,config={'K':K,'N':10}).keys()
    #将每位用户预测结果存入本地
    output = open(result_pt+userCF_prediction_result_pt %K, 'wb')
    # Pickle dictionary using protocol 0.bb
    pickle.dump(prediction_dict, output)
    print "用户协同过滤推荐算法预测结果存入本地!"
    # Pickle the list using the highest protocol available.
    output.close()
    return prediction_dict

def random_predict(train_dataset,test_dataset,K):
    train_dict = train_dataset.user_item_dict
    test_dict = test_dataset.user_item_dict
    randomRM = RandomModel(train_dict)
    #triain ItemCF_model
    print "基于随机推荐算法模型........"
    #recommendation for every user in train_Dataset
    users = train_dict.keys()
    user_nums = len(users)
    i = 0
    prediction_dict ={}
    for user in users:
        prediction_dict[user]=randomRM.recommendation(user,config={'K':K,'N':10}).keys()
    #将每位用户预测结果存入本地
    output = open(result_pt+random_prediction_result_pt %K, 'wb')
    # Pickle dictionary using protocol 0.
    pickle.dump(prediction_dict, output)
    print "基于随机推荐算法预测结果存入本地!"
    # Pickle the list using the highest protocol available.
    output.close()
    return prediction_dict
    
def topn_predict(train_dataset,test_dataset):
    train_dict = train_dataset.user_item_dict
    test_dict = test_dataset.user_item_dict
    topn_model = topnModel()
    topn_model.fit(train_dict)
    #triain ItemCF_model
    print "热门商品推荐算法模型........"
    #recommendation for every user in train_Dataset
    users = train_dict.keys()
    user_nums = len(users)
    i = 0
    prediction_dict ={}
    for user in users:
        prediction_dict[user]=topn_model.recommendation(user,config={'N':10}).keys()
    #将每位用户预测结果存入本地
    output = open(result_pt+topn_prediction_result_pt, 'wb')
    # Pickle dictionary using protocol 0.
    pickle.dump(prediction_dict, output)
    print "热门商品推荐算法预测结果存入本地!"
    # Pickle the list using the highest protocol available.
    output.close()
    return prediction_dict

def lfm_predict(train_dataset,test_dataset,F=10):
    train_dict = train_dataset.user_item_dict
    test_dict = test_dataset.user_item_dict
    LFM = lfm(F=F,lmbd=0.1,alpha=0.1,max_iter = 300)
    lfm_model= LFM.fit(train_dict)
    #triain lfm_model
    print "LFM推荐算法模型........"
    #recommendation for every user in train_Dataset
    users = train_dict.keys()
    user_nums = len(users)
    i = 0
    prediction_dict ={}
    for user in users:
        prediction_dict[user]=lfm_model.recommendation(user,config={'N':10}).keys()
    #将每位用户预测结果存入本地
    output = open(result_pt+lfm_prediction_result_pt %F, 'wb')
    # Pickle dictionary using protocol 0.
    pickle.dump(prediction_dict, output)
    print "LFM推荐算法预测结果存入本地!"
    # Pickle the list using the highest protocol available.
    output.close()
    return prediction_dict

inputPath = "./ml_latest_small/ratings.csv"
result_pt = "./Predict_Result/"
userCF_prediction_result_pt ='UserCF_Result/%s_usercf_prediction_dict.pkl'
itemCF_prediction_result_pt ='ItemCF_Result/%s_itemcf_prediction_dict.pkl'
random_prediction_result_pt = 'Random_Result/%s_random_prediction_dict.pkl'
topn_prediction_result_pt = 'Topn_Result/topn_prediction_dict.pkl'
lfm_prediction_result_pt = 'LFM_Result/%s_lfm_prediction_dict.pkl'
recommendation_enginer = {0:random_prediction_result_pt,
                          1:topn_prediction_result_pt,
                          2:userCF_prediction_result_pt,
                          3:itemCF_prediction_result_pt,
                          4:lfm_prediction_result_pt}
recommendator = {0:'randomRM',1:'topnRM',2:'userCF',3:'itemCF',4:'lfm'}
train_dataset,test_dataset = read_data_sets(inputPath,with_split=True)
Ks = [5,10,20,40,80,160]
i = input("please choose recommendation engineer:\n0->randomRM, 1->topnRM, 2->userBased, 3->itemBased, 4->lfm\n")
assert i  in [0,1,2,3,4] ,"recomendation enginer not exists"
print "The recommendator is "+recommendator[i]
if i in [0,2,3]:
    for K in Ks:
        path = (result_pt+recommendation_enginer[i] %K)
        flag = os.path.exists(path)
        if flag:prediction_dict = load_result(path)
        elif i is 0:
                prediction_dict = random_predict(train_dataset,test_dataset,K)
        elif i is 2:
            prediction_dict = userCF_predict(train_dataset,test_dataset,K)
        elif i is 3:
            prediction_dict = itemCF_predict(train_dataset,test_dataset,K)
        evaluator = Evaluator(train_dataset.user_item_dict,test_dataset.user_item_dict,prediction_dict,N=10)
        (precise,recall,coverage,popularity) = evaluate(evaluator)
        print ('K:%3d,  precise:%2.2f%%,  recall:%2.2f%%,  coverage:%2.2f%%,  popularity:%2.2f'
               %(K,precise*100,recall*100,coverage*100,popularity))

elif i is 1:
    path = (result_pt+recommendation_enginer[i])
    flag = os.path.exists(path)
    prediction_dict = load_result(path) if flag else\
    topn_predict(train_dataset,test_dataset)
    evaluator = Evaluator(train_dataset.user_item_dict,test_dataset.user_item_dict,prediction_dict,N=10)
    (precise,recall,coverage,popularity) = evaluate(evaluator)
    print ('precise:%2.2f%%,  recall:%2.2f%%,  coverage:%2.2f%%,  popularity:%2.2f'
           %(precise*100,recall*100,coverage*100,popularity))
elif i is 4:
    Fs = [10]
    for F in Fs:
        path = (result_pt+recommendation_enginer[i] % F )
        flag = os.path.exists(path)
        prediction_dict = load_result(path) if flag else\
        lfm_predict(train_dataset,test_dataset,F=F)
        evaluator = Evaluator(train_dataset.user_item_dict,test_dataset.user_item_dict,prediction_dict,N=10)
        (precise,recall,coverage,popularity) = evaluate(evaluator)
        print ('F:%3d,  precise:%2.2f%%,  recall:%2.2f%%,  coverage:%2.2f%%,  popularity:%2.2f'
               %(F,precise*100,recall*100,coverage*100,popularity))