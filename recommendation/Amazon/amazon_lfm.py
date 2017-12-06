#!/usr/bin/env python
# -*- coding:utf-8 -*-
from pyspark import SparkContext,SparkConf
from operator import add,itemgetter
from pyspark.sql import SparkSession,Row
from pandas import *
from pyspark.sql.functions import *
from pyspark.ml.feature import Bucketizer
from pyspark.sql.types import *
import time
import math
class ItemCFModel(object):  
    def __init__(self,df,item_pair_sim=None,spark=None,topN=10,NN=100):
        self.spark = spark
        if (self.spark is None):
            conf = SparkConf().setAppName("itemCF").setMaster("yarn-cluster")
            self.spark = SparkSession.builder\
                .config(conf=conf) \
                .enableHiveSupport()\
                .getOrCreate()
        sc = spark.sparkContext 
        self.df = df
        self.item_pair_sim = item_pair_sim
        self.topN = topN
        self.NN = NN

    def itemSimilarity(self):
        # RDD[(uid,(aid,score))] 
        user_item_score = self.df.rdd.map(lambda x:(x[0],[x[1],x[2]]))
        item_score_pair = user_item_score.join(user_item_score)\
                        .map(lambda x:((x[1][0][0],x[1][1][0]),(x[1][0][1],x[1][1][1])))
        item_pair_ALL = item_score_pair.map(lambda f:(f[0], f[1][0] * f[1][1])).reduceByKey(add,300)
        item_pair_XX_YY = item_pair_ALL.filter(lambda f:f[0][0] == f[0][1])
        item_pair_XY = item_pair_ALL.filter(lambda f:f[0][0] != f[0][1])
        #RDD[(aid1,score11 * score11 + score21 * score21)] 
        item_XX_YY = item_pair_XX_YY.map(lambda f:(f[0][0], f[1]))
        #RDD(aid1,((aid1,aid2,XY),XX))
        item_XY_XX = item_pair_XY.map(lambda f:(f[0][0], (f[0][0], f[0][1], f[1]))).join(item_XX_YY) 
        #RDD[(aid2,((aid1,aid2,
        #           score11 * score12 + score21 * score22,score11 * score11 + score21 * score21),
        #           score12 * score12 + score22 * score22))] 
        item_XY_XX_YY = item_XY_XX.map(lambda f:(f[1][0][1],(f[1][0][0],f[1][0][1],f[1][0][2],f[1][1]))).join(item_XX_YY)  
        # item_XY_XX_YY中的(aid1,aid2,XY,XX,YY)) 
        # RDD[(aid1,aid2,
        # score11 * score12 + score21 * score22,score11 * score11 + score21 * score21,score12 * score12 + score22 * score22)]       
        item_pair_XY_XX_YY = item_XY_XX_YY.map(lambda f:(f[1][0][0], f[1][0][1], f[1][0][2], f[1][0][3], f[1][1]))  
        # item_pair_XY_XX_YY为(aid1,aid2,XY / math.sqrt(XX * YY)) 
        # RDD[(aid1,aid2,
        # score11 * score12 + score21 * score22 / math.sqrt((score11 * score11 + score21 * score21)*(score12 * score12 + score22 * score22))] 
        item_pair_sim = item_pair_XY_XX_YY.map(lambda f :(f[0], (f[1], f[2] / math.sqrt(f[3] * f[4]))))  
        return item_pair_sim
    

    def train(self):
        item_pair_sim = self.itemSimilarity()
        item_pair_sim.cache()  
        self.item_pair_sim=item_pair_sim
    
    def Normalization(self,item_pair_sim):
        def norm(x):
            m =  __builtin__.max([i[1] for i in x])
            for i in x:
                yield i/m
        return item_pair_sim.mapValues(lambda x:norm(x))
    


def recommend(df,item_pair_sim,NN=100,topN =10,Normalization=False):
    def itemNN(item_pair_sim,NN=100,Normalization=False):
        item_sim = item_pair_sim.filter(lambda f:f[1][1]>0.05)\
                            .groupByKey()\
                            .mapValues(list)
        if Normalization:
            item_sim = Normalization(item_sim)
        item_simNN = item_sim.mapValues(lambda x:sorted(x,key=itemgetter(1),reverse=True)[:NN])\
                            .collectAsMap()
        return item_simNN
    
    def getOrElse(f,item_sim_bd):
        items_sim = item_sim_bd.value.get(f[0][1]) 
        if items_sim is None:
            items_sim = [("0", 0.0)]
        for w in items_sim:
            yield ((f[0][0],w[0]),w[1]*f[1])
            
    user_item_score = df.rdd.map(lambda x:((x[0],x[1]),x[2]))
    item_sim_bd = sc.broadcast(itemNN(item_pair_sim))
#     /* 
#      * 提取item_sim_user_score为((user,item2),sim * score) 
#      * RDD[(user,item2),sim * score] 
#      */  

    user_item_simscore = user_item_score.flatMap(lambda f:getOrElse(f,item_sim_bd))\
                                        .filter(lambda f:f[1]> 0.03)  
#       /*
#      * 聚合user_item_simscore为 (user,（item2,sim1 * score1 + sim2 * score2）)
#      * 假设user观看过两个item,评分分别为score1和score2，item2是与user观看过的两个item相似的item,相似度分别为sim1，sim2 
#      * RDD[(user,item2),sim1 * score1 + sim2 * score2）)] 
#      */  
    user_item_rank = user_item_simscore.reduceByKey(add,1000)  

#     /* 
#      * 过滤用户已看过的item,并对user_item_rank基于user聚合 
#      * RDD[(user,CompactBuffer((item2,rank2）,(item3,rank3)...))] 
#      */  
    user_items_ranks = user_item_rank.subtractByKey(user_item_score)\
                                     .map(lambda f:(f[0][0], (f[0][1], f[1])))\
                                     .groupByKey()  
#     /* 
#      * 对user_items_ranks基于rank降序排序，并提取topN,其中包括用户已观看过的item 
#      * RDD[(user,ArrayBuffer((item2,rank2）,...,(itemN,rankN)))] 
#      */  
    user_items_ranks_desc = user_items_ranks.mapValues(list)\
                            .mapValues(lambda x:sorted(x,key=itemgetter(1),reverse=True)[:topN])
    return user_items_ranks_desc

from Evaluator import Evaluator
def evaluate(evaluator):
    precise = evaluator.precision()
    coverage = evaluator.coverage()
    popularity = evaluator.popularity()
    recall = evaluator.recall()
    return precise,recall,coverage,popularity

if __name__ == "__main__": 
    import os
    PYSPARK_PYTHON = "/usr/bin/python2.7"
    os.environ["PYSPARK_PYTHON"] = PYSPARK_PYTHON
    conf = SparkConf().setAppName("amazonItemCF").setMaster("yarn-client")
    conf.set("spark.shuffle.file.buffer","128k").set("spark.reducer.maxSizeInFlight","96M")
    conf.set('spark.yarn.dist.files',
            'file:/root/hadoop-2.6/spark/python/lib/pyspark.zip,file:/root/hadoop-2.6/spark/python/lib/py4j-0.10.4-src.zip')
    conf.setExecutorEnv('PYTHONPATH','pyspark.zip:py4j-0.10.4-src.zip')
    conf.set('spark.executor.cores','30')
    conf.set('spark.executor.memory','95g')
    conf.set('spark.executor.instances','4')
    spark = SparkSession.builder\
        .config(conf=conf) \
        .enableHiveSupport()\
        .getOrCreate()
    spark.sql('set spark.sql.broadcastTimeout=30000')
    sc=spark.sparkContext 
    sc.setLogLevel('WARN')

    start = time.time()
    inputPath = "data/amazon/complete_csv"
    schema = StructType([
        StructField("user", StringType(), True),
        StructField("item", StringType(), True),
        StructField("rating", DoubleType(), True),
        StructField("text", StringType(), True),
        StructField("summary", StringType(), True)])
    inputPath = "data/ml_latest_small/ratings.csv"
    schema = StructType([
            StructField("user", StringType(), True),
            StructField("item", StringType(), True),
            StructField("rating", DoubleType(), True),
            StructField("timestamp", LongType(), True)])
#    ratingDf = spark.read.csv(inputPath,header=True,schema=schema)
    inputPath = "data/amazon/rating_parquet"
    ratingDf = spark.read.parquet(inputPath)
    ratingDf = ratingDf.repartition(60)
#    ratingDf,_ = ratingDf.randomSplit([1.0,999.0])
    user_count = ratingDf.groupBy('user').agg(count('*').alias('user_count')).filter('user_count>4')
    item_count = ratingDf.groupBy('item').agg(count('*').alias('item_count')).filter('item_count>4')
    ratingDf = ratingDf.join(user_count,on='user',how='inner')
    ratingDf = ratingDf.join(item_count,on='item',how='inner')
    ratingDf.printSchema()
    ratingDf.show(5)
    n = ratingDf.count()
    print 'total lines: %s' %n
    train,test = ratingDf.randomSplit([4.0,1.0],seed=40)
    itemCF = ItemCFModel(df=train,spark=spark)
    itemCF.train()
    recTopN = recommend(train,itemCF.item_pair_sim)
    
    pre = spark.createDataFrame(data=recTopN.flatMapValues(lambda x:x).map(lambda x:(x[0],x[1][0],x[1][1])),
                                schema=['user','item','rating'])
    evaluator = Evaluator(train,test,pre)
    (precise,recall,coverage,popularity) = evaluate(evaluator)
    print ('precise:%2.2f%%,  recall:%2.2f%%,  coverage:%2.2f%%,  popularity:%2.2f'
            %(precise*100,recall*100,coverage*100,popularity))
    pre.write.csv(path='data/amazon/result/amazon_lfm_'+'_'.join(time.ctime().split()[:3]),mode='overwrite')
    end = time.time()
    print 'spend %s s' %(end-start)
    import os
    os.popen('python email.py')
