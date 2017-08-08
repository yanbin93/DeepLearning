#!/usr/bin/env python
# -*- coding:utf-8 -*-
from keras.preprocessing import image as kimage
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
import os
import numpy as np
from re import match
import pickle
class Vgg16(object):
    def __init__(self,imageDir='./poster/',prediction_result="prediction_result.csv"):
        self.imageDir = imageDir
        self.prediction_result = prediction_result
        
    def save_predict_res(self,matrix_res):
        prediction_result=self.prediction_result
        pe = dict([(x,[]) for x in range(matrix_res.shape[0])])
        for i in range(matrix_res.shape[0]):
            pe[i].extend(matrix_res[i])
        df = pd.DataFrame(data=pe)
        df.to_csv(prediction_result)

    def load_result_matrix(self):
        file = self.prediction_result
        if not os.path.exists(file):
            return None
        m_r = pd.read_csv(file,sep=',')
        f_r = np.zeros((m_r.shape[1]-1,m_r.shape[0]))
        for i in range(m_r.values.shape[1]-1):
            f_r[i] = m_r[str(i)].values.tolist()
        return f_r 

    def train(self):
        poster_pt = self.imageDir
        total_mvs = len([x for x in os.listdir(poster_pt) if match("\d*.jpg",x)])
        total_mvs = 1000
        image = [0]*total_mvs
        x     = [0]*total_mvs
        for i in range(total_mvs):
            image[i] = kimage.load_img(poster_pt+str(i)+'.jpg',target_size=(224,224))
            x[i] = kimage.img_to_array(image[i])
            x[i] = np.expand_dims(x[i],axis=0)
            x[i] = preprocess_input(x[i])
        model = VGG16(include_top=False,weights='imagenet')
        prediction = [0]*total_mvs
        matrix_res = np.zeros([total_mvs,25088])
        for i in range(total_mvs):
            prediction[i]=model.predict(x[i]).ravel()
            matrix_res[i,:]=prediction[i]
    #     print model.predict(x[0])
        return matrix_res
    
    def simaility(self,matrix_res):
        similarity_deep = matrix_res.dot(matrix_res.T)
        norms = np.array([np.sqrt(np.diagonal(similarity_deep))])
#         print (norms.dot(norms.T)).shape
        similarity_deep = (similarity_deep/(norms*norms.T))
        return similarity_deep
    
    def topN(self,target_item,similarity_deep,N=5):
        similarity_deep = self.simaility(matrix_res)
        topN_list = [x for x in np.argsort(similarity_deep[target_item])[:-N-1:-1]]
        return topN_list
        
    def predictAll(self,similarity_deep):
        poster_pt = self.imageDir
        total_mvs = len([x for x in os.listdir(poster_pt) if match("\d*.jpg",x)])
        total_mvs = 1000
        prediction_list = dict([(x,[]) for x in range(total_mvs)])
        for i in prediction_list.keys():
            prediction_list[i] = self.topN(i,similarity_deep)
        return prediction_list
if __name__ == '__main__':
    vgg16 = Vgg16()
    matrix_res = vgg16.load_result_matrix()
    if matrix_res is None:
        print "trian model....."
        matrix_res = vgg16.train()
        vgg16.save_predict_res(matrix_res)
        print "Saved prediction_res in %s" %vgg16.prediction_result
    similarity_deep = vgg16.simaility(matrix_res)
    #output = open('similarity.pkl', 'wb')
    # Pickle dictionary using protocol 0.
    #pickle.dump(URL_IMDB, output)
    # Pickle the list using the highest protocol available.
    #output.close()
    np.save('simaility',similarity_deep)
    predict_result = vgg16.predictAll(similarity_deep)
    
    
    
    
   