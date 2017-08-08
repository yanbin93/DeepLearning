#/usr/bin/env python
#-*- coding:utf-8-*-
import requests
import json
import os
import urllib
import os.path
import pickle
import pandas as pd
class MovieDB(object):
    def __init__(self,url = 'http://api.themoviedb.org/3',
                    headers = {'Accept':'application/json'},
                    payload = {'api_key':'e0ec617d1ba0c2c3adb4f43df47f9ab4'}):
        self.url = url
        self.headers = headers
        self.payload = payload
        
    def getImageBasePath(self,size='w185'):
        response = requests.get(self.url+'/configuration',params=self.payload,headers=self.headers)
        response = json.loads(response.text)
        base_url = response['images']['base_url']+'w185'
        return base_url
    
    def get_poster(self,imdb):
        #query themovie.org API for movie poster path.
        base_url = self.getImageBasePath()
        file_path = ''
        imdb_id = 'tt0{0}'.format(imdb)
        movie_url = self.url+'/movie/{:}/images'.format(imdb_id)
        response = requests.get(movie_url,params=self.payload,headers=self.headers)
        try:
            file_path = json.loads(response.text)['posters'][0]['file_path']
        except:
            print('Failed to get url for imdb: {0}'.format(imdb))
        return base_url+file_path
    
    def getIDs(self,link_f="./ml-latest-small/links.csv",download_posters=3000):
        df_id = pd.read_csv(link_f,sep=',')
        idx_to_mv = {}
        for row in df_id.itertuples():
            idx_to_mv[row[1]-1] = row[2]
        mvs = [0]*len(idx_to_mv.keys())
        for i in range(len(mvs)):
            if i in idx_to_mv.keys() and len(str(idx_to_mv[i])) == 6:
                mvs[i] = idx_to_mv[i]
        mvs = list(filter(lambda imdb:imdb!=0,mvs))
#         shuffle_index = range(len(mvs))
#         np.random.shuffle(shuffle_index)
#         mvs = [mvs[i] for i in mvs]
        mvs = mvs[:download_posters]
        return mvs
    
    def down_poster(self,mvs,poster_pt='./poster/'):
        base_url = self.getImageBasePath()
        total_mvs = len(mvs)
        URL = [0]*total_mvs
        URL_IMDB = {'url':[],'imdb':[],'pic':[]}
        i=0
        for m in mvs:
            if(os.path.exists(poster_pt+str(i)+'.jpg')):
                print('Skip downloading exists jpg: {0}.jpg'.format(poster_pt+str(i)))
                i += 1
                continue
            URL[i] = self.get_poster(m)
            if(URL[i] == base_url):
                print('Bad imdb id: {0}'.format(m))
                mvs.remove(m)
                continue
            print('No.{0}: Downloading jpg(imdb {1}) {2}'.format(i,m,URL[i]))
            urllib.urlretrieve(URL[i],poster_pt+str(i)+'.jpg')
            URL_IMDB['url'].append(URL[i])
            URL_IMDB['imdb'].append(m)
            URL_IMDB['pic'].append(i)
            i += 1
        output = open('URL_IMDB.pkl', 'wb')
        # Pickle dictionary using protocol 0.
        pickle.dump(URL_IMDB, output)
        # Pickle the list using the highest protocol available.
        output.close()
        return URL_IMDB
if __name__ == '__main__':        
    movieDb = MovieDB()
    mvs = movieDb.getIDs()
    URL_IMDB = movieDb.down_poster(mvs)
#   url=movieDb.get_poster(mvs[0])