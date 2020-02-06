from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import pandas as pd
import random as r

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

import tensorflow as tf
import cv2
import numpy as np
from scipy import spatial
# import matplotlib.pyplot as plt
# from tensorflow.keras.models import load_model
# Define a flask app

app = Flask(__name__)

# imgs = movie_info['img_link'].values[100:108]
# ts = movie_info['title'].values[100:108]
movie_info = pd.read_csv('data/finalscrape.csv')
ratings = pd.read_csv('data/ratings.csv')
movies = pd.read_csv('data/movies.csv')
genome_scores = pd.read_csv('data/genome-scores.csv')
genome_tags = pd.read_csv('data/genome-tags.csv')
links = pd.read_csv('data/links.csv')


movieProperties = ratings.groupby('movieId').agg({'rating': [np.size, np.mean]})
movieNumRatings = pd.DataFrame(movieProperties['rating']['size'])
movieNormalizedNumRatings = movieNumRatings.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
g = list(movies['genres'])
g = [x.split('|') for x in g]
flat_list = []
for sublist in g:
    for item in sublist:
        flat_list.append(item)
genre_list = sorted(set(flat_list))
movieDict = {}
for m in movies.values:
    try:
        movieID = int(m[0])
        name = m[1]
        genres = m[2].split("|")
        g = [1 if i in genres else 0 for i in genre_list]
        movieDict[movieID] = (name, np.array(list(g)), movieNormalizedNumRatings.loc[movieID].get('size'), movieProperties.loc[movieID].rating.get('mean'))
    except:
        pass

def ComputeDistance(a, b):
    genresA = a[1]
    genresB = b[1]
    genreDistance = spatial.distance.cosine(genresA, genresB)
    popularityA = a[2]
    popularityB = b[2]
    popularityDistance = abs(popularityA - popularityB)
    return genreDistance + popularityDistance

ComputeDistance(movieDict[1], movieDict[4])

import operator

def getNeighbors(movieID, K=10):
    distances = []
    for movie in movieDict:
        if movie != movieID:
            dist = ComputeDistance(movieDict[movieID], movieDict[movie])
            distances.append((movie, dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(K):
        neighbors.append(distances[x][0])
    return neighbors
    
K=10
avgRating = 0
    
def userGen(user):
    ratingThresh = np.percentile(ratings[ratings.userId == user]['rating'].values,90)
    moreRate = ratings[(ratings.userId == user) & (ratings.rating >= ratingThresh)]

    def f1(x):
        genomeThresh = np.percentile(genome_scores[genome_scores.movieId == x]['relevance'].values,99)
        moreRelevance = genome_scores[(genome_scores.movieId == x) & (genome_scores.relevance >= genomeThresh)]
        print("F1")
        return moreRelevance['tagId']

    def ltags(x):
        k = f1(x['movieId'][x.index[0]])
        if len(k) != 1 :
            i = 1
            while(len(k)>=10 and i < len(x.index)):
                k_prev = k
                k = pd.merge(k, f1(x['movieId'][x.index[i]]), how='inner', on=['tagId'])
                if len(k) == 0:
                    k = k_prev
                i = i+1
        print("LTAGS")
        return pd.DataFrame(k,columns=['tagId'])

    tagList = ltags(moreRate)

    def f2(x):
        genomeThresh = np.percentile(genome_scores[genome_scores.tagId == x]['relevance'].values,99.9)
        # print(genomeThresh)
        # print(genome_scores[genome_scores.tagId == x])
        moreRelevance = genome_scores[(genome_scores.tagId == x) & (genome_scores.relevance >= genomeThresh)]
        print("F2")
        return moreRelevance['movieId']

    def lmov(x):
        k = f2(x['tagId'][x.index[0]])
        if len(k) != 1 :
            i = 1
            while(len(k)>=10 and i < len(x.index)):
                k_prev = k
                k = pd.merge(k, f2(x['tagId'][x.index[i]]), how='inner', on=['movieId'])
                if len(k) == 0:
                    k = k_prev
                i = i+1
        print("LMOV")
        return pd.DataFrame(k)

    movieList = lmov(tagList)
    
    def genreSelect(x):
        k = []
        for i in x.values:
            k.append(movies[movies.movieId == i[0]]['genres'].values[0].split('|'))
        flat_list = []
        for sublist in k:
            for item in sublist:
                flat_list.append(item)
        genList = []
        m1 = max(set(flat_list), key=flat_list.count)
        genList.append(m1)
        m2 = max(set(list(filter(lambda a: a != m1, flat_list))), key=flat_list.count)
        genList.append(m2)
        print("GEN_LIST")
        return genList
        
    FinalAns = []
    FinalAns.append(genreSelect(movieList))
    FinalAns.append(movieList['movieId'].values.tolist())
    return FinalAns
    
imgs = []
ts = []

def topList():
    global ratings
    summ = ratings.groupby('movieId').sum()
    cleaned = summ.drop('userId',axis=1)
    c2 = cleaned.drop('timestamp',axis=1)
    c3 = c2.sort_values(by=['rating'],ascending=False)
    topl = c3.iloc[:8,:]
    topList = list(topl.index)
    return topList
    
def gselect(x):
    global ratings,movies
    selectgenre=ratings.groupby('movieId').agg({'rating':[np.size,np.mean]})
    selectgenre.columns = [''] * len(selectgenre.columns)
    selectgenre.columns = ['count', 'mean']

    usergenre = x 
    genresdf = movies[movies['genres'].str.match(usergenre)]
    dfinal = genresdf.merge(selectgenre, on="movieId", how = 'inner')
    dfinal = dfinal.sort_values(by =['mean'], ascending=False)
    dfinal1 = dfinal[dfinal['count'] > 25]
    dfinal2 = dfinal1[['title','mean']].head(4)
    fin = list(dfinal2['title'])
    return fin
    

print('Check http://127.0.0.1:5000/')



@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('login.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Get the file from post request
        u = request.form['username']
        p = request.form['password']
        
        print(u,p)
        
        Fin = userGen(int(u))
        print(Fin)
        Mlist = Fin[1][:8]
        GenList = Fin[0]
        imgs = []
        ts = [] 
        imgstop = []
        tstop = []
        for i in Mlist:
            # k = links[links.movieId == i]
            # movie_info[movie_info.tmdb == links[links.movieId == i.iloc[0,2]]
            # ts.append(movie_info[movie_info.tmdbId == k['tmdbId']]['title'])
            # imgs.append(movie_info[movie_info.tmdbId == k['tmdbId']]['img_link']) 
            ts.append(movie_info[movie_info.tmdbId == links[links.movieId == i].iloc[0,2]].iloc[0,4])
            imgs.append(movie_info[movie_info.tmdbId == links[links.movieId == i].iloc[0,2]].iloc[0,8])
            # print(ts)
            # print(imgs) 
        topL = topList()
        for i in topL:
            tstop.append(movie_info[movie_info.tmdbId == links[links.movieId == i].iloc[0,2]].iloc[0,4])
            imgstop.append(movie_info[movie_info.tmdbId == links[links.movieId == i].iloc[0,2]].iloc[0,8])
            
        g1img = []
        g2img = []
        
        g1tit = gselect(GenList[0])
        g2tit = gselect(GenList[1])
        
        for i in g1tit:
            try:
                ii = "".join(list(i)[:-7])
                print(ii)
                g1img.append(movie_info[movie_info.title == ii]['img_link'].values[0])
            except:
                g1img.append('https://image.tmdb.org/t/p/w300_and_h450_bestv2/rhIRbceoE9lR4veEXuwCC2wARtG.jpg')
                
                
        for i in g2tit:
            try:
                ii = "".join(list(i)[:-7])
                print(ii)
                g2img.append(movie_info[movie_info.title == ii]['img_link'].values[0])
            except:
                g2img.append('https://image.tmdb.org/t/p/w300_and_h450_bestv2/rhIRbceoE9lR4veEXuwCC2wARtG.jpg')
        
            
        global ratings
        useList = set(ratings['userId'])
        print(useList)
            
        
        if p == 'a':
            return render_template('landing.html',username=u,img=imgs,title=ts,gen=GenList,topimg=imgstop,toptit=tstop,gen1=g1tit,gen2=g2tit,gimg1=g1img,gimg2=g2img)
        else:
            return render_template('login.html')        
    return None
    
@app.route('/movie', methods=['GET'])
def movie():
    mov = request.args.get('name')
    print(mov)
    global movie_info
    global movies
    # year = movie_info[movie_info.title == mov]['year'].values
    year = movie_info.loc[movie_info['title'] == mov]['year'].values[0]
    im = movie_info.loc[movie_info['title'] == mov]['img_link'].values[0]
    ov = movie_info.loc[movie_info['title'] == mov]['overview'].values[0]
    r = movie_info.loc[movie_info['title'] == mov]['rating'].values[0]
    tid = movie_info[movie_info.title == mov].iloc[0,3]
    print(tid)
    movid = links[links.tmdbId == tid].iloc[0,0]
    print(movid)
    # print(movies.head())
    # tsm.append(movie_info[movie_info.tmdbId == links[links.movieId == i].iloc[0,2]].iloc[0,4])
    # imgsm.append(movie_info[movie_info.tmdbId == links[links.movieId == i].iloc[0,2]].iloc[0,8])
    imgsm = []
    tsm = [] 
    midsm = getNeighbors(movid, K)
    for i in midsm:
        tsm.append(movie_info[movie_info.tmdbId == links[links.movieId == i].iloc[0,2]].iloc[0,4])
        imgsm.append(movie_info[movie_info.tmdbId == links[links.movieId == i].iloc[0,2]].iloc[0,8])
    
    
    # print(year1)
    return render_template('movie.html',name=mov,yr=year,img=im,desc=ov,rate=r,img1=imgsm,title=tsm)


if __name__ == '__main__':
    # app.run(port=5002, debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
