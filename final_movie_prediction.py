# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 21:27:01 2019

@author: uttam3in
"""
from scipy import spatial
from sklearn.feature_extraction.text import CountVectorizer
import math
import pandas as pd
import numpy as np
ratings = pd.read_csv("D:/roops_backup/projects/data mining project/movie_recommender/movie_dataset.csv")
ratings[['id','title','vote_average','vote_count']]
new = pd.DataFrame(ratings[['id','title','vote_average']])
#new.shape


popular = pd.DataFrame(ratings['popularity'])
vcount = pd.DataFrame(ratings['vote_count'])
#new.loc[0].get('vote_average')
#genres = ratings['genres']
normalized_popularity = popular.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
normalized_votecount =  vcount.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
#normalized_popularity
#normalized_votecount

#popular.min()
normalized_votecount.loc[1].get('vote_count')
genres = ratings['genres'].fillna('')

cv = CountVectorizer()
count = cv.fit_transform(genres)
#print (count.toarray())
a = count.toarray()
moviedict = {}
for i in range(4810):
    moviedict[i] = (new.loc[i].get('id'),new.loc[i].get('title'),new.loc[i].get('vote_average'),normalized_votecount.loc[i].get('vote_count'),normalized_popularity.loc[i].get('popularity'),a[i])

def dis(x,y):
    
    y1 = x[4]
    z1 = x[5]
    
    y2 = y[4]
    z2 = y[5]
    genreDistance = spatial.distance.cosine(z1, z2)
    d = (y2-y1)*(y2-y1) + genreDistance*genreDistance 
    srt = math.sqrt(d)
    #print(genreDistance)
    return srt
def ComputeDistance(a, b):
    #genresA = a[5]
    #genresB = b[5]
    #genreDistance = spatial.distance.cosine(genresA, genresB)
    #popularityA = a[4]
    #popularityB = b[4]
    srt = dis(a,b)
    #popularityDistance = abs(popularityA - popularityB)
    #print (srt)
   # print (genreDistance)
    return srt 
    
#ComputeDistance(moviedict[0], moviedict[437])

import operator

def getNeighbors(movieID, K):
    distances = []
    for movie in moviedict:
        if (movie != movieID):
            dist = ComputeDistance(moviedict[movieID], moviedict[movie])
            if(dist <= 0.65):
             distances.append((movie, dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(K):
        neighbors.append((moviedict[distances[x][0]][1],distances[x][1]))
        print(moviedict[distances[x][0]][1])
    return neighbors
avgRating = 0
K = 15
getNeighbors(94, K)
