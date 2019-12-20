
import pandas as pd
ratings = pd.read_csv("D:/roops_backup/projects/data mining project/movie_recommender/movie_dataset.csv")
ratings[['id','title','vote_average','vote_count']]
new = pd.DataFrame(ratings[['id','title','vote_average','vote_count']])
import numpy as np

def get_index(title):
    return ratings[ratings.title == title]["index"].values[0]

popular = pd.DataFrame(ratings['popularity'])
#new.loc[0].get('vote_average')
#genres = ratings['genres']
normalized_popularity = popular.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))

genres = ratings['genres'].fillna('')
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
count = cv.fit_transform(genres)
#print (count.toarray())
a = count.toarray()

movieDict = {}
for i in range(4810):
    movieDict[i] = (new.loc[i].get('id'),new.loc[i].get('title'),new.loc[i].get('vote_average'),new.loc[i].get('vote_count'),normalized_popularity.loc[i].get('popularity'),a[i])
#print (moviedict[49])
#print (moviedict[93])
#from scipy import spatial

def ComputeDistance(a, b):
    genresA = a[5]
    genresB = b[5]
    genreDistance = spatial.distance.cosine(genresA, genresB)
    popularityA = a[4]
    popularityB = b[4]
    popularityDistance = abs(popularityA - popularityB)
    return genreDistance + popularityDistance
    
ComputeDistance(movieDict[2], movieDict[4])

def genre(a, b):
    genresA = a[5]
    genresB = b[5]
    genreDistance = spatial.distance.cosine(genresA, genresB)
    return genreDistance
#import operator

#def getNeighbors(movieID, K):
#    distances = []
 #   for movie in movieDict:
 #       if (movie != movieID):
 #           dist = ComputeDistance(movieDict[movieID], movieDict[movie])
 #           if (dist <= .55):
 #            distances.append((movie, dist))
 #   distances.sort(key=operator.itemgetter(1))
 #   neighbors = []
 #   for x in range(K):
 #       neighbors.append(distances[x][0])
 #   return neighbors

#K = 10
#avgRating = 0
#neighbors = getNeighbors(93, K)
#for neighbor in neighbors:
#    avgRating += movieDict[neighbor][3]
#    print (movieDict[neighbor][1] + " " + str(movieDict[neighbor][2]))
#avgRating /= K   
 # if (movie != movieID):
 #           dist1 = genre(movieDict[movieID], movieDict[movie])
 #           dist2 = ComputeDistance(movieDict[movieID], movieDict[movie])
 #           if(dist1 <=.25):
 #               distances.append((movie, dist1))
 #           if(dist1 >.25 and dist1 <= .35 and dist2 <= .50): 
 #               distances.append((movie, dist2))
 #           dist = ComputeDistance(movieDict[movieID], movieDict[movie])
 #           if(dist <= 0.55):
  #           distances.append((movie, dist))
import operator

def getNeighbors(movieID, K):
    distances = []
    for movie in movieDict:
        if (movie != movieID):
            dist = ComputeDistance(movieDict[movieID], movieDict[movie])
            if(dist <= 0.55):
             distances.append((movie, dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(K):
        neighbors.append((movieDict[distances[x][0]][1],distances[x][1]))
    return neighbors
avgRating = 0

#id = 61

def main(index,K):
  print ("the given movie is",movieDict[index][1])
  neighbors = getNeighbors(index, K)
  for i in range(K):
    print (neighbors[i])  

K = 10
movie="Unbreakable"
index = get_index(movie)
index=int(index)
print (index)
main(index,K)