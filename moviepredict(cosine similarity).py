import pandas as pd
import numpy as nm
import movie_predict_knn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_title(index):
    return df[df.index == index]["title"].values[0]

def get_index(title):
    return df[df.title == title]["index"].values[0]

df = pd.read_csv("D:/roops_backup/projects/data mining project/movie_recommender/movie_dataset.csv")
#print (df.head())

features = ['keywords','cast','genres','director']

for feature in features:
    df[feature] = df[feature].fillna('')

def combine_features(row):
    return row['keywords'] +" "+row['cast']+" "+row['genres']+" "+row['director']

df["combined_features"] = df.apply(combine_features,axis=1)

#print (df["combined_features"].head())

cv = CountVectorizer()
count_matrix = cv.fit_transform(df["combined_features"])

cosine_sim = cosine_similarity(count_matrix)

movie="Avengers: Age of Ultron"

index = get_index(movie)
index=int(index)
#print (index)

a=[]
similar_movies = list(enumerate(cosine_sim[index]))
sorted_similar_movies = sorted(similar_movies,key=lambda x:x[1],reverse=True)
i=0
print("result with cosine similarity")
for movies in sorted_similar_movies:
    print (get_title(movies[0]))
    a.append(get_title(movies[0]))
    i=i+1
    if (i>5):
       break


print("result with knn")
K=5
b=[]
movie_predict_knn.main(index,K)
#a.append(b)
#print(b)

