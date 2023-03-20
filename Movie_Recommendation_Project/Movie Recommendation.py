import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option("display.max_columns", None)  # Bütün sütunları gösterir
pd.set_option("display.max_rows", None)  # Bütün satırları gösterir
pd.set_option("display.expand_frame_repr", False)
pd.set_option("display.float_format", lambda x: "%.5f" % x)

movies = pd.read_csv("datasets/tmdb_5000_movies.csv")
credits = pd.read_csv("datasets/tmdb_5000_credits.csv")

movies.head()


def check_df(dataframe):
    print("**********head**********")
    print(dataframe.head())
    print("**********isna**********")
    print(dataframe.isnull().sum())
    print("**********shape*********")
    print(dataframe.shape)
    print("*********info***********")
    print(dataframe.info())
    print("*********describe*******")
    print(dataframe.describe().T)
    print("*********nuniq*******")
    print(dataframe.nunique())


check_df(movies)  # (4803, 20)

check_df(credits)  # (4803 , 4)

movies = movies.merge(credits, on="title")
movies.head()  # (4803, 23)

movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

movies.head()

import ast  # The ast module helps Python applications to process trees of the Python abstract syntax grammar.


# The abstract syntax itself might change with each Python release; this module helps to find out programmatically what the current grammar looks like.


def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name'])
    return L


movies.dropna(inplace=True)  # NA değerleri veri setinden attık

movies['genres'] = movies['genres'].apply(convert)  # Eski karışık yazı yerine,sadece türlerin yer aldığı yazı
movies.head()

#
movies['keywords'] = movies['keywords'].apply(convert)
movies.head()


#
def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter < 3:
            L.append(i["name"])
            counter += 1
        else:
            break
    return L


movies["cast"] = movies["cast"].apply(convert3)

movies["cast"].head()

movies['cast'] = movies['cast'].apply(lambda x: x[0:3])
movies['cast'].head()


# 0   [Sam Worthington, Zoe Saldana, Sigourney Weaver]
# 1       [Johnny Depp, Orlando Bloom, Keira Knightley]
# 2        [Daniel Craig, Christoph Waltz, Léa Seydoux]
# 3        [Christian Bale, Michael Caine, Gary Oldman]
# 4      [Taylor Kitsch, Lynn Collins, Samantha Morton]

#
def fetch_director(text):  # Crew'dan Director seçmek için kullanıyoruz.
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L


movies["crew"] = movies["crew"].apply(fetch_director)

movies["crew"].head(5)

#
movies["overview"][3]


# movies["overview"] = movies["overview"].apply(
#     lambda x: x.split()) # ayırmak için örn: [Ali comes to us] - [Ali, comes, to, us]


def collapse(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ", ""))
    return L1


movies['cast'] = movies['cast'].apply(collapse)
movies['crew'] = movies['crew'].apply(collapse)
movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)  # boşlukları sildik
movies.head()

#
movies['overview'] = movies['overview'].apply(lambda x: x.split())
movies["tags"] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
movies["tags"][0]

new = movies.drop(columns=['overview', 'genres', 'keywords', 'cast', 'crew'])
# new.head()

new['tags'] = new['tags'].apply(lambda x: " ".join(x))
new["tags"][0]

new["tags"] = new["tags"].apply(lambda x: x.lower())

import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)


new["tags"] = new["tags"].apply(stem)

new["tags"][0]

###

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=5000, stop_words="english")  # 5000 en sık geçen kelimeyi aldık.

vectors = cv.fit_transform(new["tags"]).toarray()  # Array haline getirdik.
vectors[0]

cv.get_feature_names()  # Bu komut ile o 5000 adet item' ın neler old. baktık.

len(cv.get_feature_names())  # Uzunluğuna baktık # 5000

from sklearn.metrics.pairwise import cosine_similarity

cosine_similarity(
    vectors).shape  # Cosine similarity measures the similarity between two vectors of an inner product space.
similarity = cosine_similarity(vectors)
# !! It is often used to measure document similarity in text analysis. !!
similarity[0]

######### MAIN FUNCTION
# new[new["title"] == "Batman Begins"].index[0] # Sonucu Tek bir değer olarak bastı. (119)

# sorted(similarity[0], reverse=True) En yakından en uzağa doğru ilişkileri verdi.

# sorted(list(enumerate(similarity[0])), reverse=True, key=lambda x: x[1])[1:6]

# new.iloc[141].title : 'Aliens vs Predator: Requiem'

def recommend(movie):
    movie_index = movies[movies["title"] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    for i in movies_list:
        print(new.iloc[i[0]].title)


recommend("Zodiac")


####

import pickle
pickle.dump(new, open("movies.pkl", "wb"))

pickle.dump(new.to_dict(), open("movie_dict.pkl", "wb"))

pickle.dump(similarity, open("similarity.pkl", "wb"))

