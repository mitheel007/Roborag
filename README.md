# Roborag
Welcome to robot a robot advisor development model for Indian stock market for any investment people higher portfolio managers stock brokers. Hedge funds mutual funds and investment companies are the one who can use our software.

import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrices import accuracy_score,confusion_matrix
from sklearn.linear_model import PassiveAggressiveClassifier
df=pd.read_csv('news.csv')
df.shape
df.head()
labels=df.label
labels.head()
x_train,x_test,y_train,y_test = train_test_split(df['text'],labels,test_size=0.2,random_state=7)
tfid_vectorizer = TfidfVectorizer(stop_words ='english',max_df = 0.7)
tfidf_train = tfidf_vectorizer.fit_transform(x_train)
tfidf_test = tfidf_vectorizer.transform(x_test)
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)
y_pred = pac.predict(tfidf_test)
score = accuracy_score(y_test,y_pred)
print(f'Accuracy',{round(score*100,2)}%')
confusion_matrix(y_test,y_pred,labels = ['FAKE','REAL'])
