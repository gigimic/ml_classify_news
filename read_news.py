import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

#Read the data
data_news=pd.read_csv('news.csv')

#Get shape and head
print(data_news.shape)
print(data_news.head(10))

labels=data_news.label
print(labels.head())

# split the dataset
x_train,x_test,y_train,y_test=train_test_split(data_news['text'], labels, test_size=0.2, random_state=7)
# Initialize a TfidfVectorizer
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)
print('training data ...')
print(x_train[:5])
print('testing data ...')
print(x_test[:5])

# Fit and transform train set, transform test set
tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)

# Initialize a PassiveAggressiveClassifier
pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)

# Predict on the test set and calculate accuracy
# y_pred=pac.predict(tfidf_test)
# score=accuracy_score(y_test,y_pred)
# print(f'Accuracy: {round(score*100,2)}%')

# 
news1 = data_news['text'][:5]
# news1df = pd.DataFrame(news1)
test_news = tfidf_test=tfidf_vectorizer.transform(news1)
y_pred1=pac.predict(test_news)
print(y_pred1)