#importing common python libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import re
import pickle

#importing required libraries from nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize 

#importing streamlit
import streamlit as st

#importing required libraries from sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

#Hiding all the unneccesary warnings
import warnings
warnings.filterwarnings("ignore")

#Loading UN sustainable development goals dataset
df = pd.read_csv("/home/surya/Desktop/OnlineClass/ThirdSemester/MDSC-302-Big Data Analytics/Theory/Assignment/undata.csv", header=0,index_col=0)
df.head()

# keeping only the texts whose suggested sdg labels is accepted and the agreement score is at least .6
print('Shape before:', df.shape)
df = df.query('agreement >= .6 and labels_positive > labels_negative').copy()
print('Shape after :', df.shape)

#Taking only required columns for modeling
text_df = df[['sdg', 'text']]
text_df.head()

#Changing the type of text data to string type
text_df['text'] = text_df['text'].astype(str)

#import nltk
#nltk.download('stopwords')
stop = stopwords.words('english')
porter = PorterStemmer()

#Function for removing punctuations from the text.
def remove_punctuation(description):
    table = str.maketrans('', '', string.punctuation)
    return description.translate(table)

#Function for removing stopwords from the text.
def remove_stopwords(text):
    text = [word.lower() for word in text.split() if word.lower() not in stop]
    return " ".join(text)

#Stemming the words to their root word
def stemmer(stem_text):
    stem_text = [porter.stem(word) for word in stem_text.split()]
    return " ".join(stem_text)
    
#Applying the above functions on the whole data
text_df['text'] = text_df['text'].apply(remove_punctuation)
text_df['text'] = text_df['text'].apply(remove_stopwords)
text_df['text'] = text_df['text'].apply(stemmer)

#Splitting the data into train and test sets.
X = text_df['text']
y = text_df['sdg']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 0)

#Pipeline for SGD Classifier
pipeline = Pipeline(
            [
                ('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', SGDClassifier(
                    loss='modified_huber',
                    penalty='l2',
                    alpha=1e-3,
                    random_state=42,
                    max_iter=100,
                    tol=None,
                )),
            ]
        )
#Train model
classifier = pipeline.fit(X_train, y_train)

#Predicting the output for test set and comparing to get the model accuracy
ytest = np.array(y_test)
y_pred = classifier.predict(X_test)
svm_acc = accuracy_score(y_pred, y_test)
print("Accuracy: ",svm_acc)

#Calculating prediction probabilities.
sampleText=["We are also focussing on scaling our adjacent businesses. In Services and Solutions, we are restructuring our distribution channels to cater to different segments while enhancing our manufacturing and execution capability. As we are seeing an urgent need for enhancing health infrastructure in the country, our Nest-In prefabricated product is now being deployed in providing COVID-19 isolation centres and for expansion of COVID-19 bed capacity across the country. We are therefore scaling up capacity and capability in this space. In the New Materials business, we are investing in creating a robust new product funnel while building strategic relationships. Sustainability and climate continues to be core focus areas in our strategy and business operations. We will continue to reduce our carbon emission footprint through process innovation and operational efficiency improvements."]

def probCal(sampleText):
	probs = classifier.predict_proba(sampleText)
	return probs
	
#Saving the model to a pickle file.
with open('model_pkl', 'wb') as files:
    pickle.dump(classifier, files)

