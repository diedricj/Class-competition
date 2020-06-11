import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.ticker import MaxNLocator
from matplotlib import gridspec
import re
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn import svm, datasets
from sklearn.linear_model import LogisticRegression



def sentiment2target(sentiment):
    return {
        'negative': 0,
        'neutral': 1,
        'positive' : 2
    }[sentiment]


def calculate_selected_text(df_row, tol = 0, clf=None, count_vectorizer=None, tfidf_transformer=None):  #modified from kaggle notebook
    
    tweet = df_row['text']
    sentiment = sentiment2target(df_row['sentiment'])
    
    if(sentiment == 1):
        return tweet
        
    words = tweet.split()
    words_len = len(words)
    subsets = [words[i:j+1] for i in range(words_len) for j in range(i,words_len)]
    
    score = 0
    selection_str = '' # This will be our choice
    lst = sorted(subsets, key = len) # Sort candidates by length
    
    
    for sentence in lst:
        
        new_score = 0 # Score for the current substring
        
        #Calculate confidence in correct prediction
        temp = sentence
        temp2 = [None]*1
        temp2[0] = ' '.join(temp)
        vectorized_subset = count_vectorizer.transform(temp2)  #transform test row to feature vector
        tfidf_subset = TfidfTransformer().fit_transform(vectorized_subset)
        #new_score = clf.predict_proba(tfidf_subset)[0][sentiment]
        new_score = clf.decision_function(tfidf_subset)[0][sentiment]
        #new_score = new_score/len(sentence)

        
        if(new_score > (score + tol)):
            score = new_score
            selection_str = sentence
            #return ' '.join(selection_str)

    # If we didn't find good substrings, return the whole text
    if(len(selection_str) == 0):
        selection_str = words
        
    return ' '.join(selection_str)

def jaccard(str1, str2): #from notebook "a simple solution using only class weights"
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

def clean_text(text):  #from kaggle notebook "a simple solution using only class weights"
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text



train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample = pd.read_csv('sample_submission.csv')
#train = train.drop(train[train.sentiment == 'neutral'].index)  #this didnt pan out

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

train_raw, test_raw = train_test_split(train, test_size=0.2, random_state=0)
train_raw.dropna(inplace=True)  #drop empty rows
test_raw.dropna(inplace=True)
train_raw['text_size_weight'] = train_raw.apply(lambda x: (len(x['text'])/len(x['selected_text'])), axis = 1)  #calculate sample weights

print(train_raw.head())
#train_raw = train_raw.drop(train_raw[train_raw.sentiment == 'neutral'].index)




#train_raw['text'] = train_raw['text'].apply(lambda x:clean_text(x))
#train_raw['selected_text'] = train_raw['selected_text'].apply(lambda x:clean_text(x))

count_vectorizer = CountVectorizer(stop_words="english",preprocessor=clean_text, max_df=.1, min_df=10)
tfidf_transformer = TfidfTransformer()

vectorized_data = count_vectorizer.fit_transform(train_raw.text)
vectorized_data_test = count_vectorizer.transform(test_raw.text)
tfidf_train = tfidf_transformer.fit_transform(vectorized_data)
tfidf_test = tfidf_transformer.transform(vectorized_data_test)


targets_train = train_raw.sentiment.apply(sentiment2target)
targets_test = test_raw.sentiment.apply(sentiment2target)

#parameters = {'C':[1, 10]}
#clf = MultinomialNB()
#clf = LogisticRegression(max_iter=100000, C=3.0, class_weight={0:.2, 1:.6, 2:.2})
clf = LinearSVC(random_state=0, max_iter=100000, C=3.0, class_weight={0:.2, 1:.6, 2:.2})
#clf = GridSearchCV(svc, parameters, sample_weight = train_raw['jaccard_selected'])
clf.fit(tfidf_train, targets_train, sample_weight = train_raw['text_size_weight'])
#clf.fit(tfidf_train, targets_train)
#clf.fit(tfidf_train, targets_train)
print("Sentiment Prediction Score", clf.score(tfidf_test, targets_test))



pd.options.mode.chained_assignment = None
count=0
test_raw['predicted_selection'] = ''
for index, row in test_raw.iterrows(): 
    count=count+1
    #print(count)
    selected_text = calculate_selected_text(row, tol = 0.1, clf=clf, count_vectorizer=count_vectorizer, tfidf_transformer=tfidf_transformer)
    test_raw.loc[test_raw['textID'] == row['textID'], ['predicted_selection']] = selected_text
    #sample.loc[sample['textID'] == row['textID'], ['selected_text']] = selected_text
    #print('The jaccard avg for the validation set is:', np.mean(test_raw['jaccard']))

    
test_raw['jaccard'] = test_raw.apply(lambda x: jaccard(x['selected_text'], x['predicted_selection']), axis = 1)

print('The jaccard score for the validation set is:', np.mean(test_raw['jaccard']))
    

