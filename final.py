import re
import pandas as pd
import gensim
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from snowballstemmer import stemmer
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from gensim.models import Word2Vec
import random
# from googletrans import Translator
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,accuracy_score, classification_report

# model = gensim.models.Word2Vec.load("models/full_grams_cbow_100_twitter.mdl")

def clean_text(text):
    search = ["أ","إ","آ","ة","_","-","/",".","،"," و "," يا ",'"',"ـ","'","ى","\\",'\n', '\t','&quot;','?','؟','!']
    replace = ["ا","ا","ا","ه"," "," ","","",""," و"," يا","","","","ي","",' ', ' ',' ',' ? ',' ؟ ',' ! ']  
    p_tashkeel = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
    text = re.sub(p_tashkeel,"", text)
    p_longation = re.compile(r'(.)\1+')
    subst = r"\1\1"
    text = re.sub(p_longation, subst, text)
    text = text.replace('وو', 'و')
    text = text.replace('يي', 'ي')
    text = text.replace('اا', 'ا')
    
    for i in range(0, len(search)):
        text = text.replace(search[i], replace[i])

    text = text.strip()
    return text
# url="https://raw.githubusercontent.com/DinaElSakaty/bachelor/main/sas-ar.csv"
data= pd.read_csv('sas-ar.csv', encoding='UTF-16') 
data=data.dropna()
# data=data.astype(str)

arabic_diacritics = re.compile(""" ّ    | # Tashdid
                             َ    | # Fatha
                             ً    | # Tanwin Fath
                             ُ    | # Damma
                             ٌ    | # Tanwin Damm
                             ِ    | # Kasra
                             ٍ    | # Tanwin Kasr
                             ْ    | # Sukun
                             ـ     # Tatwil/Kashida
                         """, re.VERBOSE)
def remove_diacritics(text):
    text = re.sub(arabic_diacritics, '', str(text))
    return text

def remove_repeating_char(text):
    return re.sub(r'(.)\1+', r'\1\1', str(text))  

def stemming(text):
    snowball = stemmer("arabic")
    stemSentence = ""
    for word in text.split():
        stem = snowball.stemWord(word)
        stemSentence += stem
        stemSentence += " "
    stemSentence = stemSentence.strip()
    return stemSentence

def process_text(text, grams=False):
    clean_text = remove_diacritics(text)
    clean_text = remove_repeating_char(clean_text)
    clean_text=stemming(clean_text)
    return clean_text

# data['EssayText'] = data['EssayText'].apply(process_text)

test_data = data.sample(frac=0.15,random_state=80)
data=data.drop(test_data.index)

X_train = data.EssayText.values
Y_train = data.Score2.values

X_val= test_data.EssayText.values
Y_val = test_data.Score2.values

pipe = make_pipeline(TfidfVectorizer(),
                    RandomForestClassifier())

param_grid = {'randomforestclassifier__n_estimators':[10, 100, 1000],
             'randomforestclassifier__max_features':['sqrt', 'log2']}

rf_model = GridSearchCV(pipe, param_grid, cv=5)
rf_model.fit(X_train,Y_train)

prediction = rf_model.predict(X_val)
print(f"Accuracy score is {accuracy_score(Y_val, prediction):.2f}")
print(classification_report(Y_val, prediction))

# # # Augmentation 

def synonym_replacement(sequence,augment,synonym):
    words = word_tokenize(sequence)
    new_sentences=[]
    for j in range(augment):
        random_word=random.choice(words)
        token=clean_text(random_word).replace(" ","_")
        if token in model.wv:
         for i in range(synonym):
             most_similar=model.wv.most_similar(token,topn=synonym)
             for term, score in most_similar:
                term = clean_text(term).replace(" ", "_")
                if term != token:
                    output=sequence.replace(random_word,term)
                    new_sentences.append(output)
    return new_sentences

def drop_duplicates(sentence):
    aug_syn=synonym_replacement(sentence, 2, 1)
    mylist = list(dict.fromkeys(aug_syn))
    return mylist

# data['EssayTextAugmented'] = (data.apply(lambda x: drop_duplicates(x.EssayText), axis=1)) 

# data.to_csv(r'C:\Users\Lenovo\Desktop\Bachelor\TakeTwo\augmented.csv')

# url="https://raw.githubusercontent.com/DinaElSakaty/bachelor/main/augmented.csv"

df= pd.read_csv('augmented.csv', encoding="utf-8") 
df.dropna()

test_data = df.sample(frac=0.15,random_state=80)
df=df.drop(test_data.index)

X_train = df.EssayText.values
Y_train = df.Score2.values

X_val= test_data.EssayText.values
Y_val = test_data.Score2.values

#  make pipeline
pipe = make_pipeline(TfidfVectorizer(),
                    RandomForestClassifier())

param_grid = {'randomforestclassifier__n_estimators':[10, 100, 1000],
             'randomforestclassifier__max_features':['auto']}

rf_model = GridSearchCV(pipe, param_grid, cv=5)
rf_model.fit(X_train,Y_train)

# # make prediction and print accuracy
prediction = rf_model.predict(X_val)
print(f"Accuracy score is {accuracy_score(Y_val, prediction):.2f}")
print(classification_report(Y_val, prediction))

