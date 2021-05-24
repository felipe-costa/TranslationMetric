from bs4 import BeautifulSoup
from nltk.stem import SnowballStemmer
from tqdm import tqdm
from nltk.stem.wordnet import WordNetLemmatizer
import pandas as pd
import config
import jieba
import os
import re 
def clean(text_list, lemmatize, stemmer,language):
    """
    Function that a receives a list of strings and preprocesses it.
    
    :param text_list: List of strings.
    :param lemmatize: Tag to apply lemmatization if True.
    :param stemmer: Tag to apply the stemmer if True.
    """
   
    updates = []
    for j in tqdm(range(len(text_list))):
        
        text = text_list[j]
        
        if language == "chinese":
            updates.append(preprocess_data_cn(text))
        else:
            #LOWERCASE TEXT
            text = text.lower()
        
            #REMOVE NUMERICAL DATA AND PUNCTUATION
            text = re.sub("[^a-zA-Z]", ' ', text)
        
            #REMOVE TAGS
            text = BeautifulSoup(text).get_text()
        
            if lemmatize:
                lemma = WordNetLemmatizer()
                text = " ".join(lemma.lemmatize(word) for word in text.split())
        
            if stemmer:
                snowball_stemmer = SnowballStemmer(language)
                text = " ".join(snowball_stemmer.stem(word) for word in text.split())
        
            updates.append(text)
        
    return updates

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def update_df(dataframe, list_updated, column_name):
    dataframe.update(pd.DataFrame({column_name: list_updated}))

def get_language(d):
    if '-en' in d:
        return "english"
    elif '-zh' in d:
        return "chinese"
    elif '-fi' in d:
        return "finnish"
    else:
        return "english"


def get_model_path(d):
    if '-en' in d:
        return config.english_embeddings
    elif '-zh' in d:
        return config.chinese_embeddings
    elif '-fi' in d:
        return config.finnish_embeddings
    else:
        return config.english_embeddings

def preprocess_data_cn(doc):
    '''
    Function: preprocess data in Chinese including cleaning, tokenzing...
    Input: 
        stopwords: Chinese stopwords list
        doc: document string
    Output: list of words
    '''       
    # clean data
    doc = re.sub(u"[^\u4E00-\u9FFF]", "", doc) # delete all non-chinese characters
    doc = re.sub(u"[儿]", "", doc) # delete 儿
    # tokenize and move stopwords 
    return doc