import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk import word_tokenize

class EmbeddingsHelper(object):
    
    def __init__(self, df,language,embedding_model,stopwords):
        self.language = language
        self.df = df
        self.scores = []
        self.embedding_model = embedding_model
        self.stopwords = stopwords
    
    def tokenize_doc(self,doc):        
        doc = doc.lower()
        doc = word_tokenize(doc)
        doc = [word for word in doc if word not in set(self.stopwords)]
        doc = [word for word in doc if word.isalpha()]
        return doc

    def doc_vector(self,doc):
        doc = [word for word in doc if word in self.embedding_model.key_to_index]
        return np.mean(self.embedding_model[doc],axis=0)
    
    def has_representation(self,doc):
        if len(doc)==0:
            # check if doc is null
            return False
        else:
        # check if at least one word of the document is in the word2vec dictionary
            return not all(word not in self.embedding_model.key_to_index for word in doc)
    def calculate_similarity(self,reference,translation):
        if not self.has_representation(reference) or not self.has_representation(translation):
        # if any of the two documents does not have representation
            return -1
        else:
            vec_reference = np.array(self.doc_vector(reference)).reshape(1,-1)
            vec_translation = np.array(self.doc_vector(translation)).reshape(1,-1)
            cos = cosine_similarity(vec_reference,vec_translation)[0][0]      
            # regularize value of cos to [-1,1]
            if cos<-1.0:cos=-1.0
            if cos>1.0:cos=1.0      
            sim = 1-np.arccos(cos)/np.pi 
            return sim
    
    def wmd(self,reference,translation):
            vec_reference = np.array(self.doc_vector(reference))
            vec_translation = np.array(self.doc_vector(translation))
            return self.embedding_mode.wmdistance(vec_reference,vec_translation)


    
    def regularize_sim(self,sims):
        sim_mean = np.mean([sim for sim in sims if sim!=-1])
        r_sims = []
        errors = 0
        for sim in sims:
            if sim==-1:
                r_sims.append(sim_mean)
                errors += 1
            else:
                r_sims.append(sim)
        return r_sims
    
    def evaluate(self):
        scores = []
        for index,doc in self.df.iterrows():  
            reference = self.tokenize_doc(doc['reference'])
            translation = self.tokenize_doc(doc['translation'])
           # calculate similarity
            score = self.calculate_similarity(reference,translation)
            scores.append(score)
       
        self.scores = self.regularize_sim(scores)

        self.df['scores'] = self.scores
        return self.scores
        

    