from nltk.tokenize import word_tokenize
import pulp
from wmd import word_mover_distance

class MetricFeature(object):
    def __init__(self,reference,translation,baseline):
        
        self.baseline = baseline
        self.reference_str = reference
        self.translation_str = translation
        self.reference = word_tokenize(reference)
        self.translation = word_tokenize(translation)
        
        self.cosine_similarity = 0
        self.wmd = 0
        
        self.missing_words_reference = 0
        self.missing_words_translation = 0
        self.total_reference_words = 0
        self.total_translation_words = 0
        
    
    def compute_totals(self):
        self.total_reference_words = len(self.reference)
        self.total_translation_words = len(self.translation)
        self.missing_words_reference = len(set(self.translation) - set(self.reference))
        self.missing_words_translation = len(set(self.reference) - set(self.translation))

        
    def similarity(self):
        vec_reference = self.baseline.tokenize_doc(self.reference_str)
        vec_translation = self.baseline.tokenize_doc(self.translation_str)
        self.cosine_similarity = self.baseline.calculate_similarity(vec_reference,vec_translation)
        self.wmd = word_mover_distance(vec_reference,vec_translation,self.baseline.model)
    
    def fit_transform(self):
        self.compute_totals()
        self.similarity()
    
    
    def __str__(self):
        str_repr =  f"Cosine Similarity: {self.cosine_similarity}\n" 
        str_repr += f"Missing words on reference: {self.missing_words_reference} \n" 
        str_repr += f"Missing words on translation: {self.missing_words_translation} \n"
        str_repr += f"Total words on reference: {self.total_reference_words} \n"
        str_repr += f"Total words on Translation: {self.total_translation_words}"
        return str_repr
