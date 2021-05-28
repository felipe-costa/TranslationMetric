from transformers import BertModel, BertTokenizerFast
import tensorflow as tf
import pandas as pd
import torch.nn.functional as F
import torch

class FeatureExtrator(object):

    def __init__(self):
        self.tokenizer = BertTokenizerFast.from_pretrained("setu4993/LaBSE")
        self.model = BertModel.from_pretrained("setu4993/LaBSE")
        self.model = self.model.eval()
    
    def similarity(self,embeddings_1, embeddings_2):
        normalized_embeddings_1 = F.normalize(embeddings_1, p=2)
        normalized_embeddings_2 = F.normalize(embeddings_2, p=2)
        return torch.matmul(normalized_embeddings_1, normalized_embeddings_2.transpose(0, 1)
    )

    def extract(self,df):
        reference = list(df.reference)
        translation = list(df.translation)

        reference_inputs = self.tokenizer(reference, return_tensors="pt", padding=True)
        translation_inputs = self.tokenizer(translation, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            reference_outputs = self.model(**reference_inputs)
            translation_outputs = self.model(**translation_inputs)
        
        reference_embeddings = reference_outputs.pooler_output
        translation_embeddings = translation_outputs.pooler_output

        return tf.concat([reference_embeddings, translation_embeddings], 1)


