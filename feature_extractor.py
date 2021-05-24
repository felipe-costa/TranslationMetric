from transformers import BertModel, BertTokenizerFast
import pandas as pd
import torch.nn.functional as F
import torch

class FeatureExtrator(object):

    def __init__(self,df):
        self.tokenizer = BertTokenizerFast.from_pretrained("setu4993/LaBSE")
        self.model = BertModel.from_pretrained("setu4993/LaBSE")
        self.model = self.model.eval()
        self.df = df
    
    def similarity(self,embeddings_1, embeddings_2):
        normalized_embeddings_1 = F.normalize(embeddings_1, p=2)
        normalized_embeddings_2 = F.normalize(embeddings_2, p=2)
        return torch.matmul(normalized_embeddings_1, normalized_embeddings_2.transpose(0, 1)
    )

    def extract(self):
        reference = list(self.df.reference)
        translation = list(self.df.translation)

        reference_inputs = self.tokenizer(reference, return_tensors="pt", padding=True)
        translation_inputs = self.tokenizer(translation, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            reference_outputs = self.model(**reference_inputs)
            translation_outputs = self.model(**translation_inputs)
        
        reference_embeddings = reference_outputs.pooler_output
        translation_embeddings = translation_outputs.pooler_output

        return self.similarity(reference_embeddings, translation_embeddings)


