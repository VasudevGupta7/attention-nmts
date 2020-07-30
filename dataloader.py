"""
TEXT PREPROCESSING FOR BUILDING NMTS

@author: vasudevgupta
"""
# using hugging face transformers for initializing word embedding
import transformers

import spacy
import re
import contractions
import unicodedata

import random

import transformers
import tensorflow as tf

import numpy as np
import pandas as pd

import yaml
import os

class preprocess_text:
    """
    DATA PREPROCESSING- NLP
        
        1) LOWERCASE ALL THE WORDS
        2) EXPAND CONTRACTIONS
        3) REMOVE ACCENTS
        4) PUT SPACE BW TOKENS
        5) <SOS>, <EOS> TOKENS APPEND
        6) MAKE VOCAB
        7) DO PADDING TO MAKE CONSTANT SEQ LENGTH IN A BUNDLE
    """
    
    def __init__(self):
        pass
    
    def remove_pattern(self, text, pattern= r'[^a-zA-Z0-9.!?, ]', replace_with= ""):
        return re.sub(pattern, replace_with, text)
    
    def tokenize_sent(self, text, nlp):
        doc= nlp(text)
        return [sent.text for sent in doc.sents]
    
    def tokenize_words(self, text, nlp):
        doc= nlp(text)
        return " ".join(tok.text for tok in doc)
    
    def expand_contractions(self, text):
        # import contractions
        # print(contractions.contractions_dict)
        return contractions.fix(text)
        
    def do_lemmatization(self, text, nlp):
        doc= nlp(text)
        return ' '.join(tok.lemma_ if tok.lemma_ != "-PRON-" else tok.text for tok in doc)
        
    def add_sos_eos(self, text, sos= False, eos= False):
        if (sos and eos):
            return "<sos> " + text + " <eos>" 
        if eos:
            return text + " <eos>"
        if sos:
            return "<sos> " + text
        return text
        
    def remove_accents(self, text):
        ## import unicodedata
        ## normalize text including accents --> ascii --> UTF-8
        # str(sents.numpy(), 'utf-8') == sents.numpy().decode('utf-8')
        return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('UTF-8', 'ignore')

def call_preprocessing(df_col, nlp_en= True, lower_= True, remove_pattern_= False, tokenize_words_= False,
               expand_contractions_= False, do_lemmatization_= False,
               sos= False, eos= False, remove_accents_= False):
    
    nlp= spacy.load('en_core_web_sm') if nlp_en else spacy.load('de_core_news_sm')
    prep= preprocess_text()
    
    if expand_contractions_:
        df_col= df_col.map(lambda text: prep.expand_contractions(text))
        
    if remove_accents_:
        df_col= df_col.map(lambda text: prep.remove_accents(text))
        
    if do_lemmatization_:
        df_col= df_col.map(lambda text: prep.do_lemmatization(text, nlp))
        
    if tokenize_words_:
        df_col= df_col.map(lambda text: prep.tokenize_words(text, nlp))
        
    if remove_pattern_:
        df_col= df_col.map(lambda text: prep.remove_pattern_(text))
    
    if eos or sos:
        df_col= df_col.map(lambda text: prep.add_sos_eos(text, sos, eos))
        
    # do lower if expanding contractions
    if lower_:
        df_col= df_col.map(lambda text: text.lower())
    return df_col

class Dataloader(object):
    
    def __init__(self, model_id):
        
        self.model= transformers.TFAutoModel.from_pretrained(model_id)
        self.tokenizer= transformers.AutoTokenizer.from_pretrained(model_id)
        
    @property
    def tokenize(self):
        return self.tokenizer.tokenize
    
    @property
    def get_ids(self):
        return self.tokenizer.convert_tokens_to_ids
    
    @property
    def get_toks(self):
        return self.tokenizer.convert_ids_to_tokens
    
    def tokenize_col(self, df_col):
        return df_col.apply(lambda x: self.get_ids(self.tokenize(x)))
    
    def get_weights(self):
        embed_weights= self.model.transformer.wte.get_weights()[0]
        return embed_weights
        
    def get_dataset(self, df_col):
        
        dataset= tf.data.Dataset.from_tensor_slices(df_col.values)
        dataset= dataset.map(lambda x: self.tokenize(x))
    
        return dataset

