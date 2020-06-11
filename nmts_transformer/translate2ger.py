"""
MAKE PREDICTION FOR BUILDING NMTS

@author: vasudevgupta
"""
import pandas as pd
import tensorflow as tf

import spacy
import contractions

from rich import print
import warnings
warnings.filterwarnings('ignore')
import os

# just ensuring that tensorflow instructions don't get displayed in terminal
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

os.chdir('/Users/vasudevgupta/Desktop/github/seq2seq/nmts_transformer')
from model import Transformer, restore_checkpoint, unidirectional_input_mask
from utils import *
from params import params

df= pd.read_csv('text/eng2ger.csv')

tokenize_eng, detokenize_eng, params.len_eng= tokenizer(df['eng_input'], True)
tokenize_ger, detokenize_ger, params.len_ger= tokenizer(df['ger_input'], False)

tokenize_eng['<pad>']= 0
detokenize_eng[0]= "<pad>"
tokenize_ger["<pad>"]= 0
detokenize_ger[0]= "<pad>"

def generate_txt(txt, params):
    """
    txt- english sent
    """
    nlp= spacy.load('en_core_web_sm')
    txt= contractions.fix(txt)
    enc_input= tf.expand_dims(tf.constant([tokenize_eng[tok.text.lower()] for tok in nlp(txt)]), 0)
    transformer= Transformer(num_blocks= params.num_blocks, dmodel= params.dmodel, 
                         depth= params.depth, num_heads= params.num_heads,
                         inp_vocab_size= params.eng_vocab, tar_vocab_size= params.ger_vocab)
    restore_checkpoint(params, transformer)
    dec_input= tf.reshape(tokenize_ger['<sos>'], (1,1))
    att= []
    dec_seq_mask= unidirectional_input_mask(enc_input, dec_input)
    logits= transformer(enc_input, dec_input, dec_seq_mask= dec_seq_mask)
    # beam_search
    bs= BeamSearch(params.k, transformer)
    sents= bs.call(enc_input, logits, params.dec_max_len)
    output= [[detokenize_ger[idx] for idx in sent] for sent in sents]
    
    # att.append(attention_weights)
    return [" ".join(sent) for sent in output]

txt= input('Type anything: ')
sents= generate_txt(txt, params)
print('[Red]' + f'Top {params.k} sentences: ')
for i in range(params.k):
    print('[Purple]'+sents[i]) if i%2 ==0 else print('[Blue]'+sents[i])
    
    