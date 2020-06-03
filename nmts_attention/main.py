"""
BUILD NMTS

@author: vasudevgupta
"""
import tensorflow as tf
import pandas as pd

import time
import rich
from rich.progress import track

from prepare_data import tokenizer
from model import Encoder, Decoder, train_step
from params import params

import os
os.chdir('/Users/vasudevgupta/Desktop/seq2seq/nmts_attention')

def train(params, x, ger_inp, ger_out):
    encoder= Encoder(params)
    decoder= Decoder(params)
    sce= tf.keras.losses.SparseCategoricalCrossentropy(from_logits= True, reduction= 'none')
    for i in track(range(params.epochs)):
        grads, loss= train_step(params, x, ger_inp, ger_out, encoder, decoder, sce)
    return grads, loss

# load data
df= pd.read_csv('text/eng2ger.csv')

# generare tokens
tokenize_eng, detokenize_eng, params.len_eng= tokenizer(df['eng_input'], True)
tokenize_ger, detokenize_ger, params.len_ger= tokenizer(df['ger_input'], False)

# num mapping
df['eng_tok']= df['eng_input'].map(lambda txt: [tokenize_eng[tok] for tok in txt.split(' ')])
# DONot use nlp object since it will break < sos > eos sepeartely and problems will happen
df['teach_force_tok']= df['ger_input'].map(lambda txt: [tokenize_ger[tok] for tok in txt.split(' ')])
df['target_tok']= df['ger_target'].map(lambda txt: [tokenize_ger[tok] for tok in txt.split(' ')])

# give input
x = tf.keras.backend.ones(shape=(2, 7))
ger_inp= tf.keras.backend.ones(shape=(2, 9))
ger_out= tf.keras.backend.ones(shape=(2, 9))
grads, loss= train(params, x, ger_inp, ger_out)




