"""
TEXT PREPROCESSING FOR BUILDING NMTS

@author: vasudevgupta
"""
import pandas as pd
import tensorflow as tf

from prepare_data import tokenizer
from model import Encoder, Decoder, restore_checkpoint
from params import params

import spacy
import contractions

from rich import print
import warnings
warnings.filterwarnings('ignore')
import os
# just to ensure that tensorflow instructions don't get displayed in terminal
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.chdir('/Users/vasudevgupta/Desktop/seq2seq/nmts_attention')

df= pd.read_csv('text/eng2ger.csv')

tokenize_eng, detokenize_eng, params.len_eng= tokenizer(df['eng_input'], True)
tokenize_ger, detokenize_ger, params.len_ger= tokenizer(df['ger_input'], False)

## Lets make some translation
def make_prediction(txt, params, greedy= False, random_sampling= True, beam_search= False):
    """
    txt- english sent
    """
    nlp= spacy.load('en_core_web_sm')
    txt= contractions.fix(txt)
    x= tf.expand_dims(tf.constant([tokenize_eng[tok.text.lower()] for tok in nlp(txt)]), 0)
    encoder= Encoder(params)
    decoder= Decoder(params)
    restore_checkpoint(params, encoder, decoder)
    dec_inp= tf.reshape(tokenize_ger['<sos>'], (1,1))
    final_tok, i= '<sos>', 0
    sent, att= [], []
    enc_seq, hidden1, hidden2= encoder(x)
    while final_tok != '<eos>':
        ypred, hidden1, hidden2, attention_weights= decoder(enc_seq, dec_inp, hidden1, hidden2)
        if random_sampling:
            idx= tf.random.categorical(ypred[:, 0, :], num_samples= 1)
        elif greedy:
            idx= tf.argmax(ypred[:, 0, :], axis= -1)
        elif beam_search:
            pass
        sent.append(detokenize_ger[tf.squeeze(idx).numpy()])
        att.append(attention_weights)
        dec_inp= idx # teacher forcing with predicted output
        if i== 10:
            break
        else:
            i+=1
    return " ".join(sent), att

txt= input('Type anything: ')
sent, att= make_prediction(txt, params)
print('[bold blue]'+sent)
