"""
BUILD NMTS

@author: vasudevgupta
"""
import tensorflow as tf
import pandas as pd
import numpy as np

import time
import rich
from rich.progress import track

import os
os.chdir('/Users/vasudevgupta/Desktop/seq2seq/nmts_attention')

from prepare_data import tokenizer, padding, make_minibatches
from model import *
from params import params

# load data
df= pd.read_csv('text/eng2ger.csv')

# generare tokens
tokenize_eng, detokenize_eng, len_eng= tokenizer(df['eng_input'], True)
tokenize_ger, detokenize_ger, len_ger= tokenizer(df['ger_input'], False)

tokenize_eng['<pad>']= 0
detokenize_eng[0]= "<pad>"
tokenize_ger["<pad>"]= 0
detokenize_ger[0]= "<pad>"

## lets update the params
params.num_samples= df.shape[0]
params.eng_vocab = len_eng+1 # adding 1 because of padding
params.ger_vocab = len_ger+1 

# lets do padding with 0: "<pad>"
df['eng_input']= df['eng_input'].map(lambda txt: padding(txt, params.en_max_len))
df['ger_input']= df['ger_input'].map(lambda txt: padding(txt, params.dec_max_len))
df['ger_target']= df['ger_target'].map(lambda txt: padding(txt, params.dec_max_len))

# num mapping
df['eng_tok']= df['eng_input'].map(lambda txt: [tokenize_eng[tok] for tok in txt.split(' ')])
# DONot use nlp object since it will break < sos > eos sepeartely and problems will happen
df['teach_force_tok']= df['ger_input'].map(lambda txt: [tokenize_ger[tok] for tok in txt.split(' ')])
df['target_tok']= df['ger_target'].map(lambda txt: [tokenize_ger[tok] for tok in txt.split(' ')])

# lets reverse the whole input eng seq
df['rev_eng_tok']= df['eng_tok'].map(lambda ls: ls[:: -1])

# Lets make minibatches
enc_seq, teach_force_seq, y= make_minibatches(df, col1= 'rev_eng_tok', col2= 'teach_force_tok', col3= 'target_tok')

## TRAINING TIME
def train(params):
    encoder= Encoder(params)
    decoder= Decoder(params)
    sce= tf.keras.losses.SparseCategoricalCrossentropy(from_logits= True, reduction= 'none')
    start= time.time()
    avg_loss= []
    
    for e in track(range(1, params.epochs+1)):
        losses= []
        st= time.time()
        for enc_seq_batch, teach_force_seq_batch, y_batch in zip(enc_seq, teach_force_seq, y):
            grads, loss= train_step(params, enc_seq_batch, teach_force_seq_batch, y_batch, encoder, decoder, sce)
            losses.append(loss.numpy())
        avg_loss.append(np.mean(losses))
        save_checkpoints(params, encoder, decoder)
        print(f'EPOCH- {e} ::::::: avgLOSS: {np.mean(losses)} ::::::: TIME: {time.time()- st}')
        print(grads) if e%4 == 0 else None
    
    save_checkpoints(params, encoder, decoder)
    print(f'total time taken: {time.time()-start}')
    return grads, avg_loss

grads, avg_loss= train(params)
print(avg_loss)