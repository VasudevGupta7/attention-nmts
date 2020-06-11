"""
LETS SEE MY NMTS USING TRANSFORMER

@author: vasudevgupta
"""
import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm
import time

import os
os.chdir('/Users/vasudevgupta/Desktop/GitHub/seq2seq/nmts_transformer')

from model import Transformer, loss_fn, train_step, save_checkpoints, restore_checkpoint
from utils import *
from params import params, LearningRate

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

# Lets make minibatches
enc_input, dec_input, dec_output= make_minibatches(df, col1= 'eng_tok', col2= 'teach_force_tok', col3= 'target_tok')

sce= tf.keras.losses.SparseCategoricalCrossentropy(from_logits= True, reduction= 'none')
transformer= Transformer(num_blocks= params.num_blocks, dmodel= params.dmodel, 
                         depth= params.depth, num_heads= params.num_heads,
                         inp_vocab_size= params.eng_vocab, tar_vocab_size= params.ger_vocab)

def train(enc_input, dec_input, dec_output, params= params, sce= sce, transformer= transformer):
    avg_loss= []
    start= time.time()
    for epoch in (range(1, 1+params.epochs)):
        st= time.time()
        losses= []
        for enc_seq, teach_force_seq, y in zip(enc_input, dec_input, dec_output):
            grads, loss= train_step(enc_seq, teach_force_seq, y,
                   transformer= transformer, sce= sce)
            losses.append(loss.numpy())
        avg_loss.append(np.mean(losses))
        if epoch%2 == 0:
            save_checkpoints(params, transformer)
        print(f"EPOCH: {epoch} ::: LOSS: {loss} ::: TIME TAKEN: {time.time()-st}")
    save_checkpoints(params, transformer)
    print('YAYY MODEL IS TRAINED')
    print(f'TOTAL TIME TAKEN- {time.time() - start}')
    return grads, avg_loss

# restore_checkpoint(params, transformer)
grads, avg_loss= train(enc_input, dec_input, dec_output)
