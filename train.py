"""BUILD NMTS

@author: vasudevgupta
"""
import tensorflow as tf
import pandas as pd
import numpy as np

import time
from tqdm import tqdm
import rich
from rich.progress import track

import os
import yaml
import argparse

from dataloader import tokenizer, padding, make_minibatches
import transformers
from transformers import Transformer, transformer_loss_fn, transformer_train_step
from utils import *
from rnn_attention import Encoder, Decoder, rnn_loss, rnn_train_step

def train_rnn(params):
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

def train_transformers(enc_input, dec_input, dec_output, params= params, sce= sce, transformer= transformer):
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


if __name__ == '__main__':
    
    parser= argparse.ArgumentParser(description='RUN THIS FILE FOR TRAINING THIS MODEL')
    parser.add_argument('--model_type', type= str, help= 'rnn_attention or transformers')
    parser.add_argument('--config', type= str, default= 'config.yaml', help= 'config file for the model')
    parser.add_argument('--save_model', action= 'store_true', default= False)
    parser.add_argument('--load_model', action= 'store_true', default= False)
    
    args= parser.parse_args()
    
    config= yaml.safe_load(open(args.config, 'r'))
    
       
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
        
    if args.model_type == 'rnn_attention':
        
        config= config['rnn_attention']
        
        # lets reverse the whole input eng seq
        df['rev_eng_tok']= df['eng_tok'].map(lambda ls: ls[:: -1])
        
        # Lets make minibatches
        enc_seq, teach_force_seq, y= make_minibatches(df, col1= 'rev_eng_tok', col2= 'teach_force_tok', col3= 'target_tok')
        
        ## TRAINING TIME
        grads, avg_loss= train_rnn(params)
        print(avg_loss)
        
    elif args.model_type == 'transformers':
        
        config= config['transformers']
     
        # Lets make minibatches
        enc_input, dec_input, dec_output= make_minibatches(df, col1= 'eng_tok', col2= 'teach_force_tok', col3= 'target_tok')
        
        depth= config['dmodel']/ config['num_heads']
        
        sce= tf.keras.losses.SparseCategoricalCrossentropy(from_logits= True, reduction= 'none')
        transformer= Transformer(num_blocks= config['num_blocks'], dmodel= config['dmodel'], 
                         depth= depth, num_heads= config['num_heads'],
                         inp_vocab_size= config['eng_vocab'], tar_vocab_size= config['ger_vocab'])
        
        # restore_checkpoint(params, transformer)
        grads, avg_loss= train_transformers(enc_input, dec_input, dec_output)
        
    else:
        print(f'input {model_type} is not supported')
        