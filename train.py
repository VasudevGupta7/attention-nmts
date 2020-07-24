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
from transformers import Transformer, TrainerTransformer
from rnn_attention import Encoder, Decoder, TrainerRNNAttention

if __name__ == '__main__':
    
    parser= argparse.ArgumentParser(description='RUN THIS FILE FOR TRAINING THIS MODEL')
    parser.add_argument('--model_type', type= str, help= 'rnn_attention or transformers')
    parser.add_argument('--config', type= str, default= 'config.yaml', help= 'config file for the model')
    parser.add_argument('--save_model', action= 'store_true', default= False, help= 'if specified, model will be saved')
    parser.add_argument('--save_evry_ckpt', action= 'store_true', default= False, help= 'if specified, every epoch will be saved')
    parser.add_argument('--load_model', action= 'store_true', default= False, help= 'if specifiled, weights will be restored before training')
    parser.add_argument('--dataset', type= str, default= 'text/eng2ger.csv', help= 'file name of dataset')
    args= parser.parse_args()
    
    config= yaml.safe_load(open(args.config, 'r'))
    # load data
    df= pd.read_csv(args.dataset)
        
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
        
        # lets reverse the whole input eng seq
        df['rev_eng_tok']= df['eng_tok'].map(lambda ls: ls[:: -1])
        
        # Lets make minibatches
        enc_seq, teach_force_seq, y= make_minibatches(df, col1= 'rev_eng_tok', col2= 'teach_force_tok', col3= 'target_tok')
        
        inputs= (enc_seq, teach_force_seq, y)
        
        encoder= Encoder(config)
        decoder= Decoder(config)
        trainer= TrainerRNNAttention(encoder, decoder, config)
        
        ## TRAINING TIME
        grads, avg_loss= trainer.train(inputs, save_model= args.save_model, 
                                     load_model= args.load_model, 
                                     save_evry_ckpt= args.save_evry_ckpt)
        
    elif args.model_type == 'transformers':
     
        # Lets make minibatches
        enc_input, dec_input, dec_output= make_minibatches(df, col1= 'eng_tok', col2= 'teach_force_tok', col3= 'target_tok')
        
        depth= config['dmodel']/ config['num_heads']
        
        transformer= Transformer(num_blocks= config['num_blocks'], dmodel= config['dmodel'], 
                         num_heads= config['num_heads'],inp_vocab_size= config['eng_vocab'], 
                         tar_vocab_size= config['ger_vocab'])
        
        trainer= TrainerTransformer(transformer, config)
        
        # restore_checkpoint(params, transformer)
        grads, avg_loss= trainer.train(inputs, save_model= args.save_model, 
                                     load_model= args.load_model, 
                                     save_evry_ckpt= args.save_evry_ckpt)
        
    else:
        print(f'input {model_type} is not supported')
        