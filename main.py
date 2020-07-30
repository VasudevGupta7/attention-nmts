"""Run this file for training the model

@author: vasudevgupta
"""
import tensorflow as tf

import pandas as pd
import numpy as np

import logging
import os
import yaml

from dataloader import Dataloader
from modeling_transformers import Transformer
from modeling_rnn_attention import Encoder, Decoder 
from trainers import TrainerRNNAttention, TrainerTransformer


if __name__ == '__main__':
    
    args.model_type= 'rnn_attention'
    
    config= yaml.safe_load(open(args.config, 'r'))
    
    # load data
    df= pd.read_csv(args.dataset)
    
    eng_dataloader= Dataloader(config['dataloader']['eng_id'])
    ger_dataloader= Dataloader(config['dataloader']['ger_id'])
    
    df['eng_input']= eng_dataloader.tokenize_col(df['eng_input'])
    df['ger_input']= eng_dataloader.tokenize_col(df['ger_input'])
    df['ger_output']= eng_dataloader.tokenize_col(df['ger_output'])    
            
    
    # DONot use nlp object since it will break < sos > eos sepeartely and problems will happen
    df['teach_force_tok']= df['ger_input'].map(lambda txt: [tokenize_ger[tok] for tok in txt.split(' ')])
    df['target_tok']= df['ger_target'].map(lambda txt: [tokenize_ger[tok] for tok in txt.split(' ')])
        
    if args.model_type == 'rnn_attention':
        
        # lets reverse the whole input eng seq
        df['rev_eng_tok']= df['eng_tok'].map(lambda ls: ls[:: -1])
        
        # Lets make minibatches
        dataset= make_minibatches(df, config, col1= 'rev_eng_tok', col2= 'teach_force_tok', col3= 'target_tok')
        
        # inputs= (enc_seq, teach_force_seq, y)
        
        encoder= Encoder(config)
        decoder= Decoder(config)
        trainer= TrainerRNNAttention(encoder, decoder, config)
        
        ## TRAINING TIME
        grads, avg_loss= trainer.train(dataset, save_model= args.save_model, 
                                     load_model= args.load_model, 
                                     save_evry_ckpt= args.save_evry_ckpt)
        
    elif args.model_type == 'transformer':
     
        # Lets make minibatches
        dataset= make_minibatches(df, config, col1= 'eng_tok', col2= 'teach_force_tok', col3= 'target_tok')
        
        dmodel= config['transformer']['dmodel']
        num_heads= config['transformer']['num_heads']
        num_blocks= config['transformer']['num_blocks']
        inp_vocab_size= config['dataloader']['eng_vocab']
        tar_vocab_size= config['dataloader']['ger_vocab']
        
        transformer= Transformer(num_blocks, dmodel, num_heads,
                                 inp_vocab_size, tar_vocab_size)
        
        trainer= TrainerTransformer(transformer, config)
        
        grads, avg_loss= trainer.train(dataset, save_model= args.save_model, 
                                     load_model= args.load_model, 
                                     save_evry_ckpt= args.save_evry_ckpt)
        
    else:
        raise ValueError(f'input {model_type} is not supported')
        