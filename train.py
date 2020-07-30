"""train NMTS on single gpu/cpu based machine

@author: vasudevgupta
"""
import tensorflow as tf

import pandas as pd
import numpy as np

import logging
import os
import yaml
import argparse

from dataloader import tokenizer, padding, make_minibatches
from modeling_transformers import Transformer
from modeling_rnn_attention import Encoder, Decoder 
from trainers import TrainerRNNAttention, TrainerTransformer

logger= logging.getLogger(__name__)

if __name__ == '__main__':
    
    parser= argparse.ArgumentParser(description='RUN THIS FILE FOR TRAINING THIS MODEL')
    parser.add_argument('--model_type', type= str, help= 'rnn_attention or transformer')
    parser.add_argument('--config', type= str, default= 'config.yaml', help= 'config file for the model')
    parser.add_argument('--save_model', action= 'store_true', default= False, help= 'if specified, model will be saved')
    parser.add_argument('--save_evry_ckpt', action= 'store_true', default= False, help= 'if specified, every epoch will be saved')
    parser.add_argument('--load_model', action= 'store_true', default= False, help= 'if specifiled, weights will be restored before training')
    parser.add_argument('--dataset', type= str, default= 'data/eng2ger.csv', help= 'file name of dataset')
    args= parser.parse_args()
    
    args.model_type= 'rnn_attention'
    
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
    # config['dataloader']['num_samples']= df.shape[0]
    # config['dataloader']['eng_vocab'] = len_eng+1 # adding 1 because of padding
    # config['dataloader']['ger_vocab'] = len_ger+1 
        
    # lets do padding with 0: "<pad>"
    df['eng_input']= df['eng_input'].map(lambda txt: padding(txt, config['dataloader']['en_max_len']))
    df['ger_input']= df['ger_input'].map(lambda txt: padding(txt, config['dataloader']['dec_max_len']))
    df['ger_target']= df['ger_target'].map(lambda txt: padding(txt, config['dataloader']['dec_max_len']))
        
    # num mapping
    df['eng_tok']= df['eng_input'].map(lambda txt: [tokenize_eng[tok] for tok in txt.split(' ')])
    
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
        