"""Run this file for training the model

@author: vasudevgupta
"""
import tensorflow as tf

import pandas as pd
import numpy as np

import logging
import os
import yaml

from train import get_args
from dataloader import Dataloader
from modeling_transformers import Transformer
from modeling_rnn_attention import Encoder, Decoder 
from trainers import TrainerRNNAttention, TrainerTransformer

if __name__ == '__main__':
    
    args= get_args()

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
        
    if not distributed:
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
        
    else:
        
            
        devices= [f'/device:GPU:{i}' for i in range(args.num_gpu)]
        strategy= tf.distribute.MirroredStrategy()
        
        config= yaml.safe_load(open(args.config, 'r'))
        # load data
        df= pd.read_csv(args.dataset)
            
        if args.model_type == 'rnn_attention':
            
            # lets reverse the whole input eng seq
            df['rev_eng_tok']= df['eng_tok'].map(lambda ls: ls[:: -1])
            
            # Lets make minibatches
            enc_seq, teach_force_seq, y= make_minibatches(df, col1= 'rev_eng_tok', col2= 'teach_force_tok', col3= 'target_tok')
            
            inputs= (enc_seq, teach_force_seq, y)
            
            with strategy.scope():
               
                enc_seq= strategy.experimental_distribute_dataset(enc_seq)
                teach_force_seq= strategy.experimental_distribute_dataset(teach_force_seq)
                y= strategy.experimental_distribute_dataset(y)
                
                encoder= Encoder(config)
                decoder= Decoder(config)
                
                trainer= TrainerRNNAttention(encoder, decoder, config)
            
                ## TRAINING TIME
                grads, avg_loss= trainer.distributed_train(inputs, save_model= args.save_model, 
                                         load_model= args.load_model, 
                                         save_evry_ckpt= args.save_evry_ckpt)
            
        elif args.model_type == 'transformers':
         
            # Lets make minibatches
            enc_input, dec_input, dec_output= make_minibatches(df, col1= 'eng_tok', col2= 'teach_force_tok', col3= 'target_tok')
            
            with strategy.scope():
                
                enc_input= strategy.experimental_distribute_dataset(enc_input)
                dec_input= strategy.experimental_distribute_dataset(dec_input)
                dec_output= strategy.experimental_distribute_dataset(dec_output)
                
                depth= config['dmodel']/ config['num_heads']
                
                transformer= Transformer(num_blocks= config['num_blocks'], dmodel= config['dmodel'], 
                                 num_heads= config['num_heads'],inp_vocab_size= config['eng_vocab'], 
                                 tar_vocab_size= config['ger_vocab'])
                
                trainer= TrainerTransformer(transformer, config)
                
                ## TRAINING TIME
                grads, avg_loss= trainer.distributed_train(inputs, save_model= args.save_model, 
                                             load_model= args.load_model, 
                                             save_evry_ckpt= args.save_evry_ckpt)
            
        else:
            print(f'input {model_type} is not supported')
    
            
            
            
            