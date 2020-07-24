"""Run this file for translation

@author: vasudevgupta
"""
import pandas as pd
import tensorflow as tf

import spacy
import contractions

import yaml
import argparse

import os
from rich import print
import warnings
warnings.filterwarnings('ignore')

# just to ensure that tensorflow instructions don't get displayed in terminal
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from dataloader import tokenizer
from transformers import Transformer, TrainerTransformer
from rnn_attention import Encoder, Decoder, TrainerRNNAttention
from beam_search import BeamSearch

## Lets make some translation
def rnn_generate_txt(txt, config, greedy= False, random_sampling= True, beam_search= False):
    """
    txt- english sent
    """
    nlp= spacy.load('en_core_web_sm')
    txt= contractions.fix(txt)
    
    x= tf.expand_dims(tf.constant([tokenize_eng[tok.text.lower()] for tok in nlp(txt)]), 0)
    
    encoder= Encoder(config)
    decoder= Decoder(config)
    
    trainer= TrainerRNNAttention(encoder, decoder, config)
    trainer.restore_checkpoint(config['rnn_attention']['ckpt_dir'])
    
    dec_inp= tf.reshape(tokenize_ger['<sos>'], (1,1))
    
    final_tok, i= '<sos>', 0
    
    sent, att= [], []
    
    enc_seq, hidden1, hidden2= encoder(x)
    
    while final_tok != '<eos>':
    
        ypred, hidden1, hidden2, attention_weights= trainer.decoder(enc_seq, dec_inp, hidden1, hidden2)
        
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

## Lets make some translation
def transformers_generate_txt(txt, params):
    """
    txt- english sent
    """
    nlp= spacy.load('en_core_web_sm')
    
    txt= contractions.fix(txt)
    
    enc_input= tf.expand_dims(tf.constant([tokenize_eng[tok.text.lower()] for tok in nlp(txt)]), 0)
    
    depth= config['transformer']['dmodel']/ config['transformer']['num_heads']
    dmodel= config['transformer']['dmodel']
    num_blocks= config['transformer']['num_blocks']
    num_heads= config['transformer']['num_heads']
    
    transformer= Transformer(num_blocks= num_blocks, dmodel= dmodel, 
                         depth= depth, num_heads= num_heads,
                         inp_vocab_size= config['dataloader']['eng_vocab'],
                         tar_vocab_size= config['dataloader']['ger_vocab'])
    
    trainer= TrainerTransformer(transformer, config)
    trainer.restore_checkpoint(config['transformer']['ckpt_dir'])
    
    dec_input= tf.reshape(tokenize_ger['<sos>'], (1,1))
    att= []
    
    dec_seq_mask= unidirectional_input_mask(enc_input, dec_input)
    logits= transformer(enc_input, dec_input, dec_seq_mask= dec_seq_mask)
    
    # beam_search
    bs= BeamSearch(config['transformer']['k', 1], trainer.transformer)
    
    sents= bs.call(enc_input, logits, params.dec_max_len)
    
    output= [[detokenize_ger[idx] for idx in sent] for sent in sents]
    
    # att.append(attention_weights)
    return [" ".join(sent) for sent in output]


if __name__ == '__main__':
    
    parser= argparse.ArgumentParser(description= 'RUN THIS FILE TO TRANLATING TO GERMAN ')
    
    parser.add_argument('--model_type', type= str, help= 'rnn_attention or transformers')
    parser.add_argument('--config', type= str, default= 'config.yaml', help= 'link configuration file of model')
    parser.add_argument('--dataset', type= str, default= 'text/eng2ger.csv', help= 'dataset to get tokenizer')
    
    args= parser.parse_args()
    
    config= yaml.safe_load(open(args.config, 'r'))
    df= pd.read_csv(args.dataset)
    
    tokenize_eng, detokenize_eng, len_eng= tokenizer(df['eng_input'], True)
    tokenize_ger, detokenize_ger, len_ger= tokenizer(df['ger_input'], False)
    
    tokenize_eng['<pad>']= 0
    detokenize_eng[0]= "<pad>"
    tokenize_ger["<pad>"]= 0
    detokenize_ger[0]= "<pad>"
    
    txt= input('Type anything: ')
    k = config['transformers'].get('k', 1)
    
    if args.model_type == 'rnn_attention':
        sent, att= rnn_generate_txt(txt, config, random_sampling=True)
        print('[bold blue]'+sent)
    
    elif args.model_type == 'transformers':
        sents= transformers_generate_txt(txt, config)
        print('[Red]' + f'Top {k} sentences: ')
        for i in range(k):
            print('[Purple]'+sents[i]) if i%2 ==0 else print('[Blue]'+sents[i])
        
    else:
        print(f'input {model_type} is not supported')