#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 21:47:02 2020

@author: vasudevgupta
"""
import tensorflow as tf
import numpy as np

class BeamSearch:
    
    def __init__(self, k, model):
        """
        k- beam search width
        model- decoding model
        """
        self.k= k
        self.model= model
        self.args= [list() for i in range(k)]
         
    def first_step(self, logits):
        """
        logits- (seqlen=1, target_vocab_size)
        """
        probs= tf.nn.softmax(logits, axis= -1)
        topk_probs= tf.math.top_k(probs, self.k)[0].numpy()
        
        topk_args= np.squeeze(tf.math.top_k(probs, self.k)[1].numpy())
        _= [self.args[i].append(topk_args[i]) for i in range(self.k)]
        dec_input= np.array(self.args)
        return dec_input, topk_probs

    def multisteps(self, enc_input, dec_input, topk_probs, tar_vocab_size):
        """
        enc_input- (1, enc_seqlen)
        dec_input- (k, seqlen)
        topk_prob- (1, seqlen)
        """
        dec_seq_mask= unidirectional_input_mask(enc_input, dec_input)
        probs= tf.nn.softmax(self.model(enc_input, dec_input, dec_seq_mask= dec_seq_mask)[:, -1, :])
        # (k, seqlen, tar_vocab_size)
        marginal_probs= np.reshape(topk_probs, (self.k, 1))*probs
        reshaped_marg_probs= marginal_probs.numpy().reshape(1,-1)
        
        topk_probs= tf.math.top_k(reshaped_marg_probs, self.k)[0].numpy()
        
        topk_args= tf.math.top_k(reshaped_marg_probs, self.k)[1].numpy()
        topk_args= self.reindex(topk_args[0], tar_vocab_size)
        
        _= [self.args[i].append(topk_args[i]) for i in range(self.k)]
        return np.array(self.args), topk_probs
        
    def call(self, enc_input, logits, tar_maxlen):
        """
        enc_input- (1, enc_seqlen)
        logits- (seqlen=1, target_vocab_size)
        tar_maxlen- int (maxlen of seq to be outputed)
        """
        dec_input, topk_probs= self.first_step(logits)
        for i in range(tar_maxlen-1):
            dec_input, topk_probs= self.multisteps(enc_input, dec_input, topk_probs, params.ger_vocab)
        return dec_input
            
    def reindex(self, topk_args, tar_vocab_size):
        """
        topk_args- 1D array/ list
        tar_vocab_size- int (vocab size for target language)
        """
        ls= []
        for i in range(self.k):
            if topk_args[i] < tar_vocab_size: 
                a= topk_args[i] 
                ls.append(a)
                continue
            else:
                while True:
                    a= topk_args[i]-tar_vocab_size
                    if a<0:
                        a= topk_args[i]
                        break
                    topk_args[i]= a
                ls.append(a)
        return ls