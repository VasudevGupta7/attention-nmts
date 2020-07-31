"""Trainer class

@author: vasudevgupta
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision

import logging

from modeling_transformers import create_padding_mask, unidirectional_input_mask
from training_utils import Trainer
from modeling_utils import OPTM

logger= logging.getLogger(__name__)
        
class TrainerRNNAttention(Trainer):
    
    def __init__(self, 
                 encoder,
                 decoder, 
                 config):
        
        ckpt_dir= config['rnn_attention']['ckpt_dir']
        super(TrainerRNNAttention, self).__init__(ckpt_dir, name= 'rnn_attention')
        
        self.encoder= encoder
        self.decoder= decoder
    
        self.config= config
        self.ckpt_dir= config['rnn_attention']['ckpt_dir']
        self.dec_max_len= config['dataloader']['dec_max_len']
        self.epochs= config['rnn_attention']['epochs']
        
        self.learning_rate= config['rnn_attention']['learning_rate']
        
        self.optimizer= OPTM(config['rnn_attention']['optimizer'], self.learning_rate)
        self.optimizer= mixed_precision.LossScaleOptimizer(self.optimizer, loss_scale= 'dynamic')
        
        self.sce= tf.keras.losses.SparseCategoricalCrossentropy(from_logits= True, reduction= 'none')
    
    # @tf.function
    def train_step(self, enc_in, dec_in, dec_out):
        
        with tf.GradientTape() as gtape:
            
            tot_loss= 0
            enc_seq, hidden1, hidden2= self.encoder(x)
            
            for i in range(self.dec_max_len):
            
                dec_in= tf.expand_dims(ger_inp[:, i], axis= 1)
                ypred, hidden1, hidden2, attention_weights= self.decoder(enc_seq, dec_in, hidden1, hidden2)
                
                timestep_loss= self.rnn_loss(tf.expand_dims(dec_out[:, i], 1), ypred)
                tot_loss+= timestep_loss
           
            avg_timestep_loss= self.optimizer.get_scaled_loss(tot_loss/self.dec_max_len)
        
        trainable_vars= self.encoder.trainable_variables + self.decoder.trainable_variables
        
        grads= gtape.gradient(avg_timestep_loss, trainable_vars)
        grads= self.optimizer.get_unscaled_gradients(grads)
        
        self.optimizer.apply_gradients(zip(grads, trainable_vars))
        
        return avg_timestep_loss
    
    # @tf.function
    def test_step(self, enc_in, dec_in, dec_out):
        
        enc_seq, hidden1, hidden2= self.encoder(x)
        
        for i in range(self.dec_max_len):
            
            dec_in= tf.expand_dims(ger_inp[:, i], axis= 1)
            ypred, hidden1, hidden2, attention_weights= self.decoder(enc_seq, dec_in, hidden1, hidden2)
                
            timestep_loss= self.rnn_loss(tf.expand_dims(dec_out[:, i], 1), ypred)
            tot_loss+= timestep_loss
           
        avg_timestep_loss= tot_loss/self.dec_max_len
        
        return avg_timestep_loss
        
    
    def rnn_loss(self, y, ypred):
        """
        EVERYTHING IS PER TIMESTEP FOR DECODER
        TO AVOID PADDING EFFECT IN CALC OF LOSS;
        WE ARE DOING USING MASK
        """
        
        loss_= self.sce(y, ypred) # loss per timestep for whole batch
        # loss -> (batch_size, 1) SINCE reduction= NONE
        
        mask= tf.cast(tf.not_equal(y, 0), tf.float32) # create mask
        loss_= mask*loss_
        
        return tf.reduce_mean(loss_)
    
class TrainerTransformer(Trainer):
    
    def __init__(self, 
                 transformer, 
                 config):
        
        ckpt_dir= config['transformers']['ckpt_dir']
        super(TrainerTransformer, self).__init__(ckpt_dir, name= 'transformer')
        
        self.transformer= transformer
        self.config= config
        
        self.batch_size= self.config['dataloader']['batch_size']
        
        self.dmodel= config['transformer']['dmodel']
        self.num_heads= config['transformer']['num_heads']
        self.depth= self.dmodel/ self.num_heads
        
        self.warmup_steps= config['transformer']['warmup_steps']
        
        if config['transformer']['learning_rate'] == 'schedule':
            self.learning_rate= LearningRate(self.dmodel, self.warmup_steps)
        else:
            self.learning_rate= config['transformer']['learning_rate']
        
        self.optimizer= OPTM(config['transformer']['optimizer'], self.learning_rate)
        self.optimizer= mixed_precision.LossScaleOptimizer(self.optimizer, loss_scale= 'dynamic')
        
        self.sce= tf.keras.losses.SparseCategoricalCrossentropy(from_logits= True, reduction= 'none')
    
    # @tf.function
    def train_step(self, enc_in, dec_in, dec_out):       
        
        # create appropriate mask
        enc_padding_mask= create_padding_mask(enc_in)
        enc_dec_padding_mask= create_padding_mask(enc_in)
        dec_seq_mask= unidirectional_input_mask(enc_in, dec_in)
        
        with tf.GradientTape() as gtape:
           
            ypred= self.transformer(enc_in, dec_in, enc_padding_mask, enc_dec_padding_mask, dec_seq_mask)
            
            loss= self.transformer_loss_fn(dec_out, ypred)
            loss= self.optimizer.get_scaled_loss(loss)
        
        self.trainable_vars= self.transformer.trainable_variables
        
        grads= gtape.gradient(loss, self.trainable_vars)
        grads= self.optimizer.get_unscaled_gradients(grads)
        
        self.optimizer.apply_gradients(zip(grads, self.transformer.trainable_variables))
        
        return loss
    
    # @tf.function
    def test_step(self, enc_in, dec_in, dec_out):       
        
        # create appropriate mask
        enc_padding_mask= create_padding_mask(enc_input)
        enc_dec_padding_mask= create_padding_mask(enc_input)
        dec_seq_mask= unidirectional_input_mask(enc_input, dec_input)
    
        ypred= self.transformer(enc_in, dec_in, enc_padding_mask, enc_dec_padding_mask, dec_seq_mask)
        loss= self.transformer_loss_fn(dec_out, ypred)
        
        return loss
    
    def transformer_loss_fn(self, y, ypred):
        
        loss_= self.sce(y, ypred)
        # loss_ -> (batch_size, seqlen)
        
        mask= tf.cast(tf.math.not_equal(y, 0), tf.float32)
        # mask -> (batch_size, seqlen)
        
        loss_ = mask*loss_
        # dividing it by global batch size
        return tf.reduce_sum(tf.reduce_mean(loss_, axis= 1)) / self.batch_size

