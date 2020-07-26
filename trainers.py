"""Trainers

@author: vasudevgupta
"""
import numpy as np
import tensorflow as tf

import time
from tqdm import tqdm
import rich
from rich.progress import track

import logging
import os
import time

from transformers import create_padding_mask, unidirectional_input_mask
from utils import LearningRate, OPTM

logger= logging.getLogger(__name__)

class TrainerRNNAttention(object):
    
    def __init__(self, encoder, decoder, config):
        
        self.encoder= encoder
        self.decoder= decoder
    
        self.config= config
        self.ckpt_dir= config['rnn_attention']['ckpt_dir']
        self.dec_max_len= config['dataloader']['dec_max_len']
        self.epochs= config['rnn_attention']['epochs']
        
        self.learning_rate= config['rnn_attention']['learning_rate']
        self.optimizer= OPTM(config['rnn_attention']['optimizer'], self.learning_rate)
        
        self.sce= tf.keras.losses.SparseCategoricalCrossentropy(from_logits= True, reduction= 'none')
    
    def train(self, dataset, save_model= False, load_model= False, save_evry_ckpt= False):
        
        # enc_seq, teach_force_seq, y= inputs[0], inputs[1], inputs[2]
        
        start= time.time()
        avg_loss= []
        
        if load_model: self.restore_checkpoint(self.ckpt_dir)
        
        for e in track(range(1, self.epochs+1)):
            
            losses= []
            st= time.time()
           
            for enc_seq_batch, teach_force_seq_batch, y_batch in dataset:
                loss, grads= self.train_step(enc_seq_batch, teach_force_seq_batch, y_batch)
                losses.append(loss.numpy())
            
            avg_loss.append(np.mean(losses))
            
            if save_evry_ckpt: self.save_checkpoints(self.ckpt_dir)
            
            logger.info(f'EPOCH- {e} ::::::: avgLOSS: {np.mean(losses)} ::::::: TIME: {time.time()- st}')
            logger.info(grads) if e%4 == 0 else None
        
        if save_model: self.save_checkpoints(self.ckpt_dir)
        
        print(f'total time taken: {time.time()-start}')
        
        return grads, avg_loss
    
    # @tf.function
    def train_step(self, x, ger_inp, ger_out):
        
        with tf.GradientTape() as gtape:
            
            tot_loss= 0
            enc_seq, hidden1, hidden2= self.encoder(x)
            
            for i in range(self.dec_max_len):
            
                dec_inp= tf.expand_dims(ger_inp[:, i], axis= 1)
                ypred, hidden1, hidden2, attention_weights= self.decoder(enc_seq, dec_inp, hidden1, hidden2)
                
                timestep_loss= self.rnn_loss(tf.expand_dims(ger_out[:, i], 1), ypred)
                tot_loss+= timestep_loss
           
            avg_timestep_loss= tot_loss/self.dec_max_len
        
        trainable_vars= self.encoder.trainable_variables + self.decoder.trainable_variables
        grads= gtape.gradient(avg_timestep_loss, trainable_vars)
        
        self.optimizer.apply_gradients(zip(grads, trainable_vars))
        
        return avg_timestep_loss, grads
    
    def rnn_loss(self, y, ypred):
        """
        EVERYTHING IS PER TIMESTEP FOR DECODER
        TO AVOID PADDING EFFECT IN CALC OF LOSS;
        WE ARE DOING USING MASK
        """
        
        loss_= self.sce(y, ypred) # loss per timestep for whole batch
        # shape= (batch_size, 1) SINCE reduction= NONE
        
        mask= tf.cast(tf.not_equal(y, 0), tf.float32) # create mask
        loss_= mask*loss_
        
        return tf.reduce_mean(loss_)
    
    def distributed_train(self):
        pass
    
    def save_checkpoints(self, ckpt_dir= 'weights/rnn_attention_ckpts'):
        
        checkpoint_dir = ckpt_dir
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        
        ckpt= tf.train.Checkpoint(optimizer= self.optimizer,
                                  encoder= self.encoder,
                                  decoder= self.decoder)
        ckpt.save(file_prefix= checkpoint_prefix)
            
    def restore_checkpoint(self, ckpt_dir= 'weights/rnn_attention'):
       
        checkpoint_dir = ckpt_dir
        
        ckpt= tf.train.Checkpoint(optimizer= self.optimizer,
                                  encoder= self.encoder,
                                  decoder= self.decoder)
        ckpt.restore(tf.train.latest_checkpoint(checkpoint_dir))

class TrainerTransformer(object):
    
    def __init__(self, transformer, config):
        
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
        
        self.sce= tf.keras.losses.SparseCategoricalCrossentropy(from_logits= True, reduction= 'none')
    
    def train(self, dataset, save_model= False, load_model= False, save_evry_ckpt= False):
        
        # enc_input, dec_input, dec_output= inputs[0], inputs[1], inputs[2]
        
        avg_loss= []
        start= time.time()
        
        if load_model: self.restore_checkpoint(self.config['transformer']['ckpt_dir'])
        
        for epoch in (range(1, 1+self.config['transformer']['epochs'])):
            
            st= time.time()
            losses= []
            
            for enc_seq, teach_force_seq, y in dataset:
              
                loss, grads= self.train_step(enc_seq, teach_force_seq, y)
                losses.append(loss.numpy())
            
            avg_loss.append(np.mean(losses))
        
            if save_evry_ckpt: self.save_checkpoints(self.config['transformer']['ckpt_dir'])
            
            logger.info(f"EPOCH: {epoch} ::: LOSS: {loss} ::: TIME TAKEN: {time.time()-st}")
        
        if save_model: self.save_checkpoints(self.config['transformer']['ckpt_dir'])
        
        print('YAYY MODEL IS TRAINED')
        print(f'TOTAL TIME TAKEN- {time.time() - start}')
        
        return grads, avg_loss

    # @tf.function
    def train_step(self, enc_input, dec_input, dec_output):       
        
        # create appropriate mask
        enc_padding_mask= create_padding_mask(enc_input)
        enc_dec_padding_mask= create_padding_mask(enc_input)
        dec_seq_mask= unidirectional_input_mask(enc_input, dec_input)
        
        with tf.GradientTape() as gtape:
           
            ypred= self.transformer(enc_input, dec_input, enc_padding_mask, enc_dec_padding_mask, dec_seq_mask)
            loss= self.transformer_loss_fn(dec_output, ypred)
        
        grads= gtape.gradient(loss, self.transformer.trainable_variables)
        
        self.optimizer.apply_gradients(zip(grads, self.transformer.trainable_variables))
        
        return loss, grads
    
    def transformer_loss_fn(self, y, ypred):
        
        loss_= self.sce(y, ypred)
        # loss_- (batch_size, seqlen)
        
        mask= tf.cast(tf.math.not_equal(y, 0), tf.float32)
        # mask- (batch_size, seqlen)
        
        loss_ = mask*loss_
        # dividing it by global batch size
        return tf.reduce_sum(tf.reduce_mean(loss_, axis= 1)) / self.batch_size
        
    # def distributed_train(self, dataset, strategy):
        
    #     self.strategy= strategy
    #     self.num_replicas= self.batch_size / self.strategy.num_replicas_in_sync
        
    #     avg_loss= []
    #     start= time.time()
        
    #     if load_model: self.restore_checkpoint(self.config['transformer']['ckpt_dir'])
        
    #     for epoch in (range(1, 1+self.config['transformer']['epochs'])):
            
    #         st= time.time()
    #         losses= []
            
    #         for enc_seq, teach_force_seq, y in dataset:
              
    #             per_replica_loss, _= self.strategy.run(self.train_step(enc_seq, teach_force_seq, y),
    #                                            args= (enc_seq, teach_force_seq, y))
                
    #             total_loss= 
                
    #             losses.append(loss.numpy())
            
    #         avg_loss.append(np.mean(losses))
        
    #         if save_evry_ckpt:
    #             self.save_checkpoints(self.config['transformer']['ckpt_dir'])
            
    #         print(f"EPOCH: {epoch} ::: LOSS: {loss} ::: TIME TAKEN: {time.time()-st}")
        
    #     if save_model: self.save_checkpoints(self.config['transformer']['ckpt_dir'])
        
    #     print('YAYY MODEL IS TRAINED')
    #     print(f'TOTAL TIME TAKEN- {time.time() - start}')
        
    
    def save_checkpoints(ckpt_dir= 'weights/transformer_ckpts'):
        checkpoint_dir = ckpt_dir
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        ckpt= tf.train.Checkpoint(optimizer= self.optimizer,
                                  transformer= self.transformer)
        ckpt.save(file_prefix= checkpoint_prefix)
            
    def restore_checkpoint(ckpt_dir= 'weights/transformer_ckpts'):
        checkpoint_dir = ckpt_dir
        ckpt= tf.train.Checkpoint(optimizer= self.optimizer,
                                  transformer= self.transformer)
        ckpt.restore(tf.train.latest_checkpoint(checkpoint_dir))

