"""Trainer class

@author: vasudevgupta
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision

import logging

from callbacks import CustomCallback
from modeling_transformers import create_padding_mask, unidirectional_input_mask
from utils import OPTM

logger= logging.getLogger(__name__)
        
class Trainer(object):
    
    def __init__(self, 
                 ckpt_dir, 
                 precision_policy= 'mixed_float16',
                 **kwargs):
        
        self.policy= mixed_precision.Policy(precision_policy)
        mixed_precision.set_policy(self.policy)
        
        self.ckpt_dir= ckpt_dir
        
        self.checkpoint= tf.train.Checkpoint(model1= kwargs.pop('model1', tf.Variable(0.)), 
                                             optimizer= kwargs.pop('optimizer', tf.Variable(0.)),
                                             model2= kwargs.pop('model2', tf.Variable(0.)))
        
        self.manager= tf.train.CheckpointManager(self.checkpoint, 
                                                 directory=self.ckpt_dir,
                                                 max_to_keep=kwargs.pop('max_to_keep', None),
                                                 keep_checkpoint_every_n_hours=kwargs.pop('keep_checkpoint_every_n_hours', None))
        
        self.callbacks= CustomCallback()
        
    def restore(self, ckpt, assert_consumed= False):
        # generally: self.manager.latest_checkpoint
        status= self.checkpoint.restore(ckpt)
        if assert_consumed:
            status.assert_consumed()
            logger.info('ckpt_restored')
        
    def train(self, 
              tr_dataset, 
              val_dataset, 
              epochs= 2, 
              restore_ckpt= False, 
              save_final_ckpt= False, 
              save_evry_ckpt= False):
        
        # enc_input, dec_input, dec_output === tr_dataset
        if restore_ckpt: self.restore(self.ckpt_dir, assert_consumed=True)
        
        for epoch in range(1, 1+epochs):
            self.callbacks.on_epoch_begin(epoch)
            
            for enc_in, dec_in, dec_out in tr_dataset:
              
                tr_loss= self.train_step(enc_in, dec_in, dec_out)
                val_loss= self.evaluate(val_dataset)
                
                step_metrics= self.callbacks.on_batch_end(tr_loss, val_loss)
            
            if save_evry_ckpt: self.manager.save()
            
            epoch_metrics= self.callbacks.on_epoch_end(epoch)
        
        if save_final_ckpt: self.manager.save()
        
        return epoch_metrics
    
    def evaluate(self, val_dataset):
        
        loss_= 0
        steps= 0
        
        for enc_in, dec_in, dec_out in val_dataset:
            
            loss= self.test_step(enc_in, dec_in, dec_out)
            loss_ += loss
            steps += 1
            
        return loss_/steps
    
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
    #                                             args= (enc_seq, teach_force_seq, y))
                
    #             total_loss= 1
                
    #             losses.append(loss.numpy())
            
    #         avg_loss.append(np.mean(losses))
        
    #         if save_evry_ckpt:
    #             self.save_checkpoints(self.config['transformer']['ckpt_dir'])
            
    #         print(f"EPOCH: {epoch} ::: LOSS: {loss} ::: TIME TAKEN: {time.time()-st}")
        
    #     if save_model: self.save_checkpoints(self.config['transformer']['ckpt_dir'])
        
    #     print('YAYY MODEL IS TRAINED')
    #     print(f'TOTAL TIME TAKEN- {time.time() - start}')
        

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
        
        self.trainable_vars= self.transformer.trainable_variables
        
        grads= gtape.gradient(loss, self.trainable_vars)
        
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

