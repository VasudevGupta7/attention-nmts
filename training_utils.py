"""train NMTS on single gpu/cpu based machine

@author: vasudevgupta
"""
import argparse
import logging

import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision

from callbacks import CustomCallback

logger= logging.getLogger(__name__)

class Trainer(object):
    
    def __init__(self, 
                 ckpt_dir, 
                 precision_policy= 'float32',
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
              strategy,
              epochs= 2, 
              restore_ckpt= False, 
              save_final_ckpt= False, 
              save_evry_ckpt= False):
        
        # enc_input, dec_input, dec_output === tr_dataset
        if restore_ckpt: self.restore(self.ckpt_dir, assert_consumed=True)
        
        for epoch in range(1, 1+epochs):
            self.callbacks.on_epoch_begin(epoch)
            
            for enc_in, dec_in, dec_out in tr_dataset:
              
                tr_loss= self.distributed_train_step(enc_in, dec_in, dec_out, strategy)
                val_loss= evaluate(val_dataset)
                
                step_metrics= self.callbacks.on_batch_end(tr_loss, val_loss)
            
            if save_evry_ckpt: self.manager.save()
            
            epoch_metrics= self.callbacks.on_epoch_end(epoch)
        
        if save_final_ckpt: self.manager.save()
        
        return epoch_metrics
    
    def evaluate(self, val_dataset):
        
        loss_= 0
        steps= 0
        
        for enc_in, dec_in, dec_out in val_dataset:
            
            loss= self.distributed_test_step(enc_in, dec_in, dec_out, strategy)
            loss_ += loss
            steps += 1
            
        return loss_/steps
    
    # @tf.function
    def distributed_train_step(self, enc_in, dec_in, dec_out, strategy):
        
        per_replica_loss, _= strategy.run(self.train_step,
                                          args= (enc_in, dec_in, dec_out))
        
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_loss, axis= None)
    
    # @tf.function
    def distributed_test_step(self, enc_in, dec_in, dec_out, strategy):
        
        per_replica_loss, _= strategy.run(self.test_step,
                                          args= (enc_in, dec_in, dec_out))
        
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_loss, axis= None)
    
def get_args():
    
    parser= argparse.ArgumentParser(description='RUN THIS FILE FOR TRAINING THIS MODEL')
    
    parser.add_argument('--model_type', type= str, help= 'rnn_attention or transformer')
    
    parser.add_argument('--config', type= str, default= 'config.yaml', help= 'config file for the model')
    parser.add_argument('--save_model', action= 'store_true', default= False, help= 'if specified, model will be saved')
    parser.add_argument('--save_evry_ckpt', action= 'store_true', default= False, help= 'if specified, every epoch will be saved')
    parser.add_argument('--load_model', action= 'store_true', default= False, help= 'if specifiled, weights will be restored before training')
    parser.add_argument('--dataset', type= str, default= 'data/eng2ger.csv', help= 'file name of dataset')
    
    parser.add_argument('--distributed', action= store_true, help= 'if you want to do distributed training')
    parser.add_argument('--num_gpu', type= int, default= 1, help= 'Num of replicas to involve in training')
    
    return parser.parse_args()


