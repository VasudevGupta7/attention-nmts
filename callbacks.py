"""Callbacks for model

@author: vasudevgupta
"""
import numpy as np
import tensorflow as tf
import time
import wandb
import logging

logger= logging.getLogger(__name__)

class CustomCallback(object):
    
    def __init__(self):
        
        self.step = 0
        
    def on_epoch_begin(self, epoch):
        
        self.st_epoch= time.time()
        self.tr_loss= []
        self.val_loss= []
        logger.info(f'epoch-{epoch} started')
        
    def on_batch_end(self, tr_loss, val_loss):
        
        self.step += 1
        self.tr_loss.append(tr_loss.numpy())
        self.val_loss.append(val_loss.numpy())
        
        # logging per step
        step_metrics= {
                    'step': self.step,
                    "step_tr_crossentropy_loss": tr_loss_.numpy(),
                    'step_val_crossentropy_loss': val_loss.numpy()
                }
        wandb.log(step_metrics)
        
        print(f"step-{self.step} ===== {step_metrics}")
        
        return step_metrics
    
    def on_epoch_end(self, epoch):
        
        # logging per epoch
        epoch_metrics= {
                'epoch': epoch,
                "epoch_tr_crossentropy_loss": np.mean(self.tr_loss),
                'epoch_val_crossentropy_loss': np.mean(self.val_loss)
            }
        wandb.log(epoch_metrics, commit= False)
        
        print(f"EPOCH-{epoch} ===== TIME TAKEN-{np.around(time.time() - self.st_epoch, 2)}sec ===== {epoch_metrics}")
        
        return epoch_metrics
    