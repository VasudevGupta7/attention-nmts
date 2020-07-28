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
        pass
        
    def on_epoch_begin(self, epoch):
        
        self.st_epoch= time.time()
        self.tr_loss= []
        self.val_loss= []
        logger.info(f'epoch-{epoch} started')
        
    def on_batch_end(self, batch, tr_loss, val_loss):
        
        self.tr_loss.append(tr_loss.numpy())
        self.val_loss.append(val_loss.numpy())
        
        # logging per step
        step_metrics= {
                    'step': batch,
                    "step_tr_crossentropy_loss": tr_loss_.numpy(),
                    'step_val_crossentropy_loss': val_loss.numpy()
                }
        wandb.log(step_metrics)
        print(f"step-{batch} ===== {step_metrics}")
    
    def on_epoch_end(self, epoch):
        
        # logging per epoch
        epoch_metrics= {
                'epoch': epoch,
                "epoch_tr_crossentropy_loss": np.mean(self.tr_lss),
                'epoch_val_crossentropy_loss': np.mean(self.val_lss)
            }
        wandb.log(epoch_metrics, commit= False)
        print(f"EPOCH-{epoch} ===== TIME TAKEN-{np.around(time.time() - self.st_epoch, 2)}sec ===== {epoch_metrics}")
        
    