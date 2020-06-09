"""
LETS SEE MY NMTS USING TRANSFORMER

@author: vasudevgupta
"""
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import time

import os
os.chdir('/Users/vasudevgupta/Desktop/GitHub/seq2seq/nmts_transformer')

from model import Transformer, loss_fn, train_step, save_checkpoints, restore_checkpoint
from params import params, LearningRate

sce= tf.keras.losses.SparseCategoricalCrossentropy(from_logits= True, reduction= 'none')
transformer= Transformer(num_blocks= params.num_blocks, dmodel= params.dmodel, 
                         depth= params.depth, num_heads= params.num_heads,
                         inp_vocab_size= params.inp_vocab_size, tar_vocab_size= params.tar_vocab_size)

def train(enc_input, dec_input, dec_output, params= params, sce= sce, transformer= transformer):
    avg_loss= []
    start= time.time()
    for epoch in (range(1, 1+params.epochs)):
        st= time.time()
        grads, loss= train_step(enc_input, dec_input, dec_output,
               transformer= transformer, sce= sce)
        avg_loss.append(loss)
        if epoch%10 == 0:
            save_checkpoints(params, transformer)
        print(f"EPOCH: {epoch} ::: LOSS: {loss} ::: TIME TAKEN: {time.time()-st}")
    save_checkpoints(params, transformer)
    print('YAYY MODEL IS TRAINED')
    print(f'TOTAL TIME TAKEN- {time.time() - start}')
    return grads, avg_loss
    
# restore_checkpoint(params, transformer)
grads, avg_loss= train(enc_input, dec_input, dec_output)
