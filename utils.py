"""

@author: vasudevgupta
"""
import tensorflow as tf
import numpy as np

import os

os.chdir('/Users/vasudevgupta/Desktop/GitHub/seq2seq/nmts_transformer')
from params import params
from model import unidirectional_input_mask

def tokenizer(df_col, nlp_en= True):
    vocab= set()
    _= [[vocab.update([tok]) for tok in text.split(" ")] for text in df_col]
    ## need to append "<sos> " token " <eos>" depending on what is df_col
    if not nlp_en:
        vocab.update(["<sos>"])
        vocab.update(["<eos>"])
    # 0 is reserved for padding
    tokenize= dict(zip(vocab, range(1, 1+len(vocab))))
    detokenize= dict(zip(range(1, 1+len(vocab)), vocab))
    return tokenize, detokenize, len(vocab)

def padding(txt_toks, max_len):
    curr_ls= txt_toks.split(" ")
    len_ls= len(curr_ls)
    _= [curr_ls.append("<pad>") for i in range(max_len-len_ls) if len(curr_ls)<max_len]
    return " ".join(curr_ls)

def make_minibatches(df, col1= 'rev_eng_tok', col2= 'teach_force_tok', col3= 'target_tok'):
    enc_seq= np.array([df[col1].values[i] for i in range(len(df[col1]))])
    enc_seq= tf.data.Dataset.from_tensor_slices(enc_seq).batch(params.batch_size)

    teach_force_seq= np.array([df[col2].values[i] for i in range(len(df[col2]))])
    teach_force_seq= tf.data.Dataset.from_tensor_slices(teach_force_seq).batch(params.batch_size)

    y= np.array([df[col3].values[i] for i in range(len(df[col3]))])
    y= tf.data.Dataset.from_tensor_slices(y).batch(params.batch_size, drop_remainder= True)
    return enc_seq, teach_force_seq, y

# scheduling learning rate as per given in paper
class LearningRate(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    SCHEDULING THE LEARNING RATE AS PER GIVEN IN PAPER
    """
    def __init__(self, dmodel, warmup_steps= 4000):
        super(LearningRate, self).__init__()
        self.dmodel= dmodel
        self.warmup_steps= warmup_steps
        
    def __call__(self, step_num):
        arg1= 1/tf.math.sqrt(tf.cast(step_num, tf.float32))
        arg2= step_num*tf.math.pow(tf.cast(self.warmup_steps, tf.float32), -1.5)
        return (1/ tf.math.sqrt(tf.cast(self.dmodel, tf.float32)))*tf.minimum(arg1, arg2)
