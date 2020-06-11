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