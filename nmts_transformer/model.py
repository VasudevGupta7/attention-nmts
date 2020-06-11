"""
Transformer for building NMTS

CLASSES/ FUNCTION AVAILABLE IN THIS FILE
    - INPUT EMBEDDING
        - NORMAL EMBEDDING
        - POSITION EMBEDDING
    - FEED FORWARD LAYER
    - MULTIHEAD ATTENTION
        - SCALED_DOT_PRODUCT_ATTENTION
    - ENCODER LAYER
    - DECODER LAYER
    - DECODER
    - ENCODER
    - TRANSFORMER
    - LOSS_FN
    - TRAIN_STEP
    - LEARNING_RATE

@author: vasudevgupta
"""

import tensorflow as tf
import numpy as np

import os
os.chdir('/Users/vasudevgupta/Desktop/GitHub/seq2seq/nmts_transformer')

from params import params

class InputEmbedding(tf.keras.layers.Layer):
    """
    EMBED_WORDS- (NUM_WORDS, EMBED_DIMS)
    """
    def __init__(self, dmodel, vocab_size):
        super(InputEmbedding, self).__init__()
        self.dmodel= dmodel
        self.embed= tf.keras.layers.Embedding(input_dim= vocab_size, output_dim= dmodel, name= "Normal_Embedding")
        self.position_embed= tf.keras.layers.Lambda(lambda emb_seq: self.position_embedding(emb_seq), name= "Position_Embedding")
        
    def position_embedding(self, emb_seq):
        # emb_seq- (batch_size, seqlen, emb_dims)
        emb_seq= emb_seq.numpy()
        l= emb_seq.shape[-2]
        angle= lambda posn, i: posn*(1/ tf.math.pow(10000, (2*i)/ self.dmodel))
        emb_seq[:, :, 0::2]= [np.sin(angle(j, np.arange(self.dmodel)[0::2])) for j in np.arange(l)]
        emb_seq[:, :, 1::2]= [np.cos(angle(j, np.arange(self.dmodel)[1::2])) for j in np.arange(l)]
        # emb_seq- (batch_size, seqlen, emb_dims)
        return tf.convert_to_tensor(emb_seq)
        
    def call(self, seq):
        # seq- (batch_size, seqlen)
        emb_seq= self.embed(seq)
        # emb_seq- (batch_size, seqlen, dmodel)
        position_emb= self.position_embed(emb_seq)
        # position_seq- (batch_size, seqlen, dmodel)
        model_emb_seq= emb_seq + position_emb
        # model_emb_seq- (batch_size, seqlen, dmodel)
        return model_emb_seq        

class MultiheadAttention(tf.keras.layers.Layer):
    
    def __init__(self, dmodel, num_heads):
        super(MultiheadAttention, self).__init__()
        self.dmodel= dmodel
        self.num_heads= num_heads
        self.depth= tf.cast(dmodel / num_heads, tf.int32) # 64
        self.linear= tf.keras.layers.Dense(self.dmodel)
        
        self.l1= tf.keras.layers.Dense(self.dmodel, name= "Query_dense") # Q
        self.l2= tf.keras.layers.Dense(self.dmodel, name= "keys_dense") # K
        self.l3= tf.keras.layers.Dense(self.dmodel, name= "value_dense") # V
        
    def scaled_dot_product_attention(self, Q, K, V, mask= None):
        """
        Q, K, V- contains query, keys, values respectively of all the words
        mask- it is having 1 for the elements which we want to exclude
            either mask for avoiding padding; dims- (seqlen, 1)
            or look up ahead mask; dims= (seqlen, seqlen)
        """
        # Q, K, V- (batch_size, num_heads, seqlen, depth)
        scores= tf.matmul(Q, K, transpose_b= True)
        # scores- (batch_size, num_heads, Qseqlen, Kseq_len); axis= -1 apply softmax
        scaled_scores= scores/ tf.math.sqrt(tf.cast(self.depth, tf.float32))
        if mask is not None:
            scaled_scores += mask*(-1e9)
        # scaled_scores- (batch_size, num_heads, Qseqlen, Kseq_len)
        attention_weights= tf.nn.softmax(scaled_scores, axis= -1)
        # attention_weights- (batch_size, num_heads, Qseqlen, Kseqlen)
        values_weighted_sum= tf.matmul(attention_weights, V)
        # values_weighted_sum- (batch_size, num_heads, Qseqlen, depth)
        return values_weighted_sum, attention_weights
        
    def multi_head_split(self, x):
        # x- (batch_size, seqlen, dmodel)
        
        (a,b,c)= x.shape
        x= tf.reshape(x, (a, b, self.num_heads, self.depth))
        # x- (batch_size, seqlen, num_heads, depth)
        
        x= tf.transpose(x, perm= (0, 2, 1, 3))
        # x- (batch_size, num_heads, seqlen, depth)
        return x

    def call(self, Q, K, V, mask= None):
        Q= self.l1(Q)
        K= self.l2(K)
        V= self.l3(V)
        # Q, K, V- (batch_size, seqlen, dmodel)
        
        Q= self.multi_head_split(Q)
        K= self.multi_head_split(K)
        V= self.multi_head_split(V)
        # Q, K, V- (batch_size, num_heads, seqlen, depth)
       
        values_weighted_sum, attention_weights= self.scaled_dot_product_attention(Q, K, V, mask)
        # values_weighted_sum- (batch_size, num_heads, Qseqlen, depth)
       
        values_weighted_sum= tf.transpose(values_weighted_sum, perm= (0, 2, 1, 3))
        # values_weighted_sum- (batch_size, Qseqlen, num_heads, depth)
       
        (a,b,c,d)= values_weighted_sum.shape ## lets concatenate all heads
        concat_multiheads= tf.reshape(values_weighted_sum, (a, b, c*d))
        # (batch_size, Qseqlen, num_heads*depth)
        seq= self.linear(concat_multiheads)
        # seq- (batch_size, Qseqlen, dmodel)
        return seq

class FeedForwardLayer(tf.keras.layers.Layer):
    
    def __init__(self, dmodel):
        super(FeedForwardLayer, self).__init__()
        self.linear1= tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(2048,
                                                     kernel_initializer= 'he_normal'))
        self.dropout= tf.keras.layers.Dropout(0.6)
        self.relu= tf.keras.layers.ReLU()
        self.linear2= tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(dmodel))

    def call(self, x):
        # x- (batch_szie, seqlen, dmodel)  
        x= self.linear1(x)
        x= self.dropout(x)
        x= self.relu(x)
        # x- (batch_size, seqlen, 2048)
        x= self.linear2(x)
        # x- (batch_szie, seqlen, dmodel)
        return x

class EncoderLayer(tf.keras.layers.Layer):

    def __init__(self, num_heads, depth, dmodel):
        super(EncoderLayer, self).__init__()
        self.num_heads= num_heads
        self.depth= depth
        self.dmodel= dmodel
        
        self.self_attention= MultiheadAttention(dmodel, num_heads)
        self.layer_norm1= tf.keras.layers.LayerNormalization()
        self.ff_layer= FeedForwardLayer(dmodel)
        self.layer_norm2= tf.keras.layers.LayerNormalization()

    def call(self, x, padding_mask= None):
        # x- (batch_size, seqlen, dmodel)
        inp1= self.self_attention(x, x, x, padding_mask)
        # inp1- (batch_size, seqlen, dmodel)
        x= self.layer_norm1(x+inp1)
        # x- (batch_size, seq, dmodel)
        inp2= self.ff_layer(x)
        # # inp2- (batch_szie, seqlen, dmodel)
        out= self.layer_norm2(x+inp2)
        # out- (batch_size, seq, dmodel)
        return out

class Encoder(tf.keras.layers.Layer):
    
    def __init__(self, num_blocks, dmodel, depth, num_heads, inp_vocab_size):
        super(Encoder, self).__init__()
        self.inp_embed= InputEmbedding(dmodel, inp_vocab_size)
        self.enc_blocks= [EncoderLayer(num_heads, depth, dmodel)
                          for i in range(num_blocks)]
        
    def call(self, x, mask= None):
        # x- (batch_size, seqlen)
        x= self.inp_embed(x)
        # x- (batch_size, seqlen, dmodel)
        for enc_block in self.enc_blocks:
            x= enc_block(x, padding_mask= mask)
        # x- (batch_size, seqlen, dmodel)
        return x
   
"""
MASKING IS LEFT IN ENCODER, DECODER
"""     

class DecoderLayer(tf.keras.layers.Layer):
    
    def __init__(self, dmodel, num_heads, depth):
        super(DecoderLayer, self).__init__()
        self.dmodel= dmodel
        self.num_heads= num_heads
        self.depth= depth
        
        self.self_attention= MultiheadAttention(dmodel, num_heads)
        self.layer_norm1= tf.keras.layers.LayerNormalization()
        self.enc_dec_attention= MultiheadAttention(dmodel, num_heads)
        self.layer_norm2= tf.keras.layers.LayerNormalization()
        self.ff_layer= FeedForwardLayer(dmodel)
        self.layer_norm3= tf.keras.layers.LayerNormalization()
        
    def call(self, dec_input, enc_output, padding_mask= None, seq_mask= None):
        # dec_input- (batch_size, tar_seqlen, dmodel)
        x= self.self_attention(dec_input, dec_input, dec_input, mask= seq_mask)
        # x- (batch_size, tar_seqlen, dmodel)
        x= self.layer_norm1(dec_input+x)
        # x- (batch_size, tar_seqlen, dmodel)
        m= x
        multi_head_out= self.enc_dec_attention(dec_input, enc_output, enc_output, mask= padding_mask)
        # multi_head_out- (batch_size, tar_seqlen, dmodel)
        x= self.layer_norm2(x+multi_head_out)
        # x- (batch_size, tar_seqlen, dmodel)
        
        ffn= self.ff_layer(x)
        # ffn- (batch_size, tar_seqlen, dmodel)
        x= self.layer_norm3(x+ffn)
        # x- (batch_size, tar_seqlen, dmodel)
        return x

class Decoder(tf.keras.layers.Layer):
    
    def __init__(self, num_blocks, dmodel, depth, num_heads, tar_vocab_size):
        super(Decoder, self).__init__()
        self.dec_embed= InputEmbedding(dmodel, tar_vocab_size)
        self.dec_blocks= [DecoderLayer(dmodel, num_heads, depth) for i in range(num_blocks)]
        
    def call(self, dec_input, enc_output, padding_mask= None, seq_mask= None):
        # dec_input- (batch_size, tar_seqlen)
        tar_input= self.dec_embed(dec_input)
        # tar_input- (batch_size, tar_seqlen, dmodel)
        for dec_block in self.dec_blocks:
            tar_input= dec_block(tar_input, enc_output, padding_mask= None, seq_mask= None)
        # tar_input- (batch_size, tar_seqlen, dmodel)
        return tar_input
    
class Transformer(tf.keras.Model):
    
    def __init__(self, num_blocks, dmodel, depth, num_heads, inp_vocab_size, tar_vocab_size):
        super(Transformer, self).__init__()
        self.encoder= Encoder(num_blocks, dmodel, depth, num_heads, inp_vocab_size)
        self.decoder= Decoder(num_blocks, dmodel, depth, num_heads, tar_vocab_size)
        self.linear= tf.keras.layers.Dense(tar_vocab_size)
        
    def call(self, enc_input, dec_input, enc_padding_mask= None, dec_padding_mask= None, dec_seq_mask= None):
        enc_output= self.encoder(enc_input, mask= enc_padding_mask)
        # x- (batch_size, enc_seqlen, dmodel)
        x= self.decoder(dec_input, enc_output, padding_mask= dec_padding_mask, seq_mask= dec_seq_mask)
        # x- (batch_size, tar_seqlen, dmodel)
        x= self.linear(x)
        # x- (batch_size, tar_seqlen, tar_vocab_size)
        return x

def loss_fn(y, ypred, sce):
    loss_= sce(y, ypred)
    # loss_- (batch_size, seqlen)
    mask= tf.cast(tf.math.not_equal(y, 0), tf.float32)
    # mask- (batch_size, seqlen)
    loss_ = mask*loss_
    return tf.reduce_mean(tf.reduce_mean(loss_, axis= 1))

# @tf.function(input_signature= [
#     tf.TensorSpec(shape= (None, None), dtype= tf.int64),
#     tf.TensorSpec(shape= (None, None), dtype= tf.int64), 
#     tf.TensorSpec(shape= (None, None), dtype= tf.int64)
#                   )]
def train_step(enc_input, dec_input, dec_output,
               transformer, sce):       
    with tf.GradientTape() as gtape:
        ypred= transformer(enc_input, dec_input, enc_padding_mask= None, dec_padding_mask= None, dec_seq_mask= None)
        loss= loss_fn(dec_output, ypred, sce)
    grads= gtape.gradient(loss, transformer.trainable_variables)
    params.optimizer.apply_gradients(zip(grads, transformer.trainable_variables))
    return grads, loss.numpy()

def save_checkpoints(params, transformer):
    checkpoint_dir = 'weights'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    ckpt= tf.train.Checkpoint(optimizer= params.optimizer,
                              transformer= transformer)
    ckpt.save(file_prefix= checkpoint_prefix)
        
def restore_checkpoint(params, transformer):
    checkpoint_dir = 'weights'
    ckpt= tf.train.Checkpoint(optimizer= params.optimizer,
                              transformer= transformer)
    ckpt.restore(tf.train.latest_checkpoint(checkpoint_dir))

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

    def multisteps(self, enc_input, dec_input, topk_probs):
        """
        enc_input- (1, enc_seqlen)
        dec_input- (k, seqlen)
        topk_prob- (1, seqlen)
        """
        probs= tf.nn.softmax(self.model(enc_input, dec_input)[:, -1, :])
        # (k, seqlen, tar_vocab_size)
        marginal_probs= np.reshape(topk_probs, (self.k, 1))*probs
        reshaped_marg_probs= marginal_probs.numpy().reshape(1,-1)
        
        topk_probs= tf.math.top_k(reshaped_marg_probs, self.k)[0].numpy()
        
        topk_args= tf.math.top_k(reshaped_marg_probs, self.k)[1].numpy()
        topk_args= self.reindex(topk_args[0], params.tar_vocab_size)
        
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
            dec_input, topk_probs= self.multisteps(enc_input, dec_input, topk_probs)
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
                    a= topk_args[i]-params.tar_vocab_size
                    if a<0:
                        a= topk_args[i]
                        break
                    topk_args[i]= a
                ls.append(a)
        return ls