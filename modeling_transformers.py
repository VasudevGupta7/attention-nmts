"""Transformer

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
    - CREATE PADDING MASK
    - UNIDIRECTIONAL INPUT MASK

@author: vasudevgupta
"""

import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import numpy as np

from dataloader import Dataloader

import logging

logger= logging.getLogger(__name__)

class InputEmbedding(tf.keras.layers.Layer):
    """
    EMBED_WORDS- (NUM_WORDS, EMBED_DIMS)
    """
    def __init__(self, dmodel, vocab_size, embed_id= None, learn_pos_embed= False, name= 'Combined-Embedding-Layer'):
        super(InputEmbedding, self).__init__(name= name)
        
        self.learn_pos_embed= learn_pos_embed
        self.dmodel= dmodel
        self.vocab_size= vocab_size
        
        if embed_id == None:
            self.embed= tf.keras.layers.Embedding(input_dim= vocab_size, output_dim= dmodel, name= "Normal_Embedding")
        else:
            self.embed = tf.keras.layers.Lambda(lambda input_ids: self.normal_embedding(embed_id, input_ids))
            
        if not self.learn_pos_embed:
            self.position_embed= tf.keras.layers.Lambda(lambda emb_seq: self.position_embedding(emb_seq), name= "Position_Embedding")
            self.position_embed.trainable= False
        else:
            raise ValueError('Currently not supported')
        
    # def normal_embedding(self, embed_id, input_ids):
        
    #     dataloader= Dataloader(embed_id)
    #     embed_weights= dataloader.get_weights()
        
        
        
    
    def position_embedding(self, emb_seq):
        # emb_seq -> (batch_size, seqlen, emb_dims)
        
        pos_emb_seq= np.empty(shape= (emb_seq.shape))
        l= pos_emb_seq.shape[-2]
        angle= lambda posn, i: posn*(1/ np.power(10000, (2*i)/ self.dmodel))
        
        pos_emb_seq[:, :, 0::2]= [np.sin(angle(j, np.arange(self.dmodel)[0::2])) for j in np.arange(l)]
        pos_emb_seq[:, :, 1::2]= [np.cos(angle(j, np.arange(self.dmodel)[1::2])) for j in np.arange(l)]
        
        # pos_emb_seq -> (batch_size, seqlen, emb_dims)
        return tf.convert_to_tensor(pos_emb_seq, dtype= tf.float32)
        
    def call(self, seq):
        # seq -> (batch_size, seqlen)
        
        emb_seq= self.embed(seq)
        # emb_seq -> (batch_size, seqlen, dmodel)
        
        position_emb= self.position_embed(emb_seq)
        # position_seq -> (batch_size, seqlen, dmodel)
        
        model_emb_seq= emb_seq + position_emb
        # model_emb_seq -> (batch_size, seqlen, dmodel)
        
        return model_emb_seq        
    
    def get_config(self):
        config= super(InputEmbedding, self).get_config()
        config.update({
            'learn_pos_embed': self.learn_pos_embed,
            'dmodel': self.dmodel,
            'vocab_size': self.vocab_size
            })
        return config

class MultiheadAttention(tf.keras.layers.Layer):
    
    def __init__(self, dmodel, num_heads, name= 'MultiHead-Attention-Layer'):
        super(MultiheadAttention, self).__init__(name= name)
        
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
       
        # Q, K, V -> (batch_size, num_heads, seqlen, depth)
        
        scores= tf.matmul(Q, K, transpose_b= True)
        # scores -> (batch_size, num_heads, Qseqlen, Kseq_len); axis= -1 apply softmax
        
        scaled_scores= scores/ tf.math.sqrt(tf.cast(self.depth, tf.float32))
        # scaled_scores -> (batch_size, num_heads, Qseqlen, Kseq_len)
        
        if mask is not None:
            scaled_scores += mask*(-1e9)
        # scaled_scores -> (batch_size, num_heads, Qseqlen, Kseq_len)
        
        attention_weights= tf.nn.softmax(scaled_scores, axis= -1)
        # attention_weights -> (batch_size, num_heads, Qseqlen, Kseqlen)
        
        values_weighted_sum= tf.matmul(attention_weights, V)
        # values_weighted_sum -> (batch_size, num_heads, Qseqlen, depth)
        
        return values_weighted_sum, attention_weights
        
    def multi_head_split(self, x):
        # x -> (batch_size, seqlen, dmodel)
        
        (a,b,c)= x.shape
        x= tf.reshape(x, (a, b, self.num_heads, self.depth))
        # x -> (batch_size, seqlen, num_heads, depth)
        
        x= tf.transpose(x, perm= (0, 2, 1, 3))
        # x -> (batch_size, num_heads, seqlen, depth)
        
        return x

    def call(self, Q, K, V, mask= None):
        
        Q= self.l1(Q)
        K= self.l2(K)
        V= self.l3(V)
        # Q, K, V -> (batch_size, seqlen, dmodel)
        
        Q= self.multi_head_split(Q)
        K= self.multi_head_split(K)
        V= self.multi_head_split(V)
        # Q, K, V -> (batch_size, num_heads, seqlen, depth)
       
        values_weighted_sum, attention_weights= self.scaled_dot_product_attention(Q, K, V, mask)
        # values_weighted_sum -> (batch_size, num_heads, Qseqlen, depth)
       
        values_weighted_sum= tf.transpose(values_weighted_sum, perm= (0, 2, 1, 3))
        # values_weighted_sum -> (batch_size, Qseqlen, num_heads, depth)
       
        (a,b,c,d)= values_weighted_sum.shape ## lets concatenate all heads
        concat_multiheads= tf.reshape(values_weighted_sum, (a, b, c*d))
        # concat_multiheads -> (batch_size, Qseqlen, num_heads*depth)
        
        seq= self.linear(concat_multiheads)
        # seq -> (batch_size, Qseqlen, dmodel)
        
        return seq
    
    def get_config(self):
        config= super(MultiheadAttention, self).get_config()
        config.update({
            'num_heads': self.num_heads,
            'dmodel': self.dmodel,
            'depth': self.depth
            })
        return config

class FeedForwardLayer(tf.keras.layers.Layer):
    
    def __init__(self, dmodel, name= 'Feed-forward-layer'):
        super(FeedForwardLayer, self).__init__(name= name)
        
        self.linear1= tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(2048,
                                                     kernel_initializer= 'he_normal'))
        self.dropout= tf.keras.layers.Dropout(0.2)
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

    def __init__(self, num_heads, depth, dmodel, name= 'encoder-layer'):
        super(EncoderLayer, self).__init__(name= name)
        
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
    
    def get_config(self):
        config= super(EncoderLayer, self).get_config()
        config.update({
            'num_heads': self.num_heads,
            'dmodel': self.dmodel,
            'depth': self.depth
            })
        return config

class Encoder(tf.keras.layers.Layer):
    
    def __init__(self, num_blocks, dmodel, depth, num_heads, inp_vocab_size, name= 'encoder'):
        super(Encoder, self).__init__(name= name)
        
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

class DecoderLayer(tf.keras.layers.Layer):
    
    def __init__(self, dmodel, num_heads, depth, name= 'decoder-layer'):
        super(DecoderLayer, self).__init__(name= name)
        
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
    
    def get_config(self):
        config= super(DecoderLayer, self).get_config()
        config.update({
            'num_heads': self.num_heads,
            'dmodel': self.dmodel,
            'depth': self.depth
            })
        return config

class Decoder(tf.keras.layers.Layer):
    
    def __init__(self, num_blocks, dmodel, depth, num_heads, tar_vocab_size, name= 'decoder'):
        super(Decoder, self).__init__(name= name)
        
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
    
    def __init__(self, num_blocks, dmodel, num_heads, inp_vocab_size, tar_vocab_size):
        super(Transformer, self).__init__()
        
        self.dmodel= dmodel
        self.depth= dmodel/ num_heads
        self.num_heads= num_heads
        
        self.encoder= Encoder(num_blocks, dmodel, self.depth, num_heads, inp_vocab_size)
        self.decoder= Decoder(num_blocks, dmodel, self.depth, num_heads, tar_vocab_size)
        self.linear= tf.keras.layers.Dense(tar_vocab_size, dtype= mixed_precision.Policy('float32'))
        
    def call(self, enc_input, dec_input, enc_padding_mask= None, enc_dec_padding_mask= None, dec_seq_mask= None):
        
        enc_output= self.encoder(enc_input, mask= enc_padding_mask)
        # x- (batch_size, enc_seqlen, dmodel)
        
        x= self.decoder(dec_input, enc_output, padding_mask= enc_dec_padding_mask, seq_mask= dec_seq_mask)
        # x- (batch_size, tar_seqlen, dmodel)
        
        x= self.linear(x)
        # x- (batch_size, tar_seqlen, tar_vocab_size)
        
        return x
    
    def get_config(self):
        config= super(Transformer, self).get_config()
        config.update({
            'num_heads': self.num_heads,
            'dmodel': self.dmodel,
            'depth': self.depth
            })
        return config

def create_padding_mask(kseq):
    # (batch_size, key_seqlen)
    
    mat= tf.cast(tf.math.equal(kseq, 0), tf.float32)
    
    return mat[:, tf.newaxis, tf.newaxis, :] #(batch_size, 1, 1, key_seqlen)

def unidirectional_input_mask(enc_input, dec_input):
    
    matrix= tf.ones((dec_input.shape[-1], enc_input.shape[-1]))
    
    lower_triang_mat= tf.linalg.band_part(matrix, -1, 0)
    # (dec_seqlen, enc_seqlen)
    
    return tf.cast(tf.math.equal(lower_triang_mat, 0), tf.float32)
