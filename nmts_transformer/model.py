"""
Transformer for building NMTS

LAYERS AVAILABLE IN THIS FILE
    - INPUT EMBEDDING
        - NORMAL EMBEDDING
        - POSITION EMBEDDING
    - FEED FORWARD LAYER
    - MULTIHEAD ATTENTION
        - SCALED_DOT_PRODUCT_ATTENTION
    - ENCODER LAYER
    - DECODER LAYER

@author: vasudevgupta
"""

import tensorflow as tf
import numpy as np

import os
os.chdir('/Users/vasudevgupta/Desktop/GitHub/seq2seq/nmts_transformer')

class LearningRate(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    SCHEDULING THE LEARNING RATE AS PER GIVEN IN PAPER
    """
    def __init__(self, dmodel, warmup_steps= 4000):
        super(LearningRate, self).__init__()
        self.dmodel= dmodel
        self.warmup_steps= warmup_steps
        
    def __call__(self, step_num):
        arg1= 1/tf.math.sqrt(tf.cast(self.step_num, tf.float32))
        arg2= step_num*tf.math.pow(self.warmup_steps, -1.5)
        return (1/ tf.math.sqrt(tf.cast(self.dmodel, tf.float32)))*tf.minimum(arg1, arg2)
    
    
class params:
    pass

params.warmup_steps= 4000
params.num_blocks= 4
params.dmodel= 256
params.num_heads= 4
params.depth= params.dmodel/params.num_heads
params.vocab_size= 10000
learning_rate= LearningRate(params.dmodel, params.warmup_steps)
params.optimizer= tf.keras.optimizers.Adam(0.001, clipvalue= 1.0)

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
        
# angle= lambda posn, i: posn*(1/ tf.math.pow(10000, (2*i)/ dmodel))
# i= np.arange(dmodel)[0::2]
# posn= np.arange(l)
# emb_seq[:, 0::2]= [angle(j, i) for j in posn]

# imp= InputEmbedding(256, 1000)
# emb_seq= np.ones((6, 7))
# emb_seq= tf.convert_to_tensor(emb_seq, dtype= tf.float32)
# imp(emb_seq)
# imp.position_embed(emb_seq)

class MultiheadAttention(tf.keras.layers.Layer):
    
    def __init__(self, dmodel, num_heads):
        super(MultiheadAttention, self).__init__()
        self.dmodel= dmodel
        self.depth= dmodel / num_heads # 64
        self.linear= tf.keras.layers.Dense(self.dmodel)
        
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

    def call(self, Q, K, V, mask= None):
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

# x= tf.ones((32, 4, 10, 64))
# ml= MultiheadAttention(256, 4)
# seq= ml(x,x,x)

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
        x= self.dropout= self.dropout(x)
        x= self.relu(x)
        # x- (batch_size, seqlen, 2048)
        x= self.linear2(x)
        # x- (batch_szie, seqlen, dmodel)
        return x

# ffn= FeedForwardLayer(256)       
# seq= np.ones((32, 7, 256))
# seq= tf.convert_to_tensor(seq, dtype= tf.float32)

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
  
    def multi_head_split(self, x):
        # x- (batch_size, seq, dmodel)
        self.l1= tf.keras.layers.Dense(self.dmodel, name= "Query_dense") # Q
        self.l2= tf.keras.layers.Dense(self.dmodel, name= "keys_dense") # K
        self.l3= tf.keras.layers.Dense(self.dmodel, name= "value_dense") # V
        
        Q= self.l1(x)
        K= self.l2(x)
        V= self.l3(x)
        # Q, K, V- (batch_size, seqlen, num_heads*depth)
        
        (a,b,c)= Q.shape
        Q= tf.reshape(Q, (a, b, self.num_heads, self.depth))
        K= tf.reshape(K, (a, b, self.num_heads, self.depth))
        V= tf.reshape(V, (a, b, self.num_heads, self.depth))
        
        # Q, K, V- (batch_size, seqlen, num_heads, depth)
        Q= tf.transpose(Q, perm= (0, 2, 1, 3))
        K= tf.transpose(K, perm= (0, 2, 1, 3))
        V= tf.transpose(V, perm= (0, 2, 1, 3))
        # Q, K, V- (batch_size, num_heads, seqlen, depth)
        return Q, K, V

    def call(self, x, padding_mask= None):
        # x- (batch_size, seqlen, dmodel)
        Q, K, V= self.multi_head_split(x)
        # Q, K, V- (batch_size, num_heads, seqlen, depth)
        inp1= self.self_attention(Q, K, V, padding_mask)
        # inp1- (batch_size, seqlen, dmodel)
        x= self.layer_norm1(x+inp1)
        # x- (batch_size, seq, dmodel)
        inp2= self.ff_layer(x)
        # # inp2- (batch_szie, seqlen, dmodel)
        out= self.layer_norm2(x+inp2)
        # out- (batch_size, seq, dmodel)
        return out
    
# el= EncoderLayer(4, 64, 256)
# x= tf.ones((32, 10, 256))
# out= el(x)

# x= tf.ones((32, 10, 64))
# Q, K, V= el.multi_head_split(x)

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
        
# enc= Encoder(4, 256, 64, 4,1000)
# x= tf.ones((32, 10))
# out= enc(x)
   
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
        Qtar, Ktar, Vtar= self.multi_head_split(dec_input)
        # Qtar, Ktar, Vtar- (batch_size, num_heads, tar_seqlen, depth)
        x= self.self_attention(Qtar, Ktar, Vtar, mask= seq_mask)
        # x- (batch_size, tar_seqlen, dmodel)
        x= self.layer_norm1(dec_input+x)
        # x- (batch_size, tar_seqlen, dmodel)
        
        Qdec, _, _= self.multi_head_split(x)
        # Qdec- (batch_size, num_heads, tar_seqlen, depth)
        _, Kenc, Venc= self.multi_head_split(dec_input)
        # Kdec, Vdec- (batch_size, num_heads, Kseqlen, depth)
        multi_head_out= self.enc_dec_attention(Qdec, Kenc, Venc, mask= padding_mask)
        # multi_head_out- (batch_size, tar_seqlen, dmodel)
        x= self.layer_norm2(x+multi_head_out)
        # x- (batch_size, tar_seqlen, dmodel)
        ffn= self.ff_layer(x)
        # ffn- (batch_size, tar_seqlen, dmodel)
        x= self.layer_norm3(x+ffn)
        # x- (batch_size, tar_seqlen, dmodel)
        return x
    
    def multi_head_split(self, x):
        # x- (batch_size, seq, dmodel)
        self.l1= tf.keras.layers.Dense(self.dmodel) # Q
        self.l2= tf.keras.layers.Dense(self.dmodel) # K
        self.l3= tf.keras.layers.Dense(self.dmodel) # V
        
        Q= self.l1(x)
        K= self.l2(x)
        V= self.l3(x)
        # Q, K, V- (batch_size, seqlen, num_heads*depth)
        
        (a,b,c)= Q.shape
        Q= tf.reshape(Q, (a, b, self.num_heads, self.depth))
        K= tf.reshape(K, (a, b, self.num_heads, self.depth))
        V= tf.reshape(V, (a, b, self.num_heads, self.depth)) 
        # Q, K, V- (batch_size, seqlen, num_heads, depth)
       
        Q= tf.transpose(Q, perm= (0, 2, 1, 3))
        K= tf.transpose(K, perm= (0, 2, 1, 3))
        V= tf.transpose(V, perm= (0, 2, 1, 3))
        # Q, K, V- (batch_size, num_heads, seqlen, depth)
        return Q, K, V

# dec= DecoderLayer(256, 4, 64)
# x= tf.ones((32, 10, 256))
# out= dec(x, x)

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
    
# dec= Decoder(4, 256, 1000)
# x= tf.ones((32, 10))
# out= dec(x, tf.ones((32, 10, 256)))
        
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

# tr= Transformer(4, 256, 64, 4, 10000, 10000)
# x= tf.ones((32, 10))
# y= tf.ones((32, 4))
# out= tr(x, y)

# y= tf.ones((32, 4))
# ypred= tf.ones((32, 4, 10000))
sce= tf.keras.losses.SparseCategoricalCrossentropy(from_logits= True, reduction= 'none')
def loss_fn(y, ypred, sce= sce):
    loss_= sce(y, ypred)
    # loss_- (batch_size, seqlen)
    mask= tf.cast(tf.math.not_equal(y, 0), tf.float32)
    # mask- (batch_size, seqlen)
    loss_ = mask*loss_
    return tf.reduce_mean(tf.reduce_mean(loss_, axis= 1))

transformer= Transformer(num_blocks= 1, dmodel= 256, depth= 64,
                         num_heads= 4, inp_vocab_size= 1000, tar_vocab_size= 1000)

# @tf.function(input_signature= [
#     tf.TensorSpec(shape= (None, None), dtype= tf.int64),
#     tf.TensorSpec(shape= (None, None), dtype= tf.int64), 
#     tf.TensorSpec(shape= (None, None), dtype= tf.int64)
#                   )]
def train_step(enc_input, dec_input, dec_output):       
    with tf.GradientTape() as gtape:
        ypred= transformer(enc_input, dec_input, enc_padding_mask= None, dec_padding_mask= None, dec_seq_mask= None)
        loss= loss_fn(y= dec_output, ypred= ypred, sce= sce)
    grads= gtape.gradient(loss, transformer.trainable_variables)
    params.optimizer.apply_gradients(zip(grads, transformer.trainable_variables))
    return grads, loss

x= tf.ones((32, 4))*0.001
enc_input= x
dec_input= x
dec_output= x
out, ls= train_step(x, x, x)
        
        
        
        
        
        
        
        
        
        
        

























