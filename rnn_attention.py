"""NMTS with RNN-Attention

@author: vasudevgupta
"""
import tensorflow as tf
import os
import logging

from utils import OPTM

logger= logging.getLogger(__name__)

class Encoder(tf.keras.Model):
    """
    ENCODER CLASS
    
    INPUT- ENG SEQ
           HIDDEN STATE LAYER1 INITIAL
           HIDDEN STATE LAYER2 INITIAL
           
    OUTPUT- OUTPUT SEQ (FOR IMPLEMENTING ATTENTION)
            HIDDEN STATE LAYER1 FINAL
            HIDDEN STATE LAYER2 FINAL
    """
    
    def __init__(self, config):
        super(Encoder, self).__init__()
        
        self.eng_vocab= config['dataloader']['eng_vocab']
        
        self.embed_size= config['rnn_attention']['embed_size']
        
        self.gru_units= config['rnn_attention']['gru_units']
        
        self.embed= tf.keras.layers.Embedding(input_dim= self.eng_vocab, 
                                              output_dim= self.embed_size)
        
        self.gru1= tf.keras.layers.GRU(units= self.gru_units, kernel_initializer= 'glorot_normal',
                                       return_sequences= True, return_state= True)
        
        self.gru2= tf.keras.layers.GRU(units= self.gru_units, kernel_initializer= 'glorot_normal',
                                       return_sequences= True, return_state= True)
    
    def call(self, input_seq):
        # shape= (batch_size, max_length_encoder_input)
        
        x= self.embed(input_seq)
        # shape= (batch_size, max_length_encoder_input, embed_dims)
        
        output_seq1, hidden1= self.gru1(x)
        # output_seq1 shape= (batch_size, max_length_encoder_input, gru_units)
        
        output_seq2, hidden2= self.gru2(output_seq1)
        # output_seq2 shape= (batch_size, max_length_encoder_input, gru_units)
        
        return output_seq2, hidden1, hidden2

class LuongAttention(tf.keras.layers.Layer):
    """
    LUONG'S MULTIPLICATIVE STYLE
    
    INPUT- ENCODER HIDDEN STATES FOR ALL TIMESTEPS
           DECODER HIDDEN STATE PER TIMESTEP
           
    OUTPUT- CONTEXT VECTOR
            ATTENTION WEIGHTS (JUST FOR VISUALIZING)
    """
    def __init__(self, config):
        super(LuongAttention, self).__init__()
        
        self.gru_units= config['rnn_attention']['gru_units']
        
        # dims same as dims of decoder gru units
        self.tdfc= tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units= self.gru_units))
    
    def call(self, en_seq, dec_output):
        # en_seq- (batch_size, max_len_encoder_input, gru_units)
        # dec_output- (batch_size, 1, gru_units)
        
        scores= tf.keras.backend.batch_dot(self.tdfc(en_seq), dec_output, axes= (2, 2))
        # scores- (batch_size, max_len_encoder_input, 1)
        
        attention_weights= tf.nn.softmax(scores, axis= 1) # alignment vector
        # attention_weights- (batch_size, max_len_encoder_input, 1)
        
        mul= en_seq*attention_weights
        # mul dims- (batch_size, max_len_encoder_input, gru_units)
        
        context_vector= tf.reduce_mean(mul, axis= 1)
        # context_vector- (batch_size, gru_units)
        
        return context_vector, attention_weights

class Decoder(tf.keras.Model):
    """
    DECODER CLASS
    
    CONCAT METHODS ARE DEFINED FOR 1 TIMESTEP INPUT AT A MOMENT
    """
    def __init__(self, config):
        super(Decoder, self).__init__()
        
        self.ger_vocab= config['dataloader']['eng_vocab']
        
        self.embed_size= config['rnn_attention']['embed_size']
        
        self.gru_units= config['rnn_attention']['gru_units']
        
        self.embed= tf.keras.layers.Embedding(input_dim= self.ger_vocab, 
                                              output_dim= self.embed_size)
        
        self.gru1= tf.keras.layers.GRU(units= self.gru_units, kernel_initializer= 'glorot_normal',
                                       return_sequences= True, return_state= True)
        
        self.gru2= tf.keras.layers.GRU(units= self.gru_units, kernel_initializer= 'glorot_normal',
                                       return_sequences= True, return_state= True)
        
        self.attention= LuongAttention(config)
        
        self.fc= tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.ger_vocab))

    def call(self, enc_seq, teach_force_seq, init_hidden1, init_hidden2):
        # en_seq- (batch_size, max_len_encoder_input, gru_units)
        # teach_force_seq- (batch_size, 1)
        # init_hidden1- (batch_size, gru_units)
        # init_hidden2- (batch_size, gru_units)
        
        x= self.embed(teach_force_seq)
        # x- (batch_size, 1, embed_dims)
        
        output_seq1, hidden1= self.gru1(x, initial_state= init_hidden1)
        # output_seq1- (batch_size, 1, gru_units)
        # hidden1- (batch_size, gru_units)
        
        output_seq2, hidden2= self.gru2(output_seq1, initial_state= init_hidden2)
        # output_seq2- (batch_size, 1, gru_units)
        # hidden2- (batch_size, gru_units)
        
        context_vector, attention_weights= self.attention(enc_seq, output_seq2)
        # context_vector dims= (batch_size, gru_units)
        # attention_weights dims= (batch_size, max_len_encoder_input, 1)
        
        x= tf.concat([output_seq2, tf.expand_dims(context_vector, 1)], axis= -1)
        # x- (batch_size, 1, gru_units+gru_units)
        
        x= tf.nn.tanh(x)
        # x- (batch_size, 1, gru_units+gru_units)
        
        y= self.fc(x)
        # y- (batch_size, 1, ger_vocab)
        
        return y, hidden1, hidden2, attention_weights