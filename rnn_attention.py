"""
NMTS with Attention

@author: vasudevgupta
"""
import tensorflow as tf
import os
import logging

from utils import ACTNFN

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
        self.eng_vocab= config['eng_vocab']
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
        self.ger_vocab= config['eng_vocab']
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

def rnn_loss(y, ypred, sce):
    """
    EVERYTHING IS PER TIMESTEP FOR DECODER
    TO AVOID PADDING EFFECT IN CALC OF LOSS;
    WE ARE DOING USING MASK
    """
    loss_= sce(y, ypred) # loss per timestep for whole batch
    # shape= (batch_size, 1) SINCE reduction= NONE
    mask= tf.cast(tf.not_equal(y, 0), tf.float32) # create mask
    loss_= mask*loss_
    return tf.reduce_mean(loss_)

class TrainerRNNAttention(object):
    
    def __init__(self, encoder, decoder, config):
        self.encoder= encoder
        self.decoder= decoder
    
        self.config= config
        self.ckpt_dir= config['rnn_attention']['ckpt_dir']
        self.dec_max_len= config['dataloader']['dec_max_len']
        self.epochs= config['rnn_attention']['epochs']
        
        self.learning_rate= config['rnn_attention']['learning_rate']
        self.optimizer= ACTNFN(config['rnn_attention']['optimizer'], self.learning_rate)
        
        self.sce= tf.keras.losses.SparseCategoricalCrossentropy(from_logits= True, reduction= 'none')
    
    def train(self, inputs, save_model= False, load_model= False, save_evry_ckpt= False):
        
        enc_seq, teach_force_seq, y= inputs[0], inputs[1], inputs[2]
        
        start= time.time()
        avg_loss= []
        
        if load_model: self.restore_checkpoint(self.ckpt_dir)
        
        for e in track(range(1, self.epochs+1)):
            losses= []
            st= time.time()
           
            for enc_seq_batch, teach_force_seq_batch, y_batch in zip(enc_seq, teach_force_seq, y):
                grads, loss= train_step(params, enc_seq_batch, teach_force_seq_batch, y_batch, encoder, decoder, sce)
                losses.append(loss.numpy())
            
            avg_loss.append(np.mean(losses))
            
            if save_evry_ckpt: self.save_checkpoints(self.ckpt_dir)
            
            print(f'EPOCH- {e} ::::::: avgLOSS: {np.mean(losses)} ::::::: TIME: {time.time()- st}')
            logger.info(grads) if e%4 == 0 else None
        
        if save_model: self.save_checkpoints(self.ckpt_dir)
        print(f'total time taken: {time.time()-start}')
        return grads, avg_loss
    
    @tf.function
    def train_step(self, x, ger_inp, ger_out):
        
        with tf.GradientTape() as gtape:
            
            tot_loss= 0
            enc_seq, hidden1, hidden2= self.encoder(x)
            
            for i in range(self.dec_max_len):
            
                dec_inp= tf.expand_dims(ger_inp[:, i], axis= 1)
                ypred, hidden1, hidden2, attention_weights= self.decoder(enc_seq, dec_inp, hidden1, hidden2)
                
                timestep_loss= loss(tf.expand_dims(ger_out[:, i], 1), ypred, self.sce)
                tot_loss+= timestep_loss
           
            avg_timestep_loss= tot_loss/self.dec_max_len
        
        trainable_vars= encoder.trainable_variables + decoder.trainable_variables
        grads= gtape.gradient(avg_timestep_loss, trainable_vars)
        
        self.optimizer.apply_gradients(zip(grads, trainable_vars))
        return grads, avg_timestep_loss
    
    def save_checkpoints(self, ckpt_dir= 'weights/rnn_attention_ckpts'):
        
        checkpoint_dir = ckpt_dir
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        
        ckpt= tf.train.Checkpoint(optimizer= self.optimizer,
                                  encoder= self.encoder,
                                  decoder= self.decoder)
        ckpt.save(file_prefix= checkpoint_prefix)
            
    def restore_checkpoint(self, ckpt_dir= 'weights/rnn_attention'):
       
        checkpoint_dir = ckpt_dir
        
        ckpt= tf.train.Checkpoint(optimizer= self.optimizer,
                                  encoder= self.encoder,
                                  decoder= self.decoder)
        ckpt.restore(tf.train.latest_checkpoint(checkpoint_dir))