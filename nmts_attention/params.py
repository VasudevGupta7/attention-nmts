"""
HYPERPARAMETERS OF NMTS

@author: vasudevgupta
"""
import tensorflow as tf

class params:
    pass

params.batch_size= 64
params.embed_size= 300
params.gru_units= 128
params.learning_rate= .001
params.optimizer= tf.keras.optimizers.RMSprop(params.learning_rate, clipvalue= 1)
params.epochs= 400

# don't tune it now
params.num_samples= 15000 ## taking only 25000 samples for training purposes
params.eng_vocab = 5776 # got this after tokenizing dataset- english
params.ger_vocab = 8960 # got this after tokenizing dataset- german
params.dec_max_len= 17
params.en_max_len= 20
# params.dec_max_len= 
