"""Some important utilities

@author: vasudevgupta
"""
import tensorflow as tf

def OPTM(actn, learning_rate):
    dictn = {
        'adam': tf.keras.optimizers.Adam(learning_rate),
        'rmsprop': tf.keras.optimizers.RMSprop(learning_rate),
        'sgd': tf.keras.optimizers.SGD(learning_rate)
        }
    return dictn[actn]

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

