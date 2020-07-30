"""train NMTS on single gpu/cpu based machine

@author: vasudevgupta
"""
import argparse
import logging

logger= logging.getLogger(__name__)


def get_args():
    
    parser= argparse.ArgumentParser(description='RUN THIS FILE FOR TRAINING THIS MODEL')
    
    parser.add_argument('--model_type', type= str, help= 'rnn_attention or transformer')
    
    parser.add_argument('--config', type= str, default= 'config.yaml', help= 'config file for the model')
    parser.add_argument('--save_model', action= 'store_true', default= False, help= 'if specified, model will be saved')
    parser.add_argument('--save_evry_ckpt', action= 'store_true', default= False, help= 'if specified, every epoch will be saved')
    parser.add_argument('--load_model', action= 'store_true', default= False, help= 'if specifiled, weights will be restored before training')
    parser.add_argument('--dataset', type= str, default= 'data/eng2ger.csv', help= 'file name of dataset')
    
    parser.add_argument('--distributed', action= store_true, help= 'if you want to do distributed training')
    parser.add_argument('--num_gpu', type= int, default= 1, help= 'Num of replicas to involve in training')
    
    return parser.parse_args()


