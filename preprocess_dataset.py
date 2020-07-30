"""Get word embedding for Machine Translation model

@author: vasudevgupta
"""
import yaml
import pandas as pd 

import tensorflow as tf
from dataloader import call_preprocessing, Dataloader
        
if __name__ == '__main__':
    
    """
    FILE READING AND INPUT OUTPUT SEPEARTION
    INPUT- ENGLISH
    OUTPUT- GERMAN
    """
    config= yaml.safe_load(open('config.yaml', 'r'))
    config= config['dataloader']
    
    with open('text/deu.txt', 'r') as file:
        data= file.read()
        dataset= data.split('\n')
    eng= []
    ger= []
    idx= random.sample(range(len(dataset)), config['dataloader']['num_samples'])
    for i in idx:
        e, g, _= dataset[i].split('\t')
        eng.append(e.lower())
        ger.append(g.lower())
    df= pd.DataFrame([eng, ger], index= ['eng', 'ger']).T
    
    
    df['eng_input']= call_preprocessing(df['eng'], nlp_en= True, lower_= True, remove_pattern_= False, tokenize_words_= True,
                    expand_contractions_= True, do_lemmatization_= False,
                    sos= False, eos= False, remove_accents_= True)
    
    df['ger_input']= call_preprocessing(df['ger'], nlp_en= False, remove_pattern_= False, tokenize_words_= True,
                    expand_contractions_= False, do_lemmatization_= False,
                    sos= True, eos= False, remove_accents_= True)
    
    df['ger_target']= call_preprocessing(df['ger'], nlp_en= False, remove_pattern_= False, tokenize_words_= True,
                    expand_contractions_= False, do_lemmatization_= False,
                    sos= False, eos= True, remove_accents_= True)
        
    df['eng_num_tokens']= df['eng_input'].map(lambda txt: len([tok for tok in txt.split(' ')]))
    # removing all the samples which are having lens> 20 [To reduce padding effect]
    df= df[df['eng_num_tokens'] <= 20]
    
    # config['dataloader']['en_max_len']= 20
    
    df['ger_num_tokens']= df['ger_input'].map(lambda txt: len([tok for tok in txt.split(' ')]))
    # removing all the samples which are having lens> 17 [To reduce padding effect]
    df= df[df['ger_num_tokens'] <= 17]
    
    # config['dataloader']['dec_max_len']= 17
    
    # df.to_csv('data/eng2ger.csv', index= False)
