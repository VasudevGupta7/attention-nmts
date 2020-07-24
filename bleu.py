"""BLEU SCORE

@author: vasudevgupta
"""
import nltk
import numpy as np
import pandas as pd

class Bleu:
    
    def __init__(self, N= 4):
        """GET THE BLEU SCORE
        INPUT THE TARGET AND PREDICTION
        """
        self.N= N
    
    def get_score(self, target, pred):
        
        ngrams_prec= []
        for n in range(1, self.N+1):
            precision= self.get_Ngram_precision(target, pred, n)
            ngrams_prec.append(precision)
            
        len_target= np.mean([len(targ) for targ in target]) if type(target[0])==list else len(target)
        
        len_penalty= 1 if len(pred) >= len_target else (1 - np.exp(len_target/len(pred)))    
        
        self.bleu_scr= len_penalty*(np.product(ngrams_prec)**0.25)
        
        return self.bleu_scr
    
    def get_Ngram_precision(self, target, pred, n):
            
        new_pred= list(nltk.ngrams(pred, n))
        count_pred= self._counter(new_pred)
        
        # if there are more than 2 sents in reference
        if type(target[0]) == list:
           
            new_target= [list(nltk.ngrams(target[i], n)) for i in range(len(target))]
            count_target= [self._counter(new_target[i]) for i in range(len(new_target))]
            
            scores= [[np.min([count_pred[tok], count_target[i][tok]])
                       if tok in new_target[i] else 0 
                       for tok in count_pred.keys()] 
                      for i in range(len(new_target))]
            
            final_score= np.max(scores, axis= 0)
        
        else:
        
            new_target= list(nltk.ngrams(target, n))
            count_target= self._counter(new_target)
            
            final_score= [np.min([count_pred[tok], count_target[tok]]) if tok in new_target else 0 for tok in count_pred.keys()]
        
        len_pred= len(new_pred) if len(new_pred)>0 else 1 # just for ensuring that no errors happen
        
        precisions= np.sum(final_score)/len_pred
        
        return precisions
        
    def _counter(self, ls):
        """Returns a dict with freq of each element in ls
        """
        freq= pd.Series(ls).value_counts()
        return dict(freq)

# just for verifying its working
if __name__ == '__main__':
    target= ' cat is walking in the garden'.split()
    pred= 'the cat is walking in the garden'.split()
    bl= Bleu(N= 4)
    print(bl.get_score(target, pred))
