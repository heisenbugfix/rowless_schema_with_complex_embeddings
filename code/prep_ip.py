import codecs
import numpy as np
import pandas as pd
import re

def preprocess_file(f):
    """Returns np.array [n_sents x 9]. Columns : [e1,e1_str,e1_start_idx,e1_end_idx,e2,e2_str,e2_start_idx,e2_end_idx,sent]"""
    with open(f,'rb') as f:
        test_lines = [(str(codecs.unicode_escape_decode(str(i)[2:-3])[0])[1:].split('\t')) for i in f.readlines()]
    test_lines = np.array([i for i in test_lines if len(i)==13])
    ip = test_lines
    #ip[:,12] = np.array([re.sub('[^\w\s]','',i.lower()) for i in ip[:,12]])
    return ip[:,[0,2,3,4,5,7,8,9,12]]

def create_entity_pairs_dict_and_table(preprocessed_sents,e1_idx=0,e2_idx=4):
    """ 
    Creates a dict for all the sentences with the same entity pairs.
    Key : <e1,e2>
    Values : [s1, s2, ...]
    """
    d = dict()
    for s in preprocessed_sents:
        key = (s[e1_idx],s[e2_idx])
        if key in d:
            d[key] = np.append(d[key],s)
        else:
            d[key] = np.array([s])
    return d

def create_entity_pairs_vocab(preprocessed_sents,e1_idx=0,e2_idx=4):
    return {tuple(i) for i in preprocessed_sents[:,[e1_idx,e2_idx]]}