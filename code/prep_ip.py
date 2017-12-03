import codecs
import numpy as np
import pandas as pd
import re

def preprocess_file(f):
    """Returns np.array [9 x n_sents]. Columns : [e1,e1_str,e1_start_idx,e1_end_idx,e2,e2_str,e2_start_idx,e2_end_idx,sent]"""
    with open(f,'rb') as f:
        test_lines = [(str(codecs.unicode_escape_decode(str(i)[2:-3])[0])[1:].split('\t')) for i in f.readlines()]
    test_lines = np.array([i for i in test_lines if len(i)==13])
    ip = test_lines.T
    ip[12] = np.array([re.sub('[^\w\s]','',i.lower()) for i in ip[12]])
    return ip[[0,2,3,4,5,7,8,9,12]]