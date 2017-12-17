import numpy as np

import sys
sys.path.insert(0,'../code/')
from prep_ip import *

## hyperparameters ##
data_path = "../data/"
max_pos_sample_size = 10
neg_sample_size_sent = 25
neg_sample_size_rel = 25
embeddings_size = 2
max_sent_size = 99

import fasttext
embeddings_model = fasttext.skipgram(data_path+'temp/001.txt',data_path+'temp/embeddings_model_2',min_count=1,dim=embeddings_size)

import pickle
with open('../data/temp/relations_dict.pickle','rb') as f:
    relations_dict = pickle.load(f)

with open(data_path+'train.txt') as f:
    relations = np.array([i.split('\t') for i in f.readlines()],dtype=np.object)
relations[:,2] = np.array([int(relations_dict[i]) for i in relations[:,2]])


ids,emb,seq_lens = preprocess_file(data_path+'kb_00',embeddings_model,embeddings_size,max_sent_size)
pairs_index = create_entity_pairs_index(ids,relations[:,[0,1]])
create_sentences_tuples(pairs_index,emb,seq_lens,relations[:,2],max_pos_sample_size,neg_sample_size_sent, neg_sample_size_rel)