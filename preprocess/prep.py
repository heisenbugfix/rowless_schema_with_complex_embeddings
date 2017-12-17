import numpy as np
import pickle
import fasttext

from prep_ip import *

## hyperparameters ##
org_data_path = '/iesl/canvas/aranjan/rowless/all_data/kb_eps_only/'
data_path = '/iesl/canvas/aranjan/rowless/'

max_pos_sample_size = 10
neg_sample_size_sent = 25
neg_sample_size_rel = 25
embeddings_size = 50
max_sent_size = 65

embeddings_model = fasttext.load_model(data_path+'temp/embeddings_model.bin')

with open('../data/temp/relations_dict.pickle','rb') as f:
    relations_dict = pickle.load(f)

with open(org_data_path+'train.txt') as f:
    relations = np.array([i.split('\t') for i in f.readlines()],dtype=np.object)
relations[:,2] = np.array([int(relations_dict[i]) for i in relations[:,2]])

ids,emb,seq_lens = preprocess_file(org_data_path+'kb_train_65',embeddings_model,embeddings_size,max_sent_size)
pairs_index = create_entity_pairs_index(ids,relations[:,[0,1]])
results = create_sentences_tuples(pairs_index,emb,seq_lens,relations[:,2],max_pos_sample_size,neg_sample_size_sent, neg_sample_size_rel)

with open(data_path+'preprocessed_train.pickle','wb') as f:
    pickle.dump(results,f,protocol=2)