import numpy as np
import pandas as pd
import re
import gc
import pickle

from wv_to_dict_util import *

org_data_path = '/iesl/canvas/aranjan/rowless/all_data/kb_eps_only/'
data_path = '/iesl/canvas/aranjan/rowless/'

max_pos_sample_size = 10
neg_sample_size_sent = 10
neg_sample_size_rel = 10
embeddings_size = 50
max_sent_size = 35


with open(data_path+'temp/word_to_id_map.pickle','rb') as f:
    word_to_id_map = pickle.load(f)
with open(data_path+'temp/id_to_emb_map.pickle','rb') as f:
    id_to_emb_map_np = pickle.load(f)
embeddings_model = word_to_id_map
print('Embeddings loaded and embeddings maps created and saved...')


with open(org_data_path+'relations_dict.pickle','rb') as f:
    relations_dict = pickle.load(f)
print('Read relations dict')


with open(org_data_path+'train.txt') as f:
    relations = np.array([i.split('\t') for i in f.readlines()],dtype=np.object)
relations[:,2] = np.array([int(relations_dict[i]) for i in relations[:,2]])


with open(org_data_path+'intermediate/ids.pickle','rb') as f:
	ids = pickle.load(f)
with open(org_data_path+'intermediate/emb.pickle','rb') as f:
	emb = pickle.load(f)
with open(org_data_path+'intermediate/se_lens.pickle','rb') as f:
	seq_lens = pickle.load(f)
print('Entity pairs, Sentence embeddings, Sequence lengths loaded')


pairs_index = create_entity_pairs_index(ids,relations[:,[0,1]])


with open(org_data_path+'intermediate/train_indexes.pickle','rb') as f:
	indices = pickle.load(f)
results = create_tuples(pairs_index,emb,seq_lens,\
	np.array(relations[:,2],dtype=np.int32),max_pos_sample_size,neg_sample_size_sent, neg_sample_size_rel, indices)

print([i.shape for i in results])

try:
	with open(data_path+'temp/preprocessed_train.pickle','wb') as f:
		pickle.dump(results,f,protocol=2)
except:
	with open(data_path+'temp/preprocessed_train.json','w') as f:
		print('JSONing results')
		json.dump([i.tolist() for i in results],f)