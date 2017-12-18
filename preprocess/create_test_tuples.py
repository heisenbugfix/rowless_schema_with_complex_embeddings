import numpy as np
import pandas as pd
import re
import gc
import pickle

from create_test_tuples_util import *

# org_data_path = '/iesl/canvas/aranjan/rowless/all_data/kb_eps_only/'
# data_path = '/iesl/canvas/aranjan/rowless/'
org_data_path = "../data/"
data_path = "../data/"

embeddings_size = 50
max_sent_size = 35
test_neg_rel_size = 10

with open(data_path+'temp/word_to_id_map.pickle','rb') as f:
    word_to_id_map = pickle.load(f)
embeddings_model = word_to_id_map
print('Embeddings loaded...')
print(len(word_to_id_map))


print('Starting preprocessing...')
ids,emb,seq_lens = preprocess_file(org_data_path+'kb_test_35_cap10',embeddings_model,max_sent_size)
print(ids.shape,emb.shape,seq_lens.shape)

with open(org_data_path+'intermediate/test_ids.pickle','wb') as f:
    pickle.dump(ids,f,protocol=2)
try:
	with open(org_data_path+'intermediate/test_emb.pickle','wb') as f:
	    pickle.dump(emb,f,protocol=2)
except:
	with open(org_data_path+'intermediate/test_emb.json','w') as f:
		print('JSONing emb')
		json.dump(emb.tolist(),f)
with open(org_data_path+'intermediate/test_se_lens.pickle','wb') as f:
    pickle.dump(seq_lens,f,protocol=2)
print('Entity pairs, Sentence embeddings, Sequence lengths pickled')

"""
with open(org_data_path+'intermediate/test_ids.pickle','rb') as f:
	ids = pickle.load(f)
with open(org_data_path+'intermediate/test_emb.pickle','rb') as f:
	emb = pickle.load(f)
with open(org_data_path+'intermediate/test_se_lens.pickle','rb') as f:
	seq_lens = pickle.load(f)
print('Entity pairs, Sentence embeddings, Sequence lengths loaded')
"""


with open(org_data_path+'temp/relations_dict.pickle','rb') as f:
    relations_dict = pickle.load(f)
print('Read relations dict')


with open(org_data_path+'test.txt') as f:
    relations = np.array([i.split('\t') for i in f.readlines()],dtype=np.object)
relations[:,2] = np.array([int(relations_dict[i]) for i in relations[:,2]])


ent_pair_idx_sent = create_entity_pairs_index_sentences(ids)
ent_pair_idx_rel = create_entity_pairs_index_relations(relations[:[0,1]])


tuples = create_test_tuples(ent_pair_idx_sent, ent_pair_idx_rel, emb, seq_lens, relations[:,2], test_neg_rel_size)
try:
	with open(data_path+'temp/preprocessed_test.pickle','wb') as f:
		pickle.dump(results,f,protocol=2)
except:
	with open(data_path+'temp/preprocessed_test.json','w') as f:
		print('JSONing results')
		json.dump([i.tolist() for i in results],f)
