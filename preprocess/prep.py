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

print('Embeddings loaded')

with open(org_data_path+'relations_dict.pickle','rb') as f:
    relations_dict = pickle.load(f)

print('Read relations dict')

with open(org_data_path+'train.txt') as f:
    relations = np.array([i.split('\t') for i in f.readlines()],dtype=np.object)
relations[:,2] = np.array([int(relations_dict[i]) for i in relations[:,2]])

print('Starting preprocessing...')
ids,emb,seq_lens = preprocess_file(org_data_path+'kb_train_65',embeddings_model,embeddings_size,max_sent_size)

with open(org_data_path+'intermediate/ids.pickle','wb') as f:
    pickle.dump(ids,f,protocol=2)
with open(org_data_path+'intermediate/emb.pickle','wb') as f:
    pickle.dump(emb,f,protocol=2)
with open(prg_data_path+'intermediate/se_lens.pickle','wb') as f:
    pickle.dump(seq_lens,f,protocol=2)

pairs_index = create_entity_pairs_index(ids,relations[:,[0,1]])
results = create_sentences_tuples(pairs_index,emb,seq_lens,relations[:,2],max_pos_sample_size,neg_sample_size_sent, neg_sample_size_rel)

with open(data_path+'temp/preprocessed_train.pickle','wb') as f:
    pickle.dump(results,f,protocol=2)