import numpy as np
import pickle
import fasttext
import json

from wv_to_dict_util import *

## hyperparameters ##
org_data_path = '/iesl/canvas/aranjan/rowless/all_data/kb_eps_only/'
data_path = '/iesl/canvas/aranjan/rowless/'

max_pos_sample_size = 10
neg_sample_size_sent = 25
neg_sample_size_rel = 25
embeddings_size = 50
max_sent_size = 35

embeddings_model = fasttext.load_model(data_path+'temp/embeddings_model.bin')

word_to_id_map = dict()
id_to_emb_map = dict()

word_to_id_map[''] = 0

words = embeddings_model.words
word_to_id_map.update(list( zip(words,list(range(3,len(words)+3))) ))

for word, word_id in word_to_id_map.items():
	id_to_emb_map[word_id] = embeddings_model[word]
id_to_emb_map[0] = [0]*embeddings_size
id_to_emb_map[1] = list(np.append(np.zeros(int(embeddings_size/2)),np.ones(embeddings_size-(int(embeddings_size/2)))))
id_to_emb_map[2] = list(np.append(np.ones(int(embeddings_size/2)),np.zeros(embeddings_size-(int(embeddings_size/2)))))

a = [[key,value] for key,value in id_to_emb_map.items()]
id_to_emb_map_np = np.array([i[1] for i in a])

with open(data_path+'temp/word_to_id_map.pickle','wb') as f:
    pickle.dump(word_to_id_map,f,protocol=2)
with open(data_path+'temp/id_to_emb_map.pickle','wb') as f:
    pickle.dump(id_to_emb_map_np,f,protocol=2)

print('Embeddings loaded and embeddings maps created and saved...')

embeddings_model = word_to_id_map




with open(org_data_path+'relations_dict.pickle','rb') as f:
    relations_dict = pickle.load(f)

print('Read relations dict')

with open(org_data_path+'train.txt') as f:
    relations = np.array([i.split('\t') for i in f.readlines()],dtype=np.object)
relations[:,2] = np.array([int(relations_dict[i]) for i in relations[:,2]])



print('Starting preprocessing...')
ids,emb,seq_lens = preprocess_file(org_data_path+'kb_train_35_cap10',embeddings_model,max_sent_size)
print(ids.shape,emb.shape,seq_lens.shape)

with open(org_data_path+'intermediate/ids.pickle','wb') as f:
    pickle.dump(ids,f,protocol=2)
try:
	with open(org_data_path+'intermediate/emb.pickle','wb') as f:
	    pickle.dump(emb,f,protocol=2)
except:
	with open(org_data_path+'intermediate/emb.json','w') as f:
		print('JSONing emb')
		json.dump(emb.tolist(),f)
with open(org_data_path+'intermediate/se_lens.pickle','wb') as f:
    pickle.dump(seq_lens,f,protocol=2)
print('Done and saved preprocessing...')



pairs_index = create_entity_pairs_index(ids,relations[:,[0,1]])
results = create_sentences_tuples(pairs_index,emb,seq_lens,np.array(relations[:,2],dtype=np.int32),max_pos_sample_size,neg_sample_size_sent, neg_sample_size_rel)

print([i.shape for i in results])

try:
	with open(data_path+'temp/preprocessed_train.pickle','wb') as f:
		a = 0/0
		pickle.dump(results,f,protocol=2)
except:
	with open(data_path+'temp/preprocessed_train.json','w') as f:
		print('JSONing results')
		json.dump([i.tolist() for i in results],f)