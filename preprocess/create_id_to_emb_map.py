import pickle
import numpy as np
import fasttext
org_data_path = '/iesl/canvas/aranjan/rowless/all_data/kb_eps_only/'
data_path = '/iesl/canvas/aranjan/rowless/'
embeddings_model = fasttext.load_model(data_path+'embeddings_model.bin')
embeddings_size = 50
with open(data_path+'temp/word_to_id_map.pickle','rb') as f:
    word_to_id_map = pickle.load(f)
id_to_emb_map = dict()
for word, word_id in word_to_id_map.items():
    id_to_emb_map[word_id] = embeddings_model[word]
id_to_emb_map[0] = [0]*embeddings_size
id_to_emb_map[1] = list(np.append(np.zeros(int(embeddings_size/2)),np.ones(embeddings_size-(int(embeddings_size/2)))))
id_to_emb_map[2] = list(np.append(np.ones(int(embeddings_size/2)),np.zeros(embeddings_size-(int(embeddings_size/2)))))
id_to_emb_map_np = np.zeros(shape=[len(word_to_id_map)+2,embeddings_size])
for key,value in id_to_emb_map.items():
    id_to_emb_map_np[key] = value