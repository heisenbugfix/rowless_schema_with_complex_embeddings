import numpy as np
import pandas as pd
import re
import gc
import pickle

global cnt
cnt = 0

def get_e1_embedding_id():
    return 1

def get_e2_embedding_id():
    return 2

def preprocess_file(f, embeddings_model, max_sent_size):
    """
    Returns
        np.array [n_sents x 2] => <e1_identity,e2_identity>
        np.array [n_sents x timesteps] => <sent_embeddings>
        np.array [n_sents x 1] => <seq_lengths>
    """
    global cnt
    def create_sent_embeddings(preprocessed_sent):
        global cnt
        sent = [str(i) for i in preprocessed_sent[sent_idx].split(' ')]
        e1_s = preprocessed_sent[e1_start_idx]
        e1_e = preprocessed_sent[e1_end_idx]
        e2_s = preprocessed_sent[e2_start_idx]
        e2_e = preprocessed_sent[e2_end_idx]
        if(len(sent)>max_sent_size):
            raise ValueError('Sentence size greater than maximum allowed')
        try:
	        if e1_e <= e2_s:
	            to_ret = [embeddings_model[i] for i in sent[:e1_s]]\
	                            + [get_e1_embedding_id()]\
	                + [embeddings_model[i] for i in sent[e1_e:e2_s]]\
	                   + [get_e2_embedding_id()]\
	                + [embeddings_model[i] for i in sent[e2_e:]]
	        elif e2_e <= e1_s:
	            to_ret = [embeddings_model[i] for i in sent[:e2_s]]\
	                            + [get_e2_embedding_id()]\
	                + [embeddings_model[i] for i in sent[e2_e:e1_s]]\
	                   + [get_e1_embedding_id()]\
	                + [embeddings_model[i] for i in sent[e1_e:]]
	        else:
	            raise ValueError('Entity 1 and Entity 2 indexes are incorrect')
        except NameError as e:
        	cnt += 1
        	# if 'gypsy' in e:
        		# raise ValueError
        	print(sent)
        	return []
        except KeyError as e:
        	cnt += 1
        	print(sent)
        	return []
        return to_ret + [0 for i in range(max_sent_size-len(to_ret))]
    
    """
    Columns of test_lines:
        0 e1
        1 e1_str
        2 e1_start_idx
        3 e1_end_idx
        4 e2
        5 e2_str
        6 e2_start_idx
        7 e2_end_idx
        8 sent
    """
    with open(f,'rb') as f:
        test_lines = [i.decode('latin1').strip().split('\t') for i in f.readlines()]
    print(f,'read...')
    test_lines = np.array([i for i in test_lines if len(i)==13],dtype=np.object)
    for i in [3,4,8,9]:
        test_lines[:,i] = test_lines[:,i].astype(np.int)
    test_lines = test_lines[:,[0,2,3,4,5,7,8,9,12]]
    print('Test lines made...')
    
    sent_idx = 8
    e1_start_idx = 2
    e1_end_idx = 3
    e2_start_idx = 6
    e2_end_idx = 7
    
    emb = []
    seq_lens = []
    ep = []
    for i in test_lines:
    	e = create_sent_embeddings(i)
    	if len(e)!=0:
    		emb.append(e)
    		seq_lens.append(len(i[8].split(' ')))
    		ep.append(i[[0,4]])
    emb = np.array(emb,dtype=np.int32)
    seq_lens = np.array(seq_lens,dtype=np.uint8)
    ep = np.array(ep)

    print('Embeddings and IDs made...')
    return ep,emb,seq_lens

def create_entity_pairs_index_sentences(entity_pairs_sents):
    """
    Creates a dict for indexes of all the sentences with the same entity pairs.
    Key : <e1,e2>
    Values : np.ndarray([s1, s2, ...])
    """ 
    d = dict()
    for idx,ep in enumerate(entity_pairs_sents):
        key = (ep[0],ep[1])
        if key in d:
            d[key] = np.append(d[key],idx)
        else:
            d[key] = np.array([idx],dtype=np.int32)
    
def create_entity_pairs_index_relations(entity_pairs_rels):
    """
    Creates a dict for indexes of all the sentences with the same entity pairs.
    Key : <e1,e2>
    Values : np.ndarray([r1, r2, ...])
    """
    d = dict()
    for idx,ep in enumerate(entity_pairs_rels):
        key = (ep[0],ep[1])
        if key in d:
            d[key] = np.append(d[key],idx)
        else:
            d[key] = np.array([idx],dtype=np.int32)
    return d

def create_test_tuples(ent_pair_idx_sent, ent_pair_idx_rel, sents_embeddings, seq_lens, relations, test_neg_rel_size):
	tuples = []
	disp_step = -1

	all_idxs_rels = np.arange(max_sent_idx)

	for ep in ent_pair_idx_sent.keys():
		disp_step += 1
		if disp_step%10==0:
			print('On entity pair',disp_step)

		if (not ent_pair_idx_sent[ep]) or len(ent_pair_idx_sent[ep])==0\
		 or (not ent_pair_idx_rel[ep]) or len(ent_pair_idx_rel[ep])==0:
			continue

		sents = sents_embeddings[ent_pair_idx_sent[ep]]
		seqs = seq_lens[ent_pair_idx_sent[ep]]

		pos_rels = relations[ent_pair_idx_rel[ep]]

		neg_rels = []
		for pos_rel in pos_rels:
			neg_idx = [i for i in all_idxs_rels if i not in ent_pair_idx_rel[ep]]
			neg_idx = np.random.choice(neg_idx,test_neg_rel_size)
			neg_block = relations[neg_idx]
			neg_rels.append(neg_block)

		tuples.append([sents,seqs,pos_rels,neg_rels])

	return tuples