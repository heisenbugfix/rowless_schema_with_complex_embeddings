import numpy as np
import pandas as pd
import re
import gc

def get_e1_embedding(embeddings_size):
    return np.append(np.ones(int(embeddings_size/2)),np.zeros(embeddings_size-(int(embeddings_size/2))))

def get_e2_embedding(embeddings_size):
    return np.append(np.zeros(int(embeddings_size/2)),np.ones(embeddings_size-(int(embeddings_size/2))))

def preprocess_file(f, embeddings_model, embeddings_size, max_sent_size):
    """
    Returns 
        np.array [n_sents x 2] => <e1_identity,e2_identity>
        np.array [n_sents x timesteps] => <sent_embeddings>
        np.array [n_sents x 1] => <seq_lengths>
    """
    def create_sent_embeddings(preprocessed_sent):
        #TODO: limit to_ret to max_sent_size, put check (max_sent_size-len(to_ret)) in return
        sent = preprocessed_sent[sent_idx].split(' ')
        e1_s = preprocessed_sent[e1_start_idx]
        e1_e = preprocessed_sent[e1_end_idx]
        e2_s = preprocessed_sent[e2_start_idx]
        e2_e = preprocessed_sent[e2_end_idx]
        if e1_e <= e2_s:
            to_ret = [embeddings_model[i] for i in sent[:e1_s]]\
                            + [list(get_e1_embedding(embeddings_size))]\
                + [embeddings_model[i] for i in sent[e1_e:e2_s]]\
                   + [list(get_e2_embedding(embeddings_size))]\
                + [embeddings_model[i] for i in sent[e2_e:]]
        elif e2_2 <= e1_s:
            to_ret = [embeddings_model[i] for i in sent[:e2_s]]\
                            + [list(get_e2_embedding(embeddings_size))]\
                + [embeddings_model[i] for i in sent[e2_e:e1_s]]\
                   + [list(get_e1_embedding(embeddings_size))]\
                + [embeddings_model[i] for i in sent[e1_e:]]
        else:
            raise ValueError('Entity 1 and Entity 2 indexes are incorrect')
        return to_ret + [np.zeros(embeddings_size) for i in range(max_sent_size-len(to_ret))]
    
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
        test_lines = [i.decode().strip().split('\t') for i in f.readlines()]
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
    emb = np.array([create_sent_embeddings(i) for i in test_lines])
    seq_lens = np.reshape(np.array([len(i.split(' ')) for i in test_lines[:,8]]),[-1,1])
    
    print('Embeddings and IDs made...')

    return test_lines[:,[0,4]],emb,seq_lens

def create_entity_pairs_vocab(preprocessed_sents, e1_idx=0, e2_idx=4):
    return {tuple(i) for i in preprocessed_sents[:,[e1_idx,e2_idx]]}

def create_entity_pairs_index(entity_pairs_sents,entity_pairs_rels):
    """
    Creates a dict for indexes of all the sentences with the same entity pairs.
    Key : <e1,e2>
    Values : [s1, s2, ...]
    """ 
    d = dict()
    for idx,ep in enumerate(entity_pairs_sents):
        key = (ep[0],ep[1])
        if key in d:
            d[key] = (np.append(d[key][0],idx),d[key][1])
        else:
            d[key] = (np.array([idx]),np.array([],dtype=np.int32))
    for idx,ep in enumerate(entity_pairs_rels):
        key = (ep[0],ep[1])
        if key in d:
            d[key] = (d[key][0],np.append(d[key][1],idx))
        else:
            d[key] = (np.array([]),np.array([idx],dtype=np.int32))
    return d

def create_sentences_tuples(entity_pairs_index, sents_embeddings, seq_lens, relations, max_pos_sample_size, neg_sample_size_sent, neg_sample_size_rel):
    ret_f1 = None
    ret_f2 = None
    ret_sent_1  = None
    ret_seq_1 = None
    ret_sent_2 = None
    ret_seq_2 = None
    ret_sent_3 = None
    ret_seq_3 = None
    ret_rel_2 = None
    ret_rel_3 = None
    
    disp_step = -1
    loop_flag = True

    def repeat_each(a,rep):
        return np.concatenate([np.repeat(i,rep) for i in a])

    all_idxs = np.arange(len(sents_embeddings))
    for ent_pair, values in entity_pairs_index.items():
        disp_step += 1
        if(disp_step%100)==0:
            print('On iterable',disp_step)
        pos_idx, pos_rel_idx = values[0], values[1]
        if pos_idx.shape[0]==0:
            continue

        # make positive samples
        if pos_idx.shape[0]>max_pos_sample_size:
            pos_idx = np.random.choice(pos_idx,max_pos_sample_size,replace=False)
        sent_1_idx = [] # indexes of sentences where entity 1 is present
        sent_2_idx = [] # indexes of sentences where entity 1 is present
        for i in range(len(pos_idx)):
            for j in range(i+1,len(pos_idx)):
                sent_1_idx.append(pos_idx[i])
                sent_2_idx.append(pos_idx[j])
        if pos_idx.shape[0]==1:
            sent_1_idx = [pos_idx[0]]
            sent_2_idx = [pos_idx[0]]
        pos_samples_len = len(sent_1_idx)
        sent_1_idx = np.concatenate([repeat_each(sent_1_idx,neg_sample_size_sent),repeat_each(sent_1_idx,neg_sample_size_rel)])
        sent_2_idx = np.concatenate([repeat_each(sent_2_idx,neg_sample_size_sent),repeat_each(sent_2_idx,neg_sample_size_rel)])
        sent_1 = sents_embeddings[sent_1_idx]
        sent_2 = sents_embeddings[sent_2_idx]
        seq_1 = seq_lens[sent_1_idx]
        seq_2 = seq_lens[sent_2_idx]
        flag_1 = np.array([True for i in range(len(sent_2))])
        rel_2 = np.zeros(shape=(len(sent_2),))

        #make negative samples for sentences
        sent_3_idx = [i for i in all_idxs if i not in pos_idx]
        sent_3_idx = np.random.choice(sent_3_idx,neg_sample_size_sent)
        sent_3_idx = np.repeat(sent_3_idx,pos_samples_len)
        sent_3 = sents_embeddings[sent_3_idx]
        seq_3 = seq_lens[sent_3_idx]
        rel_3 = np.zeros(shape=(neg_sample_size_sent,))

        #make negative samples for relations
        rel_3_idx = [i for i in all_idxs if i not in pos_rel_idx]
        rel_3_idx = np.random.choice(rel_3_idx,neg_sample_size_rel)
        rel_3 = np.concatenate([rel_3,relations[rel_3_idx]])
        sent_3 = np.concatenate([sent_3,np.zeros(shape=(neg_sample_size_rel,sent_3.shape[1],sent_3.shape[2]))])
        seq_3 = np.concatenate([seq_3,np.zeros(shape=(neg_sample_size_rel,1))])

        flag_2 = np.concatenate([[True for i in range(neg_sample_size_sent)],[False for i in range(neg_sample_size_rel)]])

        if loop_flag:
            loop_flag = False
            ret_f1 = flag_1 
            ret_f2 = flag_2 
            ret_sent_1 = sent_1
            ret_seq_1 = seq_1 
            ret_sent_2 = sent_2
            ret_seq_2 = seq_2 
            ret_sent_3 = sent_3
            ret_seq_3 = seq_3
            ret_rel_2 = rel_2
            ret_rel_3 = rel_3
        else:
            ret_f1 = np.concatenate([ret_f1,flag_1])
            ret_f2 = np.concatenate([ret_f2,flag_2])
            ret_sent_1 = np.concatenate([ret_sent_1,sent_1])
            ret_seq_1 = np.concatenate([ret_seq_1,seq_1])
            ret_sent_2 = np.concatenate([ret_sent_2,sent_2])
            ret_seq_2 = np.concatenate([ret_seq_2,seq_2])
            ret_sent_3 = np.concatenate([ret_sent_3,sent_3])
            ret_seq_3 = np.concatenate([ret_seq_3,seq_3])
            ret_rel_2 = np.concatenate([ret_rel_2,rel_2])
            ret_rel_3 = np.concatenate([ret_rel_3,rel_3])

        gc.collect()
    return ret_f1, ret_f2, ret_sent_1, ret_seq_1, ret_sent_2, ret_seq_2, ret_sent_3, ret_seq_3, ret_rel_2, ret_rel_3