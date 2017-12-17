import numpy as np
import pandas as pd
import re
import gc

def get_e1_embedding(embeddings_size):
    return np.append(np.ones(int(embeddings_size/2)),np.zeros(embeddings_size-(int(embeddings_size/2))))


def get_e2_embedding(embeddings_size):
    return np.append(np.zeros(int(embeddings_size/2)),np.ones(embeddings_size-(int(embeddings_size/2))))


def create_sent_embeddings(preprocessed_sent, embeddings_model, embeddings_size, max_sent_size, sent_idx = 8,e1_start_idx = 2,e1_end_idx = 3,e2_start_idx = 6,e2_end_idx = 7):
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


def preprocess_sent(meta, embeddings_model, embeddings_size, max_sent_size):
    meta = meta.strip().split('\t')
    meta = np.array(meta,dtype=np.object)
    if len(meta)!=13:
        return None
    for i in [3,4,8,9]:
        meta[i] = int(meta[i])
    meta = meta[[0,2,3,4,5,7,8,9,12]]
    emb = create_sent_embeddings(meta, embeddings_model, embeddings_size, max_sent_size)
    seq_len = len(meta[-1].split())
    return emb,seq_len


def read_from_line(file_obj,line_number):
    file_obj.seek(0,0)
    for i in range(line_number):
        file_obj.readline()
    return file_obj.readline()


def preprocess_sent_idx(file_obj,sent_idx, embeddings_model, embeddings_size, max_sent_size):
    meta = read_from_line(file_obj,sent_idx)
    return(preprocess_sent(meta, embeddings_model, embeddings_size, max_sent_size))


def make_ep_sent_idx(file_name):
    ret = dict()
    line_idx = -1
    with open(file_name,'rb') as f:
        for meta in f.readline():
            line_idx += 1
            meta = meta.decode().strip().split('\t')
            e1 = meta[0]
            e2 = meta[5]
            key = (e1,e2)
            if key in ret:
                ret[key] = np.append(ret[key],line_idx)
            else:
                ret[key] = np.array([line_idx])
    return ret


def make_ep_relations(file_name,relations_dict):
    ret = dict()
    with open(file_name,'rt') as f:
        for meta in f.readline():
            meta = meta.split('\t')
            e1 = meta[0]
            e2 = meta[1]
            key = (e1,e2)
            if key in ret:
                ret[key] = np.append(ret[key],relations_dict[meta[2]])
            else:
                ret[key] = np.array(relations_dict[meta[2]])
    return ret


def repeat_each(a,rep):
        return np.concatenate([np.repeat(i,rep) for i in a])


def tuples_for_entity_pair(pos_idx, pos_rel_idx, all_idxs, relation_keys, max_pos_sample_size, neg_sample_size_sent, neg_sample_size_rel, processed_sents):
    # pos_idx : indexes of sentences the entity pair is present in
    # pos_rel_idx : relations the entity pair is present in
    # relation_keys : list of relation_keys indexed by line number
    gc.collect()

    if pos_idx.shape[0]==0:
        return None
    if pos_idx.shape[0]>max_pos_sample_size:
        pos_idx = np.random.choice(pos_idx,max_pos_sample_size,replace=False)

    # indexes of all possible negative sentences
    neg_idx = [i for i in all_idxs if i not in pos_idx]
    neg_idx = np.random.choice(neg_idx,neg_sample_size_sent)
    
    # preprocess all possible positive sentences
    for sent_idx in pos_idx:
        if sent_idx not in processed_sents:
            processed_sents[sent_idx] = preprocess_sent_idx(sent_idx)

    # preprocess all possible negative sentences
    for sent_idx in neg_idx:
        if sent_idx not in processed_sents:
            processed_sents[sent_idx] = preprocess_sent_idx(sent_idx)

    # prepare s1,s2 indices
    if pos_idx.shape[0]==1:
        sent_1_idx = [pos_idx[0]]
        sent_2_idx = [pos_idx[0]]
    else:
        sent_1_idx = []
        sent_2_idx = []
        for i in range(len(pos_idx)):
            for j in range(i+1,len(pos_idx)):
                sent_1_idx.append(pos_idx[i])
                sent_2_idx.append(pos_idx[j])
    pos_samples_len = len(sent_1_idx)
    sent_1_idx = np.concatenate([repeat_each(sent_1_idx,neg_sample_size_sent),repeat_each(sent_1_idx,neg_sample_size_rel)])
    sent_2_idx = np.concatenate([repeat_each(sent_2_idx,neg_sample_size_sent),repeat_each(sent_2_idx,neg_sample_size_rel)])

    # prepare s3 indices
    sent_3_idx = np.repeat(neg_idx,pos_samples_len)

    # make s1,s2,r2 tuples
    sent_1 = np.array([processed_sents[sent_idx][0] for sent_idx in sent_1_idx])
    sent_2 = np.array([processed_sents[sent_idx][0] for sent_idx in sent_2_idx])
    seq_1 = np.array([processed_sents[sent_idx][1] for sent_idx in sent_1_idx],dtype=np.uint8)
    seq_2 = np.array([processed_sents[sent_idx][1] for sent_idx in sent_2_idx],dtype=np.uint8)
    rel_2 = np.zeros(shape=(len(sent_2),))

    # make s3 tuples
    sent_3 = np.array([processed_sents[sent_idx][0] for sent_idx in sent_3_idx])
    sent_3 = np.concatenate([sent_3,np.zeros(shape=(neg_sample_size_rel,sent_3.shape[1],sent_3.shape[2]))])
    seq_3 = np.array([processed_sents[sent_idx][1] for sent_idx in sent_3_idx],dtype=np.uint8)
    seq_3 = np.concatenate([seq_3,np.zeros(shape=(neg_sample_size_rel,1),dtype=np.uint8)])

    # make rel3 tuples
    rel_3_idx = [i for i in all_idxs if i not in pos_rel_idx]
    rel_3_idx = np.random.choice(rel_3_idx,neg_sample_size_rel)
    rel_3 = np.zeros(shape=(neg_sample_size_sent,))
    rel_3 = np.concatenate([rel_3,relation_keys[rel_3_idx]])

    # flags
    flag_1 = np.array([True for i in range(len(sent_2))])
    flag_2 = np.concatenate([[True for i in range(neg_sample_size_sent)],[False for i in range(neg_sample_size_rel)]])

    return flag_1, flag_2, sent_1, seq_1, sent_2, seq_2, sent_3, seq_3, rel_2, rel_3