import codecs
import numpy as np
import pandas as pd
import re
import gc

data_path = '../data/'

def make_text_transcript(sents_files, max_sentence_size=0, text_transcript_overwrite=True):
    if text_transcript_overwrite:
        with open(data_path+'temp/text_transcript.txt','wt') as f:
            f.write('')
    for sent_file in sents_files:
        try:
            with open(sent_file,'rb') as f:
                test_lines = [(str(codecs.unicode_escape_decode(str(i)[2:-3])[0])[1:].split('\t')) for i in f.readlines()]
            test_lines = np.array([i[-1] for i in test_lines if len(i)==13])
            max_sentence_size = max([len(i.split(' ')) for i in test_lines]+[max_sentence_size])
            text = ' '.join(test_lines)
            with open(data_path+'temp/text_transcript.txt','at') as f:
                f.write(text+' ')
        except FileNotFoundError:
            print(sent_file,'does not exist')
        gc.collect()
    return max_sentence_size

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
    Columns:
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
        test_lines = [(str(codecs.unicode_escape_decode(str(i)[2:-3])[0])[1:].split('\t')) for i in f.readlines()]
    test_lines = np.array([i for i in test_lines if len(i)==13],dtype=np.object)
    for i in [3,4,8,9]:
        test_lines[:,i] = test_lines[:,i].astype(np.int)
    test_lines = test_lines[:,[0,2,3,4,5,7,8,9,12]]
    
    sent_idx = 8
    e1_start_idx = 2
    e1_end_idx = 3
    e2_start_idx = 6
    e2_end_idx = 7
    emb = np.array([create_sent_embeddings(i) for i in test_lines])
    seq_lens = np.reshape(np.array([len(i.split(' ')) for i in test_lines[:,8]]),[-1,1])
    
    return test_lines[:,[0,4]],emb,seq_lens

def create_entity_pairs_vocab(preprocessed_sents, e1_idx=0, e2_idx=4):
    return {tuple(i) for i in preprocessed_sents[:,[e1_idx,e2_idx]]}

def create_entity_pairs_index(entity_pairs_index):
    """
    Creates a dict for indexes of all the sentences with the same entity pairs.
    Key : <e1,e2>
    Values : [s1, s2, ...]
    e1_idx: the index with the identity of entity 1 in preprocessed sentences
    e2_idx: the index with the identity of entity 2 in preprocessed sentences
    """ 
    d = dict()
    for idx,ep in enumerate(entity_pairs_index):
        key = (ep[0],ep[1])
        if key in d:
            d[key] = np.append(d[key],idx)
        else:
            d[key] = np.array([idx])
    return d

def create_sentences_tuples(entity_pairs_index, sents_embeddings, max_pos_sample_size, neg_sample_size_sent, neg_sample_size_rel):
    # TODO:
    # - make neg_sample_size pairs with same sentences, with a diff neg sample for each
    # - with entity pairs with more than max_pos_sample_size, make only max_pos_sample_size*neg_sample_size sentences
    tuples = np.array()
    for ent_pair, idxs in entity_pairs_index:
        idxs = idxs[:max_pos_sample_size]
        for same_pair_idx in idxs:
            neg_samples = sents_embeddings[np.random.choice(range(len(preprocessed_sents)).remove(idxs),neg_sample_size)]
            np.append(tuples,np.concatenate(np.tile(sents_embeddings[same_pair_idx],(1,neg_sample_size)),neg_samples,axis=1))