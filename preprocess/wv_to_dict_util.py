import numpy as np
import pandas as pd
import re
import gc
import pickle

org_data_path = '/iesl/canvas/aranjan/rowless/all_data/kb_eps_only/'
data_path = '/iesl/canvas/aranjan/rowless/'

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
    def create_sent_embeddings(preprocessed_sent):
        #TODO: limit to_ret to max_sent_size, put check (max_sent_size-len(to_ret)) in return
        sent = preprocessed_sent[sent_idx].split(' ')
        e1_s = preprocessed_sent[e1_start_idx]
        e1_e = preprocessed_sent[e1_end_idx]
        e2_s = preprocessed_sent[e2_start_idx]
        e2_e = preprocessed_sent[e2_end_idx]
        if(len(sent)>max_sent_size):
            raise ValueError('Sentence size greater than maximum allowed')
        if e1_e <= e2_s:
            to_ret = [embeddings_model[i] for i in sent[:e1_s]]\
                            + [get_e1_embedding_id()]\
                + [embeddings_model[i] for i in sent[e1_e:e2_s]]\
                   + [get_e2_embedding_id()]\
                + [embeddings_model[i] for i in sent[e2_e:]]
        elif e2_2 <= e1_s:
            to_ret = [embeddings_model[i] for i in sent[:e2_s]]\
                            + [get_e2_embedding_id()]\
                + [embeddings_model[i] for i in sent[e2_e:e1_s]]\
                   + [get_e1_embedding_id()]\
                + [embeddings_model[i] for i in sent[e1_e:]]
        else:
            raise ValueError('Entity 1 and Entity 2 indexes are incorrect')
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
    emb = [create_sent_embeddings(i) for i in test_lines]
    emb = np.array(emb,dtype=np.int32)

    seq_lens = [len(i.split(' ')) for i in test_lines[:,8]]
    seq_lens = np.array(seq_lens,dtype=np.uint8)
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
            d[key] = (np.array([idx],dtype=np.int32),np.array([],dtype=np.int32))
    for idx,ep in enumerate(entity_pairs_rels):
        key = (ep[0],ep[1])
        if key in d:
            d[key] = (d[key][0],np.append(d[key][1],idx))
        else:
            d[key] = (np.array([],dtype=np.int32),np.array([idx],dtype=np.int32))
    return d


def create_sentences_indexes(entity_pairs_index, max_sent_idx, max_rel_idx, max_pos_sample_size, neg_sample_size_sent, neg_sample_size_rel, spl_sent_idx, spl_rel_idx):
    disp_step = 0
    loop_flag = True
    print('Starting creating indices tuples') 
    flag_2_block_count = 0

    def repeat_each(a,rep):
        return np.concatenate([np.repeat(i,rep) for i in a])

    all_idxs = np.arange(max_sent_idx)
    all_idxs_rels = np.arange(max_rel_idx)

    for ent_pair, values in entity_pairs_index.items():
        disp_step += 1
        if(disp_step%5)==0:
            print('On iterable',disp_step)
            print(ret_sent_1_idx.shape, ret_sent_2_idx.shape, ret_sent_3_idx.shape, ret_rel_3_idx.shape, flag_2_block_count)
            if(disp_step%50)==0:
                with open(org_data_path+'intermediate/train_indexes.pickle','wb') as f:
                    results = [ret_sent_1_idx, ret_sent_2_idx, ret_sent_3_idx,  ret_rel_3_idx]
                    pickle.dump(results,f,protocol=2)
                gc.collect()

        pos_idx, pos_rel_idx = values[0], values[1]

        if pos_idx.shape[0]==0:
            continue
        
        # make positive samples
        if pos_idx.shape[0]==1:
            sent_1_idx = [pos_idx[0]]
            sent_2_idx = [pos_idx[0]]
        else:
            if pos_idx.shape[0]>max_pos_sample_size:
                pos_idx = np.random.choice(pos_idx,max_pos_sample_size,replace=False)
            sent_1_idx = [] # indexes of sentences where entity 1 is present
            sent_2_idx = [] # indexes of sentences where entity 2 is present
            for i in range(len(pos_idx)):
                for j in range(i+1,len(pos_idx)):
                    sent_1_idx.append(pos_idx[i])
                    sent_2_idx.append(pos_idx[j])
        
        pos_samples_len = len(sent_1_idx)

        sent_1_idx = np.concatenate([repeat_each(sent_1_idx,neg_sample_size_sent),repeat_each(sent_1_idx,neg_sample_size_rel)])
        sent_2_idx = np.concatenate([repeat_each(sent_2_idx,neg_sample_size_sent),repeat_each(sent_2_idx,neg_sample_size_rel)])
        
        #make negative samples for sentences
        sent_3_idx = [i for i in all_idxs if i not in pos_idx]
        sent_3_idx = np.random.choice(sent_3_idx,neg_sample_size_sent)
        sent_3_idx = np.repeat(sent_3_idx,pos_samples_len)
        sent_3_idx = np.concatenate([sent_3_idx,np.full([neg_sample_size_rel,],spl_sent_idx,dtype=np.int32)])

        #make negative samples for relations
        rel_3_idx = [i for i in all_idxs_rels if i not in pos_rel_idx]
        rel_3_idx = np.random.choice(rel_3_idx,neg_sample_size_rel)
        rel_3_idx = np.concatenate([np.full([neg_sample_size_sent,],spl_rel_idx,dtype=np.int32),rel_3_idx])

        flag_2_block_count += 1

        if loop_flag:
            loop_flag = False
            ret_sent_1_idx = sent_1_idx
            ret_sent_2_idx = sent_2_idx
            ret_sent_3_idx = sent_3_idx
            ret_rel_3_idx = rel_3_idx
        else:
            ret_sent_1_idx = np.concatenate([ret_sent_1_idx,sent_1_idx])
            ret_sent_2_idx = np.concatenate([ret_sent_2_idx,sent_2_idx])
            ret_sent_3_idx = np.concatenate([ret_sent_3_idx,sent_3_idx])
            ret_rel_3_idx = np.concatenate([ret_rel_3_idx,rel_3_idx])

    return ret_sent_1_idx, ret_sent_2_idx, ret_sent_3_idx, ret_rel_3_idx, flag_2_block_count

def create_tuples(entity_pairs_index, sents_embeddings, seq_lens, relations, max_pos_sample_size, neg_sample_size_sent, neg_sample_size_rel):
    sent_1_idx, sent_2_idx, sent_3_idx, rel_3_idx, flag_2_block_count\
    = create_sentences_indexes(entity_pairs_index, len(sents_embeddings), len(relations),\
        max_pos_sample_size, neg_sample_size_sent, neg_sample_size_rel, len(sents_embeddings)+1, len(relations)+1)

    with open(org_data_path+'intermediate/train_indexes.pickle','wb') as f:
        results = [ret_sent_1_idx, ret_sent_2_idx, ret_sent_3_idx, ret_rel_2_idx, ret_rel_3_idx]
        pickle.dump(results,f,protocol=2)
    
    print('Indices done')

    sents_embeddings = np.append(sents_embeddings,np.zeros(shape=[sents_embeddings.shape[1]]))
    relations = np.append(relations,[0])
    
    sent_1 = sents_embeddings[sent_1_idx]
    print('Sent 1 done')
    
    sent_2 = sents_embeddings[sent_2_idx]
    print('Sent 2 done')
    
    seq_1 = seq_lens[sent_1_idx]
    print('Seq 1 done')
    
    seq_2 = seq_lens[sent_2_idx]
    print('Seq 2 done')
    
    rel_2 = np.zeros(shape=(len(sent_2),),dtype=np.int32)
    print('Rel 2 done')
    
    sent_3 = sents_embeddings[sent_3_idx]
    print('Sent 3 done')
    
    seq_3 = seq_lens[sent_3_idx]
    print('Seq 3 done')
    
    rel_3 = relations[rel_3_idx]
    print('Rel 3 done')
    
    flag_1 = np.array([True for i in range(len(sent_2))])
    print('Flag 1 done')
    
    flag_2 = np.repeat(np.concatenate([[True for i in range(neg_sample_size_sent)],[False for i in range(neg_sample_size_rel)]]),flag_2_block_count)
    print('Flag 2 done')

    return flag_1, flag_2, sent_1, seq_1, sent_2, seq_2, sent_3, seq_3, rel_2, rel_3
