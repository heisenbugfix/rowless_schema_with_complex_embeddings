from __future__ import division
import sys

sys.path.insert(0, "./code")
from model import RowlessModel
import pickle as pkl
import numpy as np
import random

# CHOOSE MODEL AND SAVING PATH
mode = 'complex'
saved_dir = "../saved_models/model_real/"
if mode == 'real':
    saved_path = "../saved_models/model_real/rowless_saved_model_real.ckpt"
else:
    saved_path = "../saved_models/model_complex/rowless_saved_model_complex.ckpt"
word_vec = pkl.load(open("../data/id_to_emb_map.pickle", 'rb'))

wordvecdim = 50
num_units = 50
num_relations = 237
vocab_size = len(word_vec)
model = RowlessModel(vocab_size=vocab_size,
                     wordvecdim=wordvecdim,
                     num_units=num_units,
                     num_relations=num_relations,
                     embedding_size=num_units,
                     emb_type=mode)
# with tf.Session() as sess:
#   new_saver = tf.train.import_meta_graph(saved_dir+"rowless_saved_model_real.ckpt.meta")
#   new_saver.restore(sess, tf.train.latest_checkpoint(saved_dir))
#   all_vars = sess.run(tf.all_variables.__name__)
#   for each in all_vars:
#       print(all_vars.name)
#   print("OK")
model.saver.restore(model.sess, saved_path)
relation_embeddings = model.sess.run(model.rel_embeddings)
# LOADING TEST DATA
test_data = pkl.load(open("../data/rowless_data/preprocessed_test.pickle", 'rb'))

# EVALUATION
rr = []
if mode == 'real':
    for elem in test_data:
        sentences = elem[0]
        seq_len = elem[1].astype(np.int32)
        pos_rel = elem[2].tolist()
        neg_rel_per_pos = elem[3]
        if mode == 'real':
            sen_rel_emb = model.sess.run(model.out_lstm_1, feed_dict={model.input_1_s1: sentences,
                                                                      model.seq_len_LSTM_1: seq_len})
            pos_rel_emb = relation_embeddings[pos_rel]
            score_of_pos = np.max(np.dot(sen_rel_emb, pos_rel_emb.T), axis=0).tolist()
            for i, pos_max in enumerate(score_of_pos):
                negrel_indx_fr_this = neg_rel_per_pos[i].tolist()
                neg_rel_emb_fr_this = relation_embeddings[negrel_indx_fr_this]
                score_of_negs = np.max(np.dot(sen_rel_emb, neg_rel_emb_fr_this.T), axis=0).tolist()
                score_of_negs = sorted(score_of_negs, reverse=True)
                # Calculate reciprocal RANK
                isAdded = False
                for i, neg_score in enumerate(score_of_negs):
                    if pos_max > neg_score:
                        isAdded = True
                        rr.append(1 / float(i + 1))
                        break
                    if not isAdded and i == (len(score_of_negs) - 1):
                        rr.append(1 / float(i + 1))
            # print ("OK")
else:
    # TEST COMPLEX EMBEDDING MODEL
    for elem in test_data:
        sentences = elem[0]
        seq_len = elem[1].astype(np.int32)
        pos_rel = elem[2]
        neg_rel_per_pos = elem[3]
        if mode == 'complex':
            sen_rel_emb_real = model.sess.run(model.out_1_real, feed_dict={model.input_1_s1: sentences,
                                                                      model.seq_len_LSTM_1: seq_len})
            sen_rel_emb_img = model.sess.run(model.out_1_im, feed_dict={model.input_1_s1: sentences,
                                                                      model.seq_len_LSTM_1: seq_len})
            dummy_sen = np.zeros(shape=[pos_rel.shape[0],1],dtype=np.int32)
            dummy_seq = np.ones(shape=pos_rel.shape, dtype=np.int32)
            dummy_bool = np.zeros(shape=pos_rel.shape, dtype=np.bool)
            pos_rel_emb_real = model.sess.run(model.out_2_real, feed_dict={model.input_2_s2: dummy_sen,
                                                                      model.seq_len_LSTM_2:dummy_seq,
                                                                      model.input_2_r2:pos_rel,
                                                                      model.use_LSTM_2:dummy_bool})
            pos_rel_emb_im = model.sess.run(model.out_2_im, feed_dict={model.input_2_s2: dummy_sen,
                                                                      model.seq_len_LSTM_2:dummy_seq,
                                                                      model.input_2_r2:pos_rel,
                                                                      model.use_LSTM_2:dummy_bool})
            score_of_pos = []
            for e_pr_real, e_pr_im in zip(pos_rel_emb_real, pos_rel_emb_im):
                max = float('-inf')
                for e_sr_real, e_sr_im in zip(sen_rel_emb_real,sen_rel_emb_img):
                    prod_sum_real = np.sum((e_pr_real*e_sr_real))
                    prod_sum_im   = np.sum((e_pr_im*e_sr_im))
                    score_sen = prod_sum_real + prod_sum_im
                    if score_sen > max:
                        max = score_sen
                score_of_pos.append(max)

            for i, pos_max in enumerate(score_of_pos):
                negrel_indx_fr_this = neg_rel_per_pos[i]
                dummy_sen = np.zeros(shape=[negrel_indx_fr_this.shape[0],1],dtype=np.int32)
                dummy_seq = np.ones(shape=negrel_indx_fr_this.shape, dtype=np.int32)
                dummy_bool = np.zeros(shape=negrel_indx_fr_this.shape, dtype=np.bool)
                neg_rel_emb_real = model.sess.run(model.out_2_real, feed_dict={model.input_2_s2: dummy_sen,
                                                                      model.seq_len_LSTM_2:dummy_seq,
                                                                      model.input_2_r2:negrel_indx_fr_this,
                                                                      model.use_LSTM_2:dummy_bool})
                neg_rel_emb_im = model.sess.run(model.out_2_im, feed_dict={model.input_2_s2: dummy_sen,
                                                                      model.seq_len_LSTM_2:dummy_seq,
                                                                      model.input_2_r2:negrel_indx_fr_this,
                                                                      model.use_LSTM_2:dummy_bool})
                score_of_negs = []
                for e_nr_real, e_nr_im in zip(neg_rel_emb_real, neg_rel_emb_im):
                    max = float('-inf')
                    for e_sr_real,e_sr_im in zip(sen_rel_emb_real,sen_rel_emb_img):
                        prod_sum_real = np.sum((e_nr_real*e_sr_real))
                        prod_sum_im   = np.sum((e_nr_im*e_sr_im))
                        neg_score_sen = prod_sum_im + prod_sum_real
                        if neg_score_sen > max:
                            max = neg_score_sen
                    score_of_negs.append(max)
                score_of_negs = sorted(score_of_negs,reverse=True)
                isAdded = False
                for i, neg_score in enumerate(score_of_negs):
                    if pos_max > neg_score:
                        isAdded = True
                        rr.append(1 / float(i + 1))
                        break
                    if not isAdded and i == (len(score_of_negs) - 1):
                        rr.append(1 / float(i + 1))

MRR = sum(rr) / len(rr)
print(mode,MRR)
