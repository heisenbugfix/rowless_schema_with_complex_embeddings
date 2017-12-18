from __future__ import division
import numpy as np
import pickle as pkl
from code_package.model import RowlessModel

train_data = pkl.load(open("../data/preprocessed_train.pickle",'rb'))
word_vec = pkl.load(open("../data/id_to_emb_map.pickle",'rb'))
modelsave_path = "../data/rowless_saved_model.ckpt"
wordvecdim = 50
num_units = 50
num_relations = 237
vocab_size = len(word_vec)
model = RowlessModel(vocab_size=vocab_size,
                     wordvecdim=wordvecdim,
                     num_units=num_units,
                     num_relations=num_relations,
                     embedding_size=num_units,
                     emb_type='complex')

#LOAD WORD EMBEDDING IN MODEL
model.sess.run(model.word_emb_init, feed_dict={model.word_emb_placeholder: word_vec})

data_len = len(train_data[0])
batch_size = 2000
num_batches = int(data_len / batch_size + 0.5)
num_epochs = 3000
best_cost = 9999999999999999999.0
# Model Optimizer
train_op = model.train_opt
for epoch in range(num_epochs):
    epoch_cost = 0.0
    for i in range(num_batches):
        if i==num_batches-1:
            indx_l = i*batch_size
            indx_r = data_len
        else:
            indx_l = i*batch_size
            indx_r = (i+1)*batch_size
        _, c = model.sess.run([train_op, model.loss],
                              feed_dict={model.use_LSTM_2    : train_data[0][indx_l: indx_r],
                                         model.use_LSTM_3    : train_data[1][indx_l: indx_r],
                                         model.input_1_s1    : train_data[2][indx_l: indx_r],
                                         model.seq_len_LSTM_1: train_data[3][indx_l: indx_r],
                                         model.input_2_s2  : train_data[4][indx_l: indx_r],
                                         model.seq_len_LSTM_2: train_data[5][indx_l: indx_r],
                                         model.input_3_s3  : train_data[6][indx_l: indx_r],
                                         model.seq_len_LSTM_3: train_data[7][indx_l: indx_r],
                                         model.input_2_r2    : train_data[8][indx_l: indx_r],
                                         model.input_3_r3    : train_data[9][indx_l: indx_r]
                                        })
        print(c)
        epoch_cost += c
    epoch_cost = epoch_cost/(num_batches)
    print("EPOCH:%d     LOSS:%f"%(epoch, epoch_cost))
    if(epoch_cost < best_cost):
        best_cost = epoch_cost
        #Save the model parameters
        model.saver.save(model.sess, save_path=modelsave_path)


