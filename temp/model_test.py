import sys
sys.path.insert(0,'../code')
import tensorflow as tf
import numpy as np
import random
from model_grad_change import RowlessModel
"""
d1,d2,d3 = type_1[0][0],type_1[0][1],type_1[0][2]
l1,l2,l3 = np.reshape(type_1[1][0],(-1)),np.reshape(type_1[1][1],(-1)),np.reshape(type_1[1][2],(-1))
for i in range(self.n_epochs):
    self.sess.run(self.train_step_0,feed_dict={self.input_1:d1,self.input_2:d2,self.input_3:d3,\
                                               self.seq_len1:l1,self.seq_len2:l2,self.seq_len3:l3})

#data: dict
    #0 : (3 x batch_size x timesteps x wordvec_dim, 3 x batch_size x 1)
    #1 : ([batch_size x timesteps x wordvec_dim],[batch_size x wordvec_dim],[batch_size x timesteps x wordvec_dim], 2 x batch_size x 1)
    #2 : ([batch_size x timesteps x wordvec_dim],[batch_size x wordvec_dim],[batch_size x wordvec_dim], batch_size x 1)
for type in range(3):
    d1,d2,d3 = type_1[0][0],type_1[0][1],type_1[0][2] #di : [total_size x timesteps x word_vec_dim]
    l1,l2,l3 = type_1[1][0],type_1[1][1],type_1[1][2] #li : [total_size x 1]
cnt = [0,0,0]
self.train_step = [None, None, None]
self.train_step[0] = tf.train.AdamOptimizer().minimize(self.loss_sentence)
if self.kb_relation_use:
    self.train_step[1] = tf.train.AdamOptimizer().minimize(self.loss_relation_1)
    self.train_step[2] = tf.train.AdamOptimizer().minimize(self.loss_relation_2)
for i in range(self.n_epochs):
    type = random.randint(0,2)
    if type==0:
        self.sess.run(self.train_step_1,feed_dict={self.input_1:d1,self.input_2:d2,self.input_3:d3,\
                                               self.seq_len1:l1,self.seq_len2:l2,self.seq_len3:l3})
"""
r = RowlessModel(12, 20, True, 10, 20)
# type_1 : 3 x batch_size x wordvec_dim
batch_size = 10
wordvec_dim = 12
time_steps = 5
type_1 = (np.array([[[[random.randint(0,10) for k in range(wordvec_dim)] for l in range(time_steps)] for j in range(batch_size)] for i in range(3)]),\
    np.array([[5 for i in range(batch_size)] for i in range(3)]))
print(type_1[0].shape,type_1[1].shape)
d1,d2,d3 = type_1[0][0],type_1[0][1],type_1[0][2]
l1, l2, l3 = np.reshape(type_1[1][0],(-1)), np.reshape(type_1[1][1],(-1)), np.reshape(type_1[1][2],(-1))
print(d1.shape)
r2 = np.array([[0 for i in range(wordvec_dim)] for i in range(batch_size)])
r3 = np.array([[0 for i in range(wordvec_dim)] for i in range(batch_size)])
print(r.sess.run(r.loss_sentence,feed_dict={r.f1:[1], r.f2:[1],\
                                            r.input_1:d1, r.input_2:d2, r.input_3:d3,\
                                            r.seq_len1:l1, r.seq_len2:l1, r.seq_len3:l3,\
                                            r.input_2_r2:r2, r.input_3_r3:r3}))
print(tf.__version__)
print("OK")