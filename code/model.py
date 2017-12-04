import tensorflow as tf
import random
import numpy as np

# TODO: Make model more generic. Need to abstractions
class RowlessModel(object):
    # Create input placeholder for the network
    def __init__(self, wordvecdim, num_units, kb_relation_use=False,
                 vocab_size=None, embedding_size=None, mode="train"):

        # For training KB relation embeddings
        if kb_relation_use:
            assert embedding_size == num_units
            assert vocab_size is not None
            self.vocab_size = vocab_size
            if embedding_size is None:
                self.embedding_size = 50
            else:
                self.embedding_size = embedding_size
        self.kb_relation_use = kb_relation_use
        self.create_placeholders_kb(kb_relation_use)
        self.create_kb_outputs()
        self.wordvec_dim = wordvecdim
        self.num_units = num_units
        self.create_placeholders_lstm()
        self.create_lstm_outputs()
        self.loss()
        self.sess = tf.Session()
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)
        for v in tf.global_variables():
            print(v.name)

    # Create KB embeddings
    def create_kb_embeddings(self, rel_ids):
        if self.kb_relation_use:
            self.rel_embeddings = tf.get_variable("relation_embeddings",
                                                  [self.vocab_size, self.embedding_size], dtype=tf.float32)
            emb_rel_ids = tf.nn.embedding_lookup(self.rel_embeddings, rel_ids)
            return emb_rel_ids

    # generate the outputs for each input
    def create_lstm_outputs(self):
        with tf.variable_scope('shared_lstm') as scope:
            self.out_input_1 = self.lstm_share(self.num_units, self.input_1, self.seq_len1)
            scope.reuse_variables()  # the variables will be reused.
            self.out_input_2 = self.lstm_share(self.num_units, self.input_2, self.seq_len2)
            scope.reuse_variables()
            self.out_input_3 = self.lstm_share(self.num_units, self.input_3, self.seq_len3)

    def create_kb_outputs(self):
        with tf.variable_scope('shared_kb') as scope:
            self.out_r2 = self.create_kb_embeddings(self.input_2_r2)
            scope.reuse_variables()  # the variables will be reused.
            self.out_r3 = self.create_kb_embeddings(self.input_3_r3)

    # Placeholders for input data to LSTM
    def create_placeholders_lstm(self):
        #shape (batch_size, timesteps, wordvec_dim)
        self.input_1 = tf.placeholder(tf.float32, [None, None, self.wordvec_dim], name="s1")
        self.seq_len1 = tf.placeholder(tf.int32, [None])
        self.input_2 = tf.placeholder(tf.float32, [None, None, self.wordvec_dim], name="s2")
        self.seq_len2 = tf.placeholder(tf.int32, [None])
        self.input_3 = tf.placeholder(tf.float32, [None, None, self.wordvec_dim], name="s3")
        self.seq_len3 = tf.placeholder(tf.int32, [None])

    # Placeholders for relation as input data
    def create_placeholders_kb(self, kb_relation_use=False):
        if kb_relation_use:
            self.input_2_r2 = tf.placeholder(tf.int32, [None, 1], name="r2")
            self.input_3_r3 = tf.placeholder(tf.int32, [None, 1], name="r3")

    # create a shared rnn layer
    def lstm_share(self, num_units, input, seq_len):
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=num_units, state_is_tuple=True)
        outputs, _ = tf.nn.dynamic_rnn(cell=lstm_cell,
                                       inputs=input,
                                       sequence_length=seq_len,
                                       dtype=tf.float32,
                                       time_major=False
                                       )

        return outputs[:,-1]

    # Loss function
    def loss(self):
        self.loss_sentence = tf.reduce_mean(-tf.log(
            tf.sigmoid(tf.reduce_sum(tf.multiply(self.out_input_1, self.out_input_2), axis=1, keep_dims=True) -
                       tf.reduce_sum(tf.multiply(self.out_input_1, self.out_input_3), axis=1, keep_dims=True))))
        if self.kb_relation_use:
            self.loss_relation_1 = -tf.log(tf.sigmoid(
                tf.reduce_sum(tf.multiply(self.out_input_1, self.out_r2), axis=1, keep_dims=True) -
                tf.reduce_sum(tf.multiply(self.out_input_1, self.out_input_3), axis=1, keep_dims=True)))
            self.loss_relation_2 = -tf.log(tf.sigmoid(
                tf.reduce_sum(tf.multiply(self.out_input_1, self.out_input_2), axis=1, keep_dims=True) -
                tf.reduce_sum(tf.multiply(self.out_input_1, self.out_r3), axis=1, keep_dims=True)))

    def train(self, type_1, type_2=None, type_3=None,n_epochs=10):
        #type_1 : (3 x batch_size x timesteps x wordvec_dim, 3 x batch_size x 1)
        self.n_epochs = n_epochs
        self.train_step_0 = tf.train.AdamOptimizer().minimize(self.loss_sentence)
        if self.kb_relation_use:
            self.train_step_1 = tf.train.AdamOptimizer().minimize(self.loss_relation_1)
            self.train_step_2 = tf.train.AdamOptimizer().minimize(self.loss_relation_2)
        d1,d2,d3 = type_1[0][0],type_1[0][1],type_1[0][2]
        l1,l2,l3 = np.reshape(type_1[1][0],(-1)),np.reshape(type_1[1][1],(-1)),np.reshape(type_1[1][2],(-1))

        for i in range(self.n_epochs):
            self.sess.run(self.train_step_0,feed_dict={self.input_1:d1,self.input_2:d2,self.input_3:d3,\
                                                       self.seq_len1:l1,self.seq_len2:l2,self.seq_len3:l3})
        """
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

        # ## Preprocessing functions ##
        # def preprocess_file(f):
        #     """Returns np.array [9 x n_sents]. Columns : [e1,e1_str,e1_start_idx,e1_end_idx,e2,e2_str,e2_start_idx,e2_end_idx,sent]"""
        #     with open(f,'rb') as f:
        #         test_lines = [(str(codecs.unicode_escape_decode(str(i)[2:-3])[0])[1:].split('\t')) for i in f.readlines()]
        #     test_lines = np.array([i for i in test_lines if len(i)==13])
        #     ip = test_lines.T
        #     ip[12] = np.array([re.sub('[^\w\s]','',i.lower()) for i in ip[12]])
        #     return ip[[0,2,3,4,5,7,8,9,12]]

        # # verify whether the variables are reused
        # for v in tf.global_variables():
        #    print(v.name)
        #
        # # concat the three outputs
        # output = tf.concat...
        #
        # # Pass it to the final_lstm layer and out the logits
        # logits = final_layer(output, ...)
        #
        # train_op = ...
        #
        # # train
        # sess.run(train_op, feed_dict{input_1: in1, input_2: in2, input_3:in3, labels: ...}

r = RowlessModel(12, 20, [2, 3, 4], [4, 6, 7], [1, 2, 3], True, 10, 20)
# type_1 : 3 x batch_size x wordvec_dim
batch_size = 10
wordvec_dim = 12
time_steps = 5
type_1 = (np.array([[[[random.randint(0,10) for k in range(wordvec_dim)] for l in range(time_steps)] for j in range(batch_size)] for i in range(3)]),\
    np.array([[5 for i in range(batch_size)] for i in range(3)]))
print(type_1[0].shape,type_1[1].shape)
d1,d2,d3 = type_1[0][0],type_1[0][1],type_1[0][2]
l1, l2, l3 = np.reshape(type_1[1][0],(-1)), np.reshape(type_1[1][1],(-1)), np.reshape(type_1[1][2],(-1))
d1_ = np.copy(d1)
d2_ = np.copy(d2)
d3_ = np.copy(d3)
print(r.sess.run(r.loss_sentence,feed_dict={r.input_1:d1_,r.input_2:d2_,r.input_3:d3_,r.seq_len1:l1,r.seq_len2:l1,r.seq_len3:l3}))
r.train(type_1)
print(r.sess.run(r.loss_sentence,feed_dict={r.input_1:d1,r.input_2:d2,r.input_3:d3}))
print(tf.__version__)
print("OK")