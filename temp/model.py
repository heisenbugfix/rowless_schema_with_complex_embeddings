import tensorflow as tf

# TODO: Make model more generic. Need to abstractions
class RowlessModel(object):
    def __init__(self, wordvecdim, num_units, seq1, seq2=None, seq3=None, kb_relation_use=False,
                 vocab_size=None, embedding_size=None, rel_ids=None, mode="train"):
        if mode == "train":
            assert seq2 is not None
            assert seq3 is not None

        # For training KB relation embeddings
        if kb_relation_use:
            assert vocab_size is not None
            assert rel_ids is not None
            self.vocab_size = vocab_size
            self.rel_ids = rel_ids
            if embedding_size is None:
                self.embedding_size = 50
            else:
                self.embedding_size = embedding_size

        self.create_kb_embeddings(kb_relation_use)
        self.wordvec_dim = wordvecdim
        self.num_units = num_units
        self.seq_len1 = seq1
        self.seq_len2 = seq2
        self.seq_len3 = seq3
        self.create_placeholders()
        self.create_outputs()
        self.sess = tf.Session()
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)
        for v in tf.global_variables():
            print(v.name)

    # Create KB embeddings
    def create_kb_embeddings(self, kb_relation_use=False):
        if kb_relation_use:
            self.rel_embeddings = tf.get_variable("relation_embeddings",
                                                  [self.vocab_size, self.embedding_size], dtype=tf.float32)
            self.emb_rel_ids = tf.nn.embedding_lookup(self.rel_embeddings, self.rel_ids)

    def create_outputs(self):
        # o/p operators for each kind of input operator
        with tf.variable_scope('shared_lstm') as scope:
            self.out_input_1 = self.lstm_share(self.num_units, self.input_1, self.seq_len1)
            scope.reuse_variables()  # the variables will be reused.
            self.out_input_2_type_13 = self.lstm_share(self.num_units, self.input_2, self.seq_len2, True)
            scope.reuse_variables()
            self.out_input_3_type_12 = self.lstm_share(self.num_units, self.input_3, self.seq_len3, True)
            self.out_input_2_type_2 = self.relation_share()
            self.out_input_3_type_3 = self.relation_share()

    def create_placeholders(self):
        #placeholders for all kinds of input
        self.input_1 = tf.placeholder(tf.float32, [None, None, self.wordvec_dim], name="s1")
        self.input_2_type_13 = tf.placeholder(tf.float32, [None, None, self.wordvec_dim], name = "s2t13")
        self.input_2_type_2 = tf.placeholder(tf.float32, [None, None, self.wordvec_dim], name = "s2t2")
        self.input_3_type_12 = tf.placeholder(tf.float32, [None, None, self.wordvec_dim], name = "s3t12")
        self.input_3_type_3 = tf.placeholder(tf.float32, [None, None, self.wordvec_dim], name = "s2t3")

    def lstm_share(self, num_units, input, seq_len):
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=num_units, state_is_tuple=True)
        outputs, _ = tf.nn.dynamic_rnn(cell=lstm_cell,
                                       inputs=input,
                                       sequence_length=seq_len,
                                       dtype=tf.float32,
                                       initial_state=tf.random_normal_initializer()
                                       )

        return outputs[-1]

    def loss(self):
        ## returns instead of making a self.loss variable
        if self.type1:
            return self.loss_util(self.out_input_1, self.out_input_2_type_13, self.out_input_3_type_12)
        elif self.type2:
            return self.loss_util(self.out_input_1, self.out_input_2_type_2, self.out_input_3_type_12)
        else:
            return self.loss_util(self.out_input_1, self.out_input_2_type_13, self.out_input_3_type_3)
    
    def loss_util(self,emb_1,emb_2,emb_3):
        return tf.log(tf.sigmoid(tf.matmul(emb_1,emb_2) - tf.matmul(emb_1,emb_3)))
    
    def relation_share(self):
        return tf.Variable(tf.truncated_normal((self.wordvec_dim),mean=0,stddev=1.5),dtype=tf.float32)

    def train(self):
        pass

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


r = RowlessModel(12, 20, [2, 3, 4], [4, 6, 7], [1, 2, 3])
print(tf.__version__)
print("OK")