import tensorflow as tf


# TODO: Make model more generic. Need to abstractions
class RowlessModel(object):
    # Create input placeholder for the network
    def __init__(self, vocab_size, wordvecdim, num_units, num_relations, embedding_size, emb_type="real"):

        assert embedding_size == num_units
        assert num_relations is not None
        self.num_relations = num_relations
        if embedding_size is None:
            self.embedding_size = 50
        else:
            self.embedding_size = embedding_size

        self.emb_type = emb_type
        #KB
        self.create_conditional_placeholders()
        self.create_placeholders_kb()
        self.create_kb_outputs()
        # WORD_EMB
        self.vocab_size = vocab_size
        self.wordvec_dim = wordvecdim
        self.create_placeholders_word_emb()
        self.create_wemb_outputs()
        #LSTM
        self.num_units = num_units
        self.create_placeholders_lstm()
        self.create_lstm_outputs()
        #OUTPUTS
        self.create_outputs_for_real_loss()
        # Add layer for complex embedding
        if self.emb_type == "complex":
            self.create_outputs_for_complex_loss()
        #LOSS, TRAIN, INITIALIZE
        self.loss()
        self.training()
        self.sess = tf.Session()
        self.init_for_word_emb()
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)
        for v in tf.global_variables():
            print(v.name)

    # Creates KB embeddings
    def create_kb_embeddings(self, rel_ids):
        self.rel_embeddings = tf.get_variable("relation_embeddings",
                                              [self.num_relations, self.embedding_size], dtype=tf.float32)
        emb_rel_ids = tf.nn.embedding_lookup(self.rel_embeddings, rel_ids)
        # emb_rel_ids = tf.reshape(emb_rel_ids, shape=[-1, emb_rel_ids.shape[2]])
        return emb_rel_ids

    # Creates conditional placeholders for switching inputs between LSTM and KB Embeddings
    def create_conditional_placeholders(self):
        self.use_LSTM_2 = tf.placeholder(tf.bool, shape=[None,])
        self.use_LSTM_3 = tf.placeholder(tf.bool, shape=[None,])

    # generates the outputs for each input
    def create_lstm_outputs(self):
        with tf.variable_scope('shared_lstm') as scope:
            self.out_lstm_1 = self.lstm_share(self.num_units, self.out_s1_wv, self.seq_len_LSTM_1)
            scope.reuse_variables()  # the variables will be reused.
            self.out_lstm_2 = self.lstm_share(self.num_units, self.out_s2_wv, self.seq_len_LSTM_2)
            scope.reuse_variables()
            self.out_lstm_3 = self.lstm_share(self.num_units, self.out_s3_wv, self.seq_len_LSTM_3)

    def create_kb_outputs(self):
        with tf.variable_scope('shared_kb') as scope:
            self.out_r2 = self.create_kb_embeddings(self.input_2_r2)
            scope.reuse_variables()  # the variables will be reused.
            self.out_r3 = self.create_kb_embeddings(self.input_3_r3)

    # creates placeholders for word_emb. This will be input to LSTM
    def create_placeholders_word_emb(self):
        self.input_1_s1 = tf.placeholder(tf.int32, [None, None], name="wvec_s1")
        self.input_2_s2 = tf.placeholder(tf.int32, [None, None], name="wvec_s2")
        self.input_3_s3 = tf.placeholder(tf.int32, [None, None], name="wvec_s3")
        self.word_emb_placeholder = tf.placeholder(tf.float32, [self.vocab_size, self.wordvec_dim])

    # Initializer for word_embedding
    def init_for_word_emb(self):
        self.word_emb_init = self.word_embeddings.assign(self.word_emb_placeholder)

    # Creates KB embeddings
    def create_word_embeddings(self, wv_ids):
        self.word_embeddings = tf.get_variable("WordVec_Embedding", [self.vocab_size, self.wordvec_dim], dtype=tf.float32,
                                           trainable=False)
        wemb_ids = tf.nn.embedding_lookup(self.word_embeddings, wv_ids)
        return wemb_ids

    def create_wemb_outputs(self):
        with tf.variable_scope('shared_wemb') as scope:
            self.out_s1_wv = self.create_word_embeddings(self.input_1_s1)
            scope.reuse_variables()  # the variables will be reused.
            self.out_s2_wv = self.create_word_embeddings(self.input_2_s2)
            scope.reuse_variables()  # the variables will be reused.
            self.out_s3_wv = self.create_word_embeddings(self.input_3_s3)


    # Placeholders for input data to LSTM
    def create_placeholders_lstm(self):
        # shape (batch_size, timesteps, wordvec_dim)
        # self.input_LSTM_1 = tf.placeholder(tf.float32, [None, None, self.wordvec_dim], name="s1")
        self.seq_len_LSTM_1 = tf.placeholder(tf.int32, [None,])
        # self.input_LSTM_2 = tf.placeholder(tf.float32, [None, None, self.wordvec_dim], name="s2")
        self.seq_len_LSTM_2 = tf.placeholder(tf.int32, [None,])
        # self.input_LSTM_3 = tf.placeholder(tf.float32, [None, None, self.wordvec_dim], name="s3")
        self.seq_len_LSTM_3 = tf.placeholder(tf.int32, [None,])

    # Placeholders for relation as input data
    def create_placeholders_kb(self):
        self.input_2_r2 = tf.placeholder(tf.int32, [None, ], name="r2")
        self.input_3_r3 = tf.placeholder(tf.int32, [None, ], name="r3")

    # create a shared rnn layer
    def lstm_share(self, num_units, input, seq_len):
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=num_units, state_is_tuple=True)
        outputs, _ = tf.nn.dynamic_rnn(cell=lstm_cell,
                                       inputs=input,
                                       sequence_length=seq_len,
                                       dtype=tf.float32,
                                       time_major=False
                                       )
        rel = self.last_relevant(outputs, seq_len)
        return rel

    def last_relevant(self, output, length):
        batch_size = tf.shape(output)[0]
        max_length = tf.shape(output)[1]
        out_size = int(output.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(output, [-1, out_size])
        relevant = tf.gather(flat, index)
        return relevant


    # Creating embedding for complex layers
    def create_complex_embeddings(self, input_dat, name):
        weight = tf.get_variable(name=name + "_ce_weights", shape=[self.embedding_size, self.embedding_size])
        bias = tf.get_variable(name=name + "_ce_bias", shape=[self.embedding_size])
        out = tf.nn.relu(tf.matmul(input_dat, weight) + bias)
        return out

    # Creating complex embedding layer
    def create_outputs_for_complex_loss(self):
        with tf.variable_scope('shared_complex_emb') as scope:
            self.out_1_real = self.create_complex_embeddings(self.out_1, "real")
            self.out_1_im = self.create_complex_embeddings(self.out_1, "complex")
            scope.reuse_variables()  # Reusing the real and imaginary emb weights across all outputs
            self.out_2_real = self.create_complex_embeddings(self.out_2, "real")
            self.out_2_im = self.create_complex_embeddings(self.out_2, "complex")
            scope.reuse_variables()
            self.out_3_real = self.create_complex_embeddings(self.out_3, "real")
            self.out_3_im = self.create_complex_embeddings(self.out_3, "complex")

    # Creating final outputs that will go in the loss function in real embeddings case
    def create_outputs_for_real_loss(self):
        self.out_1 = self.out_lstm_1
        self.out_2 = tf.where(self.use_LSTM_2, self.out_lstm_2, self.out_r2)
        self.out_3 = tf.where(self.use_LSTM_3, self.out_lstm_3, self.out_r3)

    # Loss function
    def loss(self):
        if self.emb_type == "real":
            self.loss = tf.reduce_mean(-tf.log(
                tf.sigmoid(tf.reduce_sum(tf.multiply(self.out_1, self.out_2), axis=1, keep_dims=True) -
                           tf.reduce_sum(tf.multiply(self.out_1, self.out_3), axis=1, keep_dims=True))))
        elif self.emb_type == "complex":
            r1_r2 = tf.reduce_sum(self.out_1_real * self.out_2_real, axis=1) + \
                        tf.reduce_sum(self.out_1_im * self.out_2_im, axis=1)
            r1_r3 = tf.reduce_sum(self.out_1_real * self.out_3_real, axis=1) + \
                        tf.reduce_sum(self.out_1_im * self.out_3_im, axis=1)
            self.loss = tf.reduce_mean(-tf.log(tf.sigmoid(r1_r2 - r1_r3)))

    def training(self):
        global_step_1 = tf.Variable(0, trainable=False, name='global_step_1')
        self.train_opt = tf.train.AdamOptimizer().minimize(self.loss, global_step=global_step_1)
        self.saver = tf.train.Saver()

# c = wordvecdim = 50
# num_units = 50
# num_relations = 237
# vocab_size = 125
# model = RowlessModel(vocab_size=vocab_size,
#                      wordvecdim=wordvecdim,
#                      num_units=num_units,
#                      num_relations=num_relations,
#                      embedding_size=num_units,
#                      emb_type='complex')
# print("OK")
