import tensorflow as tf


# TODO: Make model more generic. Need to abstractions
class RowlessModel(object):
    # Create input placeholder for the network
    def __init__(self, wordvecdim, num_units, kb_relation_use=False,
                 vocab_size=None, embedding_size=None, mode="train", emb_type="real"):

        # For training KB relation embeddings
        if kb_relation_use:
            assert embedding_size == num_units
            assert vocab_size is not None
            self.vocab_size = vocab_size
            if embedding_size is None:
                self.embedding_size = 50
            else:
                self.embedding_size = embedding_size

        self.emb_type = emb_type
        self.kb_relation_use = kb_relation_use
        self.create_conditional_placeholders()
        self.create_placeholders_kb(kb_relation_use)
        self.create_kb_outputs()
        self.wordvec_dim = wordvecdim
        self.num_units = num_units
        self.create_placeholders_lstm()
        self.create_lstm_outputs()
        self.create_outputs_for_real_loss()
        # Add layer for complex embedding
        if self.emb_type == "complex":
            self.create_outputs_for_complex_loss()
        self.loss()
        self.training()
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
            emb_rel_ids = tf.reshape(emb_rel_ids, shape=[-1, emb_rel_ids.shape[2]])
            return emb_rel_ids

    # Creating conditional placeholders for switching inputs between LSTM and KB Embeddings
    def create_conditional_placeholders(self):
        self.use_LSTM_2 = tf.placeholder(tf.bool)
        self.use_LSTM_3 = tf.placeholder(tf.bool)

    # generate the outputs for each input
    def create_lstm_outputs(self):
        with tf.variable_scope('shared_lstm') as scope:
            self.out_lstm_1 = self.lstm_share(self.num_units, self.input_1, self.seq_len1)
            scope.reuse_variables()  # the variables will be reused.
            self.out_lstm_2 = self.lstm_share(self.num_units, self.input_2, self.seq_len2)
            scope.reuse_variables()
            self.out_lstm_3 = self.lstm_share(self.num_units, self.input_3, self.seq_len3)

    def create_kb_outputs(self):
        with tf.variable_scope('shared_kb') as scope:
            self.out_r2 = self.create_kb_embeddings(self.input_2_r2)
            scope.reuse_variables()  # the variables will be reused.
            self.out_r3 = self.create_kb_embeddings(self.input_3_r3)

    # Placeholders for input data to LSTM
    def create_placeholders_lstm(self):
        # shape (batch_size, timesteps, wordvec_dim)
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

        return outputs[:, -1]

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

    # Creating final outputs that will go in the loss function
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
                        tf.reduce_sum(self.out_1_im * self.out_1_im, axis=1)
            r1_r3 = tf.reduce_sum(self.out_1_real * self.out_3_real, axis=1) + \
                        tf.reduce_sum(self.out_1_im * self.out_3_im, axis=1)
            self.loss = tf.reduce_mean(r1_r2 - r1_r3)


            # self.loss_sentence = tf.reduce_mean(-tf.log(
            #     tf.sigmoid(tf.reduce_sum(tf.multiply(self.out_lstm_1, self.out_lstm_2), axis=1, keep_dims=True) -
            #                tf.reduce_sum(tf.multiply(self.out_lstm_1, self.out_lstm_3), axis=1, keep_dims=True))))
            # if self.kb_relation_use:
            #     self.loss_relation_1 = tf.reduce_mean(-tf.log(tf.sigmoid(
            #         tf.reduce_sum(tf.multiply(self.out_lstm_1, self.out_r2), axis=1, keep_dims=True) -
            #         tf.reduce_sum(tf.multiply(self.out_lstm_1, self.out_lstm_3), axis=1, keep_dims=True))))
            #     self.loss_relation_2 = tf.reduce_mean(-tf.log(tf.sigmoid(
            #         tf.reduce_sum(tf.multiply(self.out_lstm_1, self.out_lstm_2), axis=1, keep_dims=True) -
            #         tf.reduce_sum(tf.multiply(self.out_lstm_1, self.out_r3), axis=1, keep_dims=True))))

    def training(self):
        global_step_1 = tf.Variable(0, trainable=False, name='global_step_1')
        self.train_opt = tf.train.AdamOptimizer().minimize(self.loss, global_step=global_step_1)
