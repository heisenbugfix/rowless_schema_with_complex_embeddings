import tensorflow as tf

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
            tf.sigmoid(
                (self.f1 * tf.reduce_sum(tf.multiply(self.out_input_1, self.out_input_2), axis=1, keep_dims=True)
                 + (1-self.f1) * tf.reduce_sum(tf.multiply(self.out_input_1, self.out_r2), axis=1, keep_dims=True))
                -
                (self.f2*tf.reduce_sum(tf.multiply(self.out_input_1, self.out_input_3), axis=1, keep_dims=True)
                 + (1-self.f2)*tf.reduce_sum(tf.multiply(self.out_input_1, self.out_r3), axis=1, keep_dims=True))
            )))
        self.loss_sentence = tf.reduce_mean(-tf.log(
            tf.sigmoid(tf.reduce_sum(tf.multiply(self.out_input_1, self.out_input_2), axis=1, keep_dims=True) -
                       tf.reduce_sum(tf.multiply(self.out_input_1, self.out_input_3), axis=1, keep_dims=True))))
        if self.kb_relation_use:
            self.loss_relation_1 = tf.reduce_mean(-tf.log(tf.sigmoid(
                tf.reduce_sum(tf.multiply(self.out_input_1, self.out_r2), axis=1, keep_dims=True) -
                tf.reduce_sum(tf.multiply(self.out_input_1, self.out_input_3), axis=1, keep_dims=True))))
            self.loss_relation_2 = tf.reduce_mean(-tf.log(tf.sigmoid(
                tf.reduce_sum(tf.multiply(self.out_input_1, self.out_input_2), axis=1, keep_dims=True) -
                tf.reduce_sum(tf.multiply(self.out_input_1, self.out_r3), axis=1, keep_dims=True))))

    def training(self):
        global_step_1 = tf.Variable(0,trainable=False, name='global_step_1')
        self.train_opt_1 = tf.train.AdamOptimizer().minimize(self.loss_sentence, global_step=global_step_1)
        grads = self.train_opt_1.compute_gradients()
        if self.kb_relation_use:
            global_step_2 = tf.Variable(0, trainable=False, name='global_step_2')
            self.train_opt_2 = tf.train.AdamOptimizer().minimize(self.loss_relation_1, global_step=global_step_2)
            global_step_3 = tf.Variable(0, trainable=False, name='global_step_3')
            self.train_opt_3 = tf.train.AdamOptimizer().minimize(self.loss_relation_2, global_step=global_step_3)
