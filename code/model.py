import tensorflow as tf


class RowLess(object):

    def __init__(self, batch_size, num_sentences, max_input_len, wordvec_dim, hidden_dim):
        self.max_input_len = max_input_len
        self.wordvec_dim = wordvec_dim
        self.num_sentences = num_sentences
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        # Placeholders
        #Word embedding input
        self.input_x = tf.placeholder(tf.float32, [None, None, self.wordvec_dim], name = "input_x")
        self.seq_len = tf.placeholder(tf.float32, [None, ], name='seq_len')
        # LSTM Layer
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.hidden_dim, reuse=True)
        # state = tf.zeros([batch_size, lstm_cell.state_size])
        # lstm_out, _ = lstm_cell(self.input_x, state)
        my_int_variable = tf.get_variable("my_int_variable", [1, 2, 3], dtype=tf.float32,
                                          initializer=tf.zeros_initializer)

        outputs, _ = tf.nn.dynamic_rnn(cell=lstm_cell,
                                        inputs=self.input_x,
                                        sequence_length=self.seq_len,
                                        dtype=tf.float32,
                                        initial_state=my_int_variable)

        self.sess = tf.Session()
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)


r = RowLess(5,2,10,50,20)
print ("OK")

