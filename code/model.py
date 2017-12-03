import tensorflow as tf

#TODO: Make model more generic. Need to abstractions
class RowlessModel(object):
    # Create input placeholder for the network
    def __init__(self, wordvecdim, num_units, seq1, seq2=None, seq3=None, mode="train"):
        if mode=="train":
            assert seq2 is not None
            assert seq3 is not None

        self.wordvec_dim  = wordvecdim
        self.num_units = num_units
        self.seq_len1 = seq1
        self.seq_len2 = seq2
        self.seq_len3 = seq3
        self.create_placeholders()
        self.create_lstm_outputs()
        self.loss()
        self.sess = tf.Session()
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)

        # generate the outputs for each input
    def create_lstm_outputs(self):
        with tf.variable_scope('shared_lstm') as scope:
            self.out_input_1 = self.lstm_share(self.num_units, self.input_1, self.seq_len1)
            scope.reuse_variables()  # the variables will be reused.
            self.out_input_2 = self.lstm_share(self.num_units, self.input_2, self.seq_len2)
            scope.reuse_variables()
            self.out_input_3 = self.lstm_share(self.num_units, self.input_3, self.seq_len3)

    def create_placeholders(self):
        self.input_1 = tf.placeholder(tf.float32, [None, None, self.wordvec_dim], name="s1")
        self.input_2 = tf.placeholder(tf.float32, [None, None, self.wordvec_dim], name = "s2")
        self.input_3 = tf.placeholder(tf.float32, [None, None, self.wordvec_dim], name = "s3")

    # create a shared rnn layer
    def lstm_share(self, num_units, input, seq_len):
       lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=num_units, state_is_tuple=True)
       outputs, _ = tf.nn.dynamic_rnn(cell=lstm_cell,
                                      inputs=input,
                                      sequence_length=seq_len,
                                      dtype=tf.float32
                                      )

       return outputs[-1]

    def loss(self):
        self.loss = tf.log(tf.sigmoid(tf.matmul(self.out_input_1, self.out_input_2) - tf.matmul(self.out_input_1, self.out_input_3)))

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



r = RowlessModel(12,20,[2,3,4],[4,6,7],[1,2,3])
print ("OK")

