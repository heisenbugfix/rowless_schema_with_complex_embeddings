from code.model import RowlessModel

saved_path = "rowless_saved_model.ckpt"
#Reload the saved data
model = RowlessModel()
model.saver.restore(model.sess, saved_path)

input_test_LSTM = None
seq_LSTM = None
relation_indexes = None
# Get the relation embedding
relations_sentences = model.sess.run(model.out_lstm_1, feed_dict={model.input_LSTM_1: input_test_LSTM,
                                                        model.seq_len_LSTM_1: seq_LSTM})

relation_emb_outs = model.sess.run(model.out_r2, feed_dict={model.input_2_r2: relation_indexes})

#Evaluation
