from __future__ import division
import numpy as np
from code.model import RowlessModel

train_data = np.load("train_file.npy")
wordvecdim = None #TODO
num_units = None #TODO
vocab_size = None #TODO
model = RowlessModel(wordvecdim=wordvecdim, num_units=num_units, vocab_size=vocab_size, embedding_size=num_units, emb_type='complex')

total_data = len(train_data) #TODO: CHANGE THIS
batch_size = 2000
num_batches = int(total_data/batch_size + 0.5)
opt = model.training()
for each in  train_data:
    #TODO : GET ALL THIS DATA FROM TRAINING FILE
    # List of conditional placeholders
    model.use_LSTM_2
    model.use_LSTM_3

    # List of input placeholders for LSTM
    model.input_LSTM_1
    model.seq_len_LSTM_1
    model.input_LSTM_2
    model.seq_len_LSTM_2
    model.input_LSTM_3
    model.seq_len_LSTM_3

    # List of input placeholders for Relation Embeddings
    model.input_2_r2
    model.input_3_r3
