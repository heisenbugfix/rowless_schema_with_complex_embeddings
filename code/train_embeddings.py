import numpy as np
import pandas as pd
import gc

org_data_path = '/iesl/canvas/aranjan/rowless/all_data/kb_eps_only/'
data_path = '/iesl/canvas/aranjan/rowless/'

embeddings_size = 25

gc.collect()

print('Starting training of embeddings..')
import fasttext
embeddings_model = fasttext.skipgram(data_path+'temp/text_transcript.txt',data_path+'temp/embeddings_model',min_count=1,dim=embeddings_size)
print('completed training of embeddings..')