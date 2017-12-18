import numpy as np
import pandas as pd
import re
import gc
import pickle

org_data_path = '/iesl/canvas/aranjan/rowless/all_data/kb_eps_only/'
data_path = '/iesl/canvas/aranjan/rowless/'

with open(org_data_path+'intermediate/train_indexes.pickle','rb') as f:
	indices = pickle.load(f)
