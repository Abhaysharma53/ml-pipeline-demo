import numpy as np
import pandas as pd

import pickle

clf = pickle.load(open('model.pkl', 'rb'))

test_data = pd.read_csv('data/processed/test_')