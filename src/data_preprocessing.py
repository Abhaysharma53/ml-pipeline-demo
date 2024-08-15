import numpy as np
import pandas as pd
import os

import re
import nltk
import string
from nltk.corpus import stopwords

train_df = pd.read_csv('data/raw/train.csv')
test_df = pd.read_csv('data/raw/test.csv')