#!/usr/bin/env python3

import os
try:
    import pickle
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    import os
    import pickle
    from sklearn.metrics import precision_recall_fscore_support
    from sklearn.metrics import accuracy_score
    import torch
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from torch.nn.utils.rnn import pad_sequence
except:
    print("Library Import Error, Please install the libraries from requirements.txt")


for item in ["results", "output"]:
    for models in ["BOW", "GloVe", "LLM"]:
        os.makedirs(f"{models}{item}", exists_ok=True)
