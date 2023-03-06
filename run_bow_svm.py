#!/usr/bin/env python3
# coding: utf-8


import numpy as np
import os
import string
import sys
import time
import json
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import LinearSVC
import random
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords


# Set random seeds
np.random.seed(42)
random.seed(42)

# Get stop words
nltk_stopw = stopwords.words("english")


def tokenize(text):
    """Function to tokenize the text based on regex
    pattern matching and removing the stop word"""
    tokens = [
        word.strip(string.punctuation)
        for word in RegexpTokenizer(r"\b[a-zA-Z][a-zA-Z0-9]{2,14}\b").tokenize(
            text
        )
    ]
    return [f.lower() for f in tokens if f and f.lower() not in nltk_stopw]


def get_dataset():
    """Function to load the dataset from the local disk,
    map the polarity i.e. -1, 0, 1 to negative, neutral and
    negative respectively. The difference between how we read
    dataset here and in other functions is, in this function
    the code assumes that the data is split into three different
    text files, seperated by newline."""

    dataset, labels = [], []

    sent_df = pd.read_csv("./data/stanford_data.csv")
    temp_dataset = sent_df["sentences"].to_list()
    dataset = [tokenize(item) for item in temp_dataset]
    labels = sent_df["labels"].to_list()
    num_tokens = [len(x) for x in dataset]

    return dataset, np.array(labels), num_tokens


dataset, labels, num_tokens = get_dataset()

# Some statistics about the data
print("Token Summary. min / avg / median / std / max:")
print(
    np.amin(num_tokens),
    np.mean(num_tokens),
    np.median(num_tokens),
    np.std(num_tokens),
    np.amax(num_tokens),
)

"""Create NP array for dataset, use TfIdf and finally split it into train test
using stratified_split"""

dataset = np.array([np.array(xi) for xi in dataset])
vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=1).fit(dataset)
word_index = vectorizer.vocabulary_
encoded_data = vectorizer.transform(dataset)

stratified_split = StratifiedShuffleSplit(
    n_splits=1, test_size=0.2, random_state=1
).split(encoded_data, labels)
train_indices, test_indices = next(stratified_split)
train_x, test_x = encoded_data[train_indices], encoded_data[test_indices]
train_labels, test_labels = labels[train_indices], labels[test_indices]
model = LinearSVC(tol=1.0e-6, max_iter=20000, verbose=1)
model.fit(train_x, train_labels)
predicted_labels = model.predict(test_x)

results = {}
results["confusion_matrix"] = confusion_matrix(
    test_labels, predicted_labels
).tolist()
results["classification_report"] = classification_report(
    test_labels, predicted_labels, digits=4, output_dict=True
)

# Since it is multiclass dataste, it is wiser to look at the classification
# report and confusion matrix rather than using accuracy, precision on binary
# classification which can be achieved using "positive or not",
# "negative or not", etc.

print(confusion_matrix(labels[test_indices], predicted_labels))
print(classification_report(labels[test_indices], predicted_labels, digits=4))
