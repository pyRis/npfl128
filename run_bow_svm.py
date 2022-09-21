import numpy as np
import os
import string
import sys
import time
import json
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import LinearSVC
import random

np.random.seed(42)
random.seed(42)


from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
nltk_stopw = stopwords.words('english')

def tokenize (text):
    tokens = [word.strip(string.punctuation) for word in
                RegexpTokenizer(r'\b[a-zA-Z][a-zA-Z0-9]{2,14}\b').tokenize(text)]
    return  [f.lower() for f in tokens if f and f.lower() not in nltk_stopw]

def get_dataset():
    dataset, labels, label_to_str  = [], [], { -1 : 'neg', 1: 'pos', 0: 'neu' }
    for dataset in ['train', 'test']:
        for idx, directory in enumerate(['neg', 'pos', 'neu']):
            dir_name = './data/' + dataset + "/" + directory
            for file in os.listdir(dir_name):
                with open (dir_name + '/' + file, 'r') as f:
                    tokens = tokenize (f.read())
                    if (len(tokens) == 0):
                        continue
                dataset.append(tokens)
                labels.append(idx)
    num_tokens = [len(x) for x in dataset]
    return dataset, np.array(labels), label_to_str, num_tokens

dataset, labels, label_to_str, num_tokens = get_dataset()
print ('Token Summary. min / avg / median / std / 85 / 86 / 87 / 88 / 89 / 90 / 95 / 99 / max:')
print (np.amin(num_tokens), np.mean(num_tokens), np.median(num_tokens), np.std(num_tokens),
        np.percentile(num_tokens, 85), np.percentile(num_tokens, 86), np.percentile(num_tokens, 87),
        np.percentile(num_tokens, 88), np.percentile(num_tokens, 89), np.percentile(num_tokens, 90),
        np.percentile(num_tokens, 95), np.percentile(num_tokens, 99), np.amax(num_tokens))

sorted_by_label = sorted(label_to_str.items(), key=lambda kv: kv[0])
ordered_name = [item[1] for item in sorted_by_label]
num_classes = len(ordered_name)

dataset=np.array([np.array(xi) for xi in dataset])
vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=1).fit(dataset)
word_index = vectorizer.vocabulary_
encoded_data = vectorizer.transform(dataset)

stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=1).split(encoded_data, labels)
train_indices, test_indices = next(stratified_split)
train_x, test_x = encoded_data[train_indices], encoded_data[test_indices]
train_labels, test_labels = labels[train_indices], labels[test_indices]
model = LinearSVC(tol=1.0e-6, max_iter=20000, verbose=1)
model.fit(train_x, train_labels)
predicted_labels = model.predict(test_x)

results = {}
results['confusion_matrix'] = confusion_matrix(test_labels, predicted_labels).tolist()
results['classification_report'] = classification_report(test_labels, predicted_labels,
                                                        digits=4, target_names=ordered_name,
                                                        output_dict=True)

print (confusion_matrix(labels[test_indices], predicted_labels))
print (classification_report(labels[test_indices], predicted_labels, digits=4, target_names=ordered_name))
