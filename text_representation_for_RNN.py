#!/usr/bin/env python3
# coding: utf-8
"""Code to prepare text representation for RNN based modules."""

from numpy import asarray
from keras.preprocessing.text import Tokenizer
from numpy import zeros
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import os
import pickle
import h5py


def encoding(tokenizer, sent, max_length):
    """Encode the sentences and padding them accordingly."""
    encoded_sent = tokenizer.texts_to_sequences(sent)
    padded_sent = pad_sequences(encoded_sent, maxlen=max_length,
                                padding='post')
    return padded_sent


def glove_vectors(glove_text_path):
    """Load glove vectors and generate corresponding embedding index."""
    embeddings_index = dict()
    glove_vec = open(glove_text_path, encoding="utf-8")
    for line in glove_vec:
        values = line.split(' ')
        word = values[0]
        coefs = asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    glove_vec.close()
    return embeddings_index


def create_embedding_matrix(tokenizer, embedding_index, vocab_size, dimension):
    """Create the embedding matrix."""
    embedding_matrix = zeros((vocab_size, dimension))
    for word, i in tokenizer.word_index.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return (embedding_matrix)


def create_matrix(sent):
    """Create embedding matrix for the dataset.

    Save it as pickle and h5py file, for better sharing.
    """
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sent)
    tokenizer.word_index
    vocab_size = len(tokenizer.word_index) + 1

    glove_text_path = os.path.join('./glove.840B.300d.txt')
    embedding_index = glove_vectors(glove_text_path)
    embedding_matrix = create_embedding_matrix(tokenizer,
                                               embedding_index, vocab_size,
                                               300)

    with open("./data/text_representation/tokenizer.pkl", 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    hf = h5py.File("./data/text_representation/embeddingMatrix_300.h5", 'w')
    hf.create_dataset('dataset_1', data=embedding_matrix)
    hf.close()
    return(vocab_size)


if __name__ == "__main__":

    """
    Main function to load the data, and create embedding matrix from the data
    using GloVe, this is implemented the most naive method to do so, and this
    can be better (and in fewer lines of codes) implemented using standard
    library functions.
    """

    data = pd.read_csv('./data/stanford_data.csv')
    data = data['sentences'].tolist()

    # Comment out from here to
    punctuation = '""''!"#$%&()*+-/:;<=>?@[\\]^_`{|}~,.''""'
    data = data.apply(lambda x: ''.join(ch for ch in x
                                        if ch not in set(punctuation)))
    data = data.str.lower()
    data = data.str.replace("[(0-9)]", " ")
    data = data.apply(lambda x: ' '.join(x.split()))
    # here if no "data cleaning" is required.

    vocab_size = create_matrix(data)
