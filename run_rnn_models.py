import os
import h5py
import pickle
import numpy as np
import pandas as pd
from tensorflow import keras
from nltk.corpus import stopwords
from sklearn.utils import shuffle
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import precision_recall_fscore_support

def data_cleaning(data, args):
    punctuation = '""''!"#$%&()*+-/:;<=>?@[\\]^_`{|}~,.''""'
    if args.removePunctuation:
        data = data.apply(lambda x: ''.join(ch for ch in x if ch not in set(punctuation)))
        data = data.str.lower()
    # remove numbers
    if args.removeNumbers:
        data = data.str.replace("[(0-9)]", " ")
    # remove redundant whitespaces, even though it's highly unlikely
    # that the sst2 data has this issue, but it's still best practice
    # to keep this as a sanity check as it's not an computationally
    # expensive operation
    data = data.apply(lambda x:' '.join(x.split()))
    data = data.tolist()
    if args.removeStopWords == "yes":
        stop_words = set(stopwords.words('english'))
        for i, val in enumerate(news):
            temp_sent = word_tokenize(val)
            filtered_sent = [w for w in temp_sent if not w in stop_words]
            data[i] = " ".join(filtered_sent)
    return data


def encoding(tokenizer, news,max_length):
    encoded_news = tokenizer.texts_to_sequences(news)
    padded_news = pad_sequences(encoded_news, maxlen=max_length, padding='post')
    return padded_news

def get_embedding_matrix(gloveVectorLength):
    print(f"Using {gloveVectorLength} Dimensional vectors.")
    hf = h5py.File((f"textRepresentation/embeddingMatrix_{gloveVectorLength}.h5"), 'r')
    embedding_matrix = hf.get('dataset_1')
    embedding_matrix = np.array(embedding_matrix)
    with open("textRepresentation/tokenizer.pkl", 'rb') as handle:
        tokenizer = pickle.load(handle)
    return(embedding_matrix,tokenizer,embedding_matrix.shape[0])

def model_accuracy(model, encoded_sent, labels):
    evalFig = model.evaluate(encoded_sent, labels, verbose=0)
    print('Accuracy: %f' % (evalFig[1]*100))

def RNN_Model(nodes, vocab_size, embedding_matrix, inputLength, gloveVectorLength):
    model = keras.models.Sequential([
            keras.layers.Embedding(vocab_size, gloveVectorLength,  weights=[embedding_matrix],input_length=inputLength),
            keras.layers.SimpleRNN(nodes),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(64, activation='sigmoid'),
            keras.layers.Dense(1, activation = 'sigmoid')
            ])
    return model

def LSTM_Model(nodes, vocab_size, embedding_matrix, inputLength,gloveVectorLength):
    model = keras.models.Sequential([
            keras.layers.Embedding(vocab_size, gloveVectorLength,  weights=[embedding_matrix],input_length=inputLength),
            keras.layers.LSTM(nodes),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(64, activation='sigmoid'),
            keras.layers.Dense(1, activation = 'sigmoid')
            ])
    return model

def GRU_Model(nodes, vocab_size, embedding_matrix, inputLength,gloveVectorLength):
    model = keras.models.Sequential([
            keras.layers.Embedding(vocab_size, gloveVectorLength,  weights=[embedding_matrix],input_length=inputLength),
            keras.layers.GRU(nodes),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(64, activation='sigmoid'),
            keras.layers.Dense(1, activation = 'sigmoid')
            ])
    return model

def biRNN_Model(nodes, vocab_size, embedding_matrix, inputLength,gloveVectorLength):
    model = keras.models.Sequential([
            keras.layers.Embedding(vocab_size, gloveVectorLength,  weights=[embedding_matrix],input_length=inputLength),
            keras.layers.Bidirectional(keras.layers.SimpleRNN(nodes)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(64, activation='sigmoid'),
            keras.layers.Dense(1, activation = 'sigmoid')
            ])
    return model

def biLSTM_Model(nodes, vocab_size, embedding_matrix, inputLength,gloveVectorLength):
    model = keras.models.Sequential([
            keras.layers.Embedding(vocab_size, gloveVectorLength,  weights=[embedding_matrix],input_length=inputLength),
            keras.layers.Bidirectional(keras.layers.LSTM(nodes)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(64, activation='sigmoid'),
            keras.layers.Dense(1, activation = 'sigmoid')
            ])
    return model

def biGRU_Model(nodes, vocab_size, embedding_matrix, inputLength,gloveVectorLength):
    model = keras.models.Sequential([
            keras.layers.Embedding(vocab_size, gloveVectorLength,  weights=[embedding_matrix],input_length=inputLength),
            keras.layers.Bidirectional(keras.layers.GRU(nodes)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(64, activation='sigmoid'),
            keras.layers.Dense(1, activation = 'sigmoid')
            ])
    return model

def model_routine(model, args, embedding_matrix, inputEmbedding, outputLabel):
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    history = model.fit(inputEmbedding,outputLabel,
              epochs=model_epochs,
              verbose=1, batch_size = 256)
