#!/usr/bin/env python3
# coding: utf-8

"""Code to implement and run RNN based modules such as LSTM, RNN, GRU."""

import os
import h5py
import pickle
import argparse
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

np.random.seed(42)


def encoding(tokenizer, sent, max_len):
    """Encode the sentence and pad to maximum length."""
    encoded_sent = tokenizer.texts_to_sequences(sent)
    padded_sent = pad_sequences(encoded_sent, maxlen=max_len, padding="post")
    return padded_sent


def load_glove(glove_vec_len):
    """Load GloVe embedding and tokenizer."""
    print(f"Using {glove_vec_len} Dimensional vectors.")
    hf = h5py.File(
        (
            "./data/text_representation/embeddingMatrix_"
            + f"{glove_vec_len}.h5"
        ),
        "r",
    )
    embedding_matrix = hf.get("dataset_1")
    embedding_matrix = np.array(embedding_matrix)
    with open("./data/text_representation/tokenizer.pkl", "rb") as handle:
        tokenizer = pickle.load(handle)
    return (embedding_matrix, tokenizer, embedding_matrix.shape[0])


def model_accuracy(model, encoded_sent, labels):
    """Evaluate the model on sentence and print accuracy."""
    evalFig = model.evaluate(encoded_sent, labels, verbose=0)
    print("Accuracy: %f" % (evalFig[1] * 100))


def RNN_Model(nodes, vocab_size, embedding_matrix, inputLength, glove_vec_len):
    """Define RNN model architecture."""
    model = keras.models.Sequential(
        [
            keras.layers.Embedding(
                vocab_size,
                glove_vec_len,
                weights=[embedding_matrix],
                input_length=inputLength,
            ),
            keras.layers.SimpleRNN(nodes),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(64, activation="sigmoid"),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    return model


def LSTM_Model(
    nodes, vocab_size, embedding_matrix, inputLength, glove_vec_len
):
    """Define LSTM model architecture."""
    model = keras.models.Sequential(
        [
            keras.layers.Embedding(
                vocab_size,
                glove_vec_len,
                weights=[embedding_matrix],
                input_length=inputLength,
            ),
            keras.layers.LSTM(nodes),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(64, activation="sigmoid"),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    return model


def GRU_Model(nodes, vocab_size, embedding_matrix, inputLength, glove_vec_len):
    """Define GRU model architecture."""
    model = keras.models.Sequential(
        [
            keras.layers.Embedding(
                vocab_size,
                glove_vec_len,
                weights=[embedding_matrix],
                input_length=inputLength,
            ),
            keras.layers.GRU(nodes),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(64, activation="sigmoid"),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    return model


def biRNN_Model(
    nodes, vocab_size, embedding_matrix, inputLength, glove_vec_len
):
    """Define Bi-RNN model architecture."""
    model = keras.models.Sequential(
        [
            keras.layers.Embedding(
                vocab_size,
                glove_vec_len,
                weights=[embedding_matrix],
                input_length=inputLength,
            ),
            keras.layers.Bidirectional(keras.layers.SimpleRNN(nodes)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(64, activation="sigmoid"),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    return model


def biLSTM_Model(
    nodes, vocab_size, embedding_matrix, inputLength, glove_vec_len
):
    """Define Bi-LSTM model architecture."""
    model = keras.models.Sequential(
        [
            keras.layers.Embedding(
                vocab_size,
                glove_vec_len,
                weights=[embedding_matrix],
                input_length=inputLength,
            ),
            keras.layers.Bidirectional(keras.layers.LSTM(nodes)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(64, activation="sigmoid"),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    return model


def biGRU_Model(
    nodes, vocab_size, embedding_matrix, inputLength, glove_vec_len
):
    """Define Bi-GRU model architecture."""
    model = keras.models.Sequential(
        [
            keras.layers.Embedding(
                vocab_size,
                glove_vec_len,
                weights=[embedding_matrix],
                input_length=inputLength,
            ),
            keras.layers.Bidirectional(keras.layers.GRU(nodes)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(64, activation="sigmoid"),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    return model


def run_model(
    model_name,
    model_type,
    category_name,
    embedding_matrix,
    vocab_size,
    input_embedding,
    output_label,
    nodes,
    model_epochs,
    inputLength,
    glove_vec_len,
    save_model="no",
):
    """Run the model according to the architecture and type."""
    model_string = model_type + model_name
    if model_string == "SimpleRNN":
        print("Executing Simple RNN")
        model = RNN_Model(
            nodes, vocab_size, embedding_matrix, inputLength, glove_vec_len
        )
    elif model_string == "BiRNN":
        print("Executing Bidirectional RNN")
        model = biRNN_Model(
            nodes, vocab_size, embedding_matrix, inputLength, glove_vec_len
        )
    elif model_string == "SimpleLSTM":
        print("Executing Simple LSTM")
        model = LSTM_Model(
            nodes, vocab_size, embedding_matrix, inputLength, glove_vec_len
        )
    elif model_string == "BiLSTM":
        print("Executing Bidirectional LSTM")
        model = biLSTM_Model(
            nodes, vocab_size, embedding_matrix, inputLength, glove_vec_len
        )
    elif model_string == "SimpleGRU":
        print("Executing Simple GRU")
        model = GRU_Model(
            nodes, vocab_size, embedding_matrix, inputLength, glove_vec_len
        )
    elif model_string == "BiGRU":
        print("Executing Bidirectional GRU")
        model = biGRU_Model(
            nodes, vocab_size, embedding_matrix, inputLength, glove_vec_len
        )

    model.compile(
        loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    print(model.summary())
    history = model.fit(
        input_embedding,
        output_label,
        epochs=model_epochs,
        verbose=1,
        batch_size=256,
    )

    mod_param_path = os.path.join(
        model_name, (model_type + model_name), category_name
    )
    category_name = category_name + ".h5"
    if save_model == "yes":
        model.save(
            os.path.join("./main/models/", mod_param_path, category_name)
        )
        print(
            "Model Saved at ",
            os.path.join("./main/models/", mod_param_path, category_name),
        )
    elif save_model == "no":
        print("Model Not Saved")
    return model, history.history["accuracy"], history.history["loss"]


def test_routine(
    model_name,
    model_type,
    category_name,
    tokenizer,
    model,
    input_embedding,
    output_labels,
    seq_len,
    save_predict_file="no",
):
    """Run the test routine for specific model."""
    test_padded_docs = encoding(tokenizer, input_embedding, seq_len)
    predictions = model.predict(test_padded_docs, batch_size=256)
    binary_pred = [None] * (len(predictions))
    for index in range(len(predictions)):
        if predictions[index] > 0.50:
            binary_pred[index] = 1
        else:
            binary_pred[index] = 0
    binary_pred = np.asarray(binary_pred)

    mod_param_path = os.path.join(
        model_name, (model_type + model_name), category_name
    )

    category_name = category_name + "_Prediction.csv"

    if save_predict_file == "yes":
        predicted = pd.DataFrame(
            {
                "News": input_embedding,
                "Predicted": predictions.tolist(),
                "Predicted_bin": binary_pred,
                "True": output_labels,
            }
        )
        predicted.to_csv(
            os.path.join("./main/output/", mod_param_path, category_name),
            index=False,
            header=True,
        )
        print("Predicted File Saved")
    elif save_predict_file == "no":
        print("Predicted File Not Saved")

    return [
        precision_recall_fscore_support(
            binary_pred, output_labels, average="binary"
        ),
        accuracy_score(output_labels, binary_pred),
    ]


def get_results(
    model_type,
    model_name,
    categories,
    cat_idx,
    iterations,
    epochs,
    nodes,
    vec_len,
    sequence_len,
    news,
    labels,
    embedding_matrix,
    tokenizer,
    vocab_size,
    save_model,
    save_predict_file,
    save_results,
):
    """Compile and save the results."""
    category_name = categories[cat_idx]
    random_seed_list = [np.random.randint(1000) for i in range(0, iterations)]
    mod_string = model_type + model_name + "_" + category_name

    # Dynamically create variables (unsure if it's a safe practice)
    vars()[mod_string + "_acc"] = np.zeros([iterations, epochs])
    vars()[mod_string + "_loss"] = np.zeros([iterations, epochs])
    vars()[mod_string + "_out"] = np.zeros([iterations, 4])

    for iteration in range(0, iterations):
        X_train, X_test, y_train, y_test = train_test_split(
            news, labels, test_size=0.2, random_state=42
        )
        padded_news = encoding(tokenizer, X_train, sequence_len)

        print(f"{iteration + 1} out of {iterations}, iterations.")

        (
            vars()[mod_string + "_model_up"],
            vars()[mod_string + "_model_acc"],
            vars()[mod_string + "_model_loss"],
        ) = run_model(
            model_name,
            model_type,
            category_name,
            embedding_matrix,
            vocab_size,
            padded_news,
            y_train[:, cat_idx],
            nodes,
            epochs,
            sequence_len,
            vec_len,
            save_model=save_model,
        )

        vars()[mod_string + "_out_up"], accuracy = test_routine(
            model_name,
            model_type,
            category_name,
            tokenizer,
            vars()[mod_string + "_model_up"],
            X_test,
            y_test[:, cat_idx],
            sequence_len,
            save_predict_file,
        )

        vars()[mod_string + "_out"][iteration, :] = np.array(
            vars()[mod_string + "_out_up"]
        )
        vars()[mod_string + "_acc"][iteration, :] = np.array(
            vars()[mod_string + "_model_acc"]
        )
        vars()[mod_string + "_loss"][iteration, :] = np.array(
            vars()[mod_string + "_model_loss"]
        )

    mod_param_path = os.path.join(
        "./main/output/", model_name, (model_type + model_name), category_name
    )

    if save_results == "yes":
        with open(os.path.join(mod_param_path, "Scores.pkl"), "wb") as handle:
            pickle.dump(
                vars()[mod_string + "_out"],
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        with open(
            os.path.join(mod_param_path, "Accuracy.pkl"), "wb"
        ) as handle:
            pickle.dump(
                vars()[mod_string + "_acc"],
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        with open(os.path.join(mod_param_path, "Loss.pkl"), "wb") as handle:
            pickle.dump(
                vars()[mod_string + "_loss"],
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        with open(
            os.path.join(mod_param_path, "randomNumbers.pkl"), "wb"
        ) as handle:
            pickle.dump(
                random_seed_list, handle, protocol=pickle.HIGHEST_PROTOCOL
            )

    precision = np.mean(vars()[mod_string + "_out"][0:2], 0)[0]
    recall = np.mean(vars()[mod_string + "_out"][0:2], 0)[1]
    f1 = np.mean(vars()[mod_string + "_out"][0:2], 0)[2]
    accuracy = np.mean(vars()[mod_string + "_out"][0:2], 0)[3]
    details = (
        "\n\nThis file has been trained as follows:\n"
        + f"Category: {category_name}"
        + f"\nSize of Training Set: {len(X_train)}"
        + f"\nSize of Test Set: {len(X_test)}"
        + f"\nModel Name: {model_type + model_name}"
        + f"\nNodes: {nodes}"
        + f"\nSequence Length: {sequence_len}"
        + f"\nEpochs: {epochs}"
        + f"\nNumber of Iterations: {iterations}"
        + "\nPlease find the results of the model in the same folder.\n\n"
        + "Mean Scores are as follows:"
        + f"\nPrecision: {precision}"
        + f"\nRecall: {recall}"
        + f"\nF1 Score: {f1}"
        + f"\nAccuracy: {accuracy}"
        + f"\nPrecision: {precision}"
        + f"\nRecall: {recall}"
        + f"\nF1: {f1}"
    )

    print(details)
    if save_results == "yes":
        with open(
            os.path.join(mod_param_path, "Output.txt"), "w"
        ) as text_file:
            print(details, file=text_file)
    else:
        print("Results Not Saved")
    return [accuracy, precision, recall, f1]


def get_data_ready(data, args):
    """More granular control for data cleaning.

    If we do not want to clean the data in text representation step.
    """
    punctuation = """""!"#$%&()*+-/:;<=>?@[\\]^_`{|}~,.""" ""
    if args.remove_punct:
        data = data.apply(
            lambda x: "".join(ch for ch in x if ch not in set(punctuation))
        )
        data = data.str.lower()
    # remove numbers
    if args.remove_num:
        data = data.str.replace("[(0-9)]", " ")
    # remove redundant whitespaces, even though it"s highly unlikely
    # that the sst2 data has this issue, but it"s still best practice
    # to keep this as a sanity check as it"s not an computationally
    # expensive operation
    data = data.apply(lambda x: " ".join(x.split()))
    data = data.tolist()
    if args.remove_stopwords:
        stop_words = set(stopwords.words("english"))
        for i, val in enumerate(data):
            temp_sent = word_tokenize(val)
            filtered_sent = [w for w in temp_sent if w not in stop_words]
            data[i] = " ".join(filtered_sent)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-seq_len",
        required=False,
        default=32,
        help="Maximum sequence length of dataset",
        type=int,
    )
    parser.add_argument(
        "-itr",
        required=False,
        default=5,
        help="Number of iterations",
        type=int,
    )
    parser.add_argument(
        "-epochs", required=False, default=5, help="Number of epochs", type=int
    )
    parser.add_argument(
        "-nodes",
        required=False,
        default=100,
        help="Number of neurons in a layer",
        type=int,
    )
    parser.add_argument(
        "-vec_len",
        required=False,
        default=300,
        help="Vector Length, based on what GloVe embedding\
                        is used",
        type=int,
    )
    parser.add_argument(
        "-remove_punct",
        action="store_true",
        help="To remove punctuation",
        type=int,
    )
    parser.add_argument(
        "-remove_num",
        action="store_true",
        help="To remove \
                        numbers",
        type=int,
    )
    parser.add_argument(
        "-remove_stopwords",
        action="store_true",
        help="To remove stopwords",
        type=int,
    )
    args = parser.parse_args()

    data = pd.read_csv("./data/stanford_data.csv")
    data = shuffle(data)
    categories = ["pos", "neu", "neg"]

    sequence_length = args.seq_len
    iterations = args.itr
    epochs = args.epochs
    nodes = args.nodes
    vec_len = args.vec_len

    architecture_name = ["RNN", "LSTM", "GRU"]
    architecture_type = ["Simple", "Bi"]

    news = get_data_ready(data["sentences"], args)
    # The idea here is to demonstrate how to convert multi-class classification
    # into a binary classification, so how do we do that?
    # We create three column labels in our dataset, namely "pos", "neg", "neu"
    # If a sentence has positive sentiment, "pos" will have a value of 1,
    # whereas "neg" and "neu" will have a value of 0. Of-course we can just
    # take one column with labels and then do a corresponding mapping to create
    # the same but I found this way to be more explicit.

    labels = np.array(data[data.columns[2:5]])
    embedding_matrix, tokenizer, vocab_size = load_glove(vec_len)
    main_list = []

    for model_type in architecture_type:
        for model_name in architecture_name:
            for cat_idx in range(0, 3):
                for i in range(0, 11):
                    # A brute force way to do a k-fold run, basically
                    # we are running our models 11 times for each
                    # binary classification task.
                    [accuracy, precision, recall, f1] = get_results(
                        model_type,
                        model_name,
                        categories,
                        cat_idx,
                        iterations,
                        epochs,
                        nodes,
                        vec_len,
                        sequence_length,
                        news,
                        labels,
                        embedding_matrix,
                        tokenizer,
                        vocab_size,
                        "yes",
                        "yes",
                        "yes",
                    )
                    temp_ls = [
                        model_type + model_name,
                        i,
                        categories[cat_idx],
                        accuracy,
                        precision,
                        recall,
                        f1,
                    ]
                    main_list.append(temp_ls)

    output_df = pd.DataFrame(
        main_list,
        columns=[
            "Model",
            "Iteration",
            "Category",
            "Accuracy",
            "Precision",
            "Recall",
            "F1_Score",
        ],
    )
    output_df.to_csv("./output/GloVe_based_results.csv")
