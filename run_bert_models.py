#!/usr/bin/env python3
# coding: utf-8


import re
import sys
import random
import time
import datetime
from typing import Dict, Any, Tuple, DefaultDict, List
import numpy as np
import pandas as pd
from collections import defaultdict
from nltk.tokenize import word_tokenize
from transformers import (
    DistilBertTokenizer,
    BERTSentimentClassifier,
    RobertaSentimentClassifier,
    AutoConfig,
    AutoModel,
    AdamW,
    get_linear_schedule_with_warmup,
)
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_and_seed() -> pd.DataFrame:
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    statement_df = pd.read_csv(
        "./data/stanford_data.csv", usecols=["sentences", "labels"]
    )
    return statement_df


def clean_statements(statement: str) -> str:
    statement = re.sub(" '", "'", statement)
    statement = re.sub(" 's", "'s", statement)
    statement = re.sub("\( ", "(", statement)
    statement = re.sub(" \)", ")", statement)
    statement = re.sub("``", '"', statement)
    statement = re.sub("''", '"', statement)
    statement = re.sub(r'\s([?.,%:!"](?:\s|$))', r"\1", statement)
    return statement


def get_ngrams(input: list, n: int) -> list:
    return [tuple(input[i : i + n]) for i in range(len(input) - n + 1)]


def format_time(elapsed: float) -> str:
    return str(datetime.timedelta(seconds=int(round((elapsed)))))


class StatementDataset(Dataset):
    def __init__(
        self,
        statements: list,
        labels: list,
        tokenizer: DistilBertTokenizer,
        max_length: int,
    ) -> None:
        self.statements = statements
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.statements)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        statement = str(self.statements[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            statement,
            max_length=self.max_length,
            padding="max_length",
            add_special_tokens=True,
            return_token_type_ids=False,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "statement_text": statement,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


class DistilBertForSequenceClassification(nn.Module):
    def __init__(self, pretrained_model_name, num_classes):
        super().__init__()

        config = AutoConfig.from_pretrained(
            pretrained_model_name, num_labels=num_classes, return_dict=False
        )

        self.distilbert = AutoModel.from_pretrained(
            pretrained_model_name, config=config
        )
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        self.classifier = nn.Linear(config.dim, num_classes)
        self.dropout = nn.Dropout(config.seq_classif_dropout)

    def forward(self, input_ids, attention_mask=None, head_mask=None):
        assert attention_mask is not None, "No Attention Mask"
        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
        )

        hidden_state = distilbert_output[0]
        pooled_output = hidden_state[:, 0]
        pooled_output = self.pre_classifier(pooled_output)
        pooled_output = nn.ReLU()(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits


def create_dataset(
    df: pd.DataFrame, tokenizer: object, max_length: int
) -> object:
    ds = StatementDataset(
        statements=df["sentences"].to_numpy(),
        labels=df["labels"].to_numpy(),
        tokenizer=tokenizer,
        max_length=max_length,
    )
    return ds


def create_dataloader(ds: object, batch_size: int) -> DataLoader:
    return DataLoader(ds, batch_size, num_workers=4)


def cv_ensemble_performance(preds: np.ndarray, labels: np.ndarray) -> None:
    preds = np.array(preds)
    summed = np.sum(preds, axis=0)
    preds = np.argmax(summed, axis=1)
    print(confusion_matrix(y_true=labels, y_pred=preds))
    print(
        classification_report(
            y_true=labels, y_pred=preds, digits=3, target_names=target_strings
        )
    )


def single_model_performance(preds: np.ndarray, labels: np.ndarray) -> None:
    print(confusion_matrix(y_true=labels, y_pred=preds))
    print(
        classification_report(
            labels, preds, digits=4, target_names=target_strings
        )
    )


def train_model(
    model: torch.nn.Module,
    device: torch.device,
    data_loader: torch.utils.data.DataLoader,
    loss_function: torch.nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    n_examples: int,
) -> Tuple[float, float]:
    model = model.train()
    losses = []
    correct_preds = 0
    complete_preds = []
    complete_labels = []

    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        _, preds = torch.max(outputs, dim=1)
        loss = loss_function(outputs, labels)
        complete_preds.append(preds.data.cpu().numpy().tolist())
        complete_labels.append(labels.data.cpu().numpy().tolist())
        correct_preds += torch.sum(preds == labels)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    complete_preds_flat = [x for y in complete_preds for x in y]
    complete_labels_flat = [x for y in complete_labels for x in y]
    acc_score = accuracy_score(
        y_true=complete_labels_flat, y_pred=complete_preds_flat
    )
    return acc_score, np.mean(losses)


def eval_model(
    model: object,
    device: str,
    data_loader: DataLoader,
    loss_function: object,
    n_examples: int,
) -> tuple:
    model = model.eval()

    losses = []
    correct_preds = 0
    complete_preds = []
    complete_labels = []
    complete_outputs = []

    with torch.no_grad():
        for item in data_loader:
            input_ids = item["input_ids"].to(device)
            attention_mask = item["attention_mask"].to(device)
            labels = item["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            _, preds = torch.max(outputs, dim=1)
            loss = loss_function(outputs, labels)

            correct_preds += torch.sum(preds == labels)
            complete_preds.append(preds.data.cpu().numpy().tolist())
            complete_labels.append(labels.data.cpu().numpy().tolist())
            complete_outputs.append(outputs.tolist())
            losses.append(loss.item())

        complete_preds_flat = [x for y in complete_preds for x in y]
        complete_labels_flat = [x for y in complete_labels for x in y]
        complete_outputs_flat = [x for y in complete_outputs for x in y]

        acc_score = accuracy_score(
            y_true=complete_labels_flat, y_pred=complete_preds_flat
        )

        return_items = (
            acc_score,
            np.mean(losses),
            complete_preds_flat,
            complete_outputs_flat,
        )

        return return_items


def train_fold(
    epochs: int,
    model: torch.nn.Module,
    device: torch.device,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    model_save_name: str,
    n_train: int,
    n_val: int,
    single_model: bool = True,
) -> Tuple[DefaultDict[str, List[float]], torch.Tensor, torch.Tensor]:
    start_time = time.time()
    history: DefaultDict[str, List[float]] = defaultdict(list)
    best_accuracy: float = 0.0

    for epoch in range(epochs):
        epoch_start_time = time.time()
        print("Epoch ", epoch + 1, "/", epochs)
        print("-" * 50)

        training_output = train_model(
            model,
            device,
            train_dataloader,
            loss_fn,
            optimizer,
            scheduler,
            n_train,
        )

        train_acc, train_loss = training_output

        val_output = eval_model(model, device, val_dataloader, loss_fn, n_val)

        val_acc, val_loss, val_preds, val_outputs = val_output

        history["train_accuracy"].append(train_acc)
        history["train_loss"].append(train_loss)
        history["val_accuracy"].append(val_acc)
        history["val_loss"].append(val_loss)
        history["val_preds"].append(val_preds)

        if val_acc > best_accuracy:
            torch.save(model.state_dict(), model_save_name)
            best_accuracy = val_acc
            best_preds = val_preds
            best_outputs = val_outputs

        print("Train Loss: ", train_loss, " | ", "Train Accuracy: ", train_acc)
        print("Val Loss: ", val_loss, " | ", "Val Accuracy: ", val_acc)
        print(
            "Epoch Train Time: ", format_time(time.time() - epoch_start_time)
        )
        print("\n")

    print("Finished Training.")
    print("Fold Train Time: ", format_time(time.time() - start_time))
    print("\n")
    if single_model:
        _, _, test_preds, test_outputs = eval_model(
            model, device, test_dataloader, loss_function, len(df_test)
        )

        single_model_performance(test_preds, df_test["labels"].values)
    return history, best_preds, best_outputs


def get_oof_and_test_preds(
    model_type: str,
    tokenizer: object,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    single_model: bool = False,
) -> Tuple[List[dict], List[torch.Tensor]]:
    oof_preds = []
    oof_outputs = []
    oof_preds_indices = []
    test_preds_list = []
    test_outputs_list = []
    history_list = []
    start_time = time.time()

    fold = 0

    x_train = train_df["sentences"]
    y_train = train_df["labels"]

    for train_index, val_index in skf.split(x_train, y_train):
        print("Fold: {}".format(fold + 1))

        x_tr = x_train.iloc[train_index]
        y_tr = y_train.iloc[train_index]
        x_va = x_train.iloc[val_index]
        y_va = y_train.iloc[val_index]

        train = pd.DataFrame(
            list(zip(x_tr, y_tr)), columns=["sentences", "labels"]
        )
        val = pd.DataFrame(
            list(zip(x_va, y_va)), columns=["sentences", "labels"]
        )

        train_ds = create_dataset(train, tokenizer, MAX_LENGTH)
        val_ds = create_dataset(val, tokenizer, MAX_LENGTH)
        test_ds = create_dataset(test_df, tokenizer, MAX_LENGTH)

        if model_type == "bert":
            model = BERTSentimentClassifier(NUM_CLASSES)
            model = model.to(device)
        elif model_type == "distilbert":
            model = DistilBertForSequenceClassification(
                pretrained_model_name=DISTILBERT_MODEL_NAME,
                num_classes=NUM_CLASSES,
            )
            model = model.to(device)
        elif model_type == "roberta":
            model = RobertaSentimentClassifier(n_classes=NUM_CLASSES)
            model = model.to(device)

        train_loader = create_dataloader(train_ds, BATCH_SIZE)
        val_loader = create_dataloader(val_ds, BATCH_SIZE)
        test_loader = create_dataloader(test_ds, BATCH_SIZE)

        training_steps = len(train_loader.dataset) * EPOCHS
        warmup_steps = int(0.3 * training_steps)
        optimizer = AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            correct_bias=True,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=training_steps,
        )

        model_save_name = "{}_fold_{}.bin".format(model_type, fold)

        history, preds, outputs = train_fold(
            epochs=EPOCHS,
            model=model,
            device=device,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            test_dataloader=test_loader,
            loss_fn=loss_function,
            optimizer=optimizer,
            scheduler=scheduler,
            model_save_name=model_save_name,
            n_train=len(train),
            n_val=len(val),
            single_model=False,
        )

        history_list.append(history)
        oof_preds.append(preds)
        oof_outputs.append(outputs)
        oof_preds_indices.append(val_index)
        _, _, test_preds, test_outputs = eval_model(
            model, device, test_loader, loss_function, len(test_df)
        )
        test_preds_list.append(test_preds)
        test_outputs_list.append(test_outputs)

        fold += 1

    print(
        str(NFOLDS),
        "Fold CV Train Time: ",
        format_time(time.time() - start_time),
    )

    return history_list, test_outputs_list


statement_df["sentences"] = statement_df["sentences"].apply(clean_statements)
statement_df["num_char"] = statement_df["sentences"].apply(len)
statement_df["num_words"] = statement_df["sentences"].apply(
    lambda x: len(x.split())
)


pos_statements = " ".join(
    statement_df.loc[statement_df["labels"] == "pos"]["sentences"].values
)
neg_statements = " ".join(
    statement_df.loc[statement_df["labels"] == "neg"]["sentences"].values
)
neu_statements = " ".join(
    statement_df.loc[statement_df["labels"] == "neu"]["sentences"].values
)


pos_statements = re.sub("[^A-Za-z]+", " ", pos_statements).strip().lower()
neg_statements = re.sub("[^A-Za-z]+", " ", neg_statements).strip().lower()
neu_statements = re.sub("[^A-Za-z]+", " ", neu_statements).strip().lower()

pos_tokens = word_tokenize(pos_statements)
neg_tokens = word_tokenize(neg_statements)
neutral_tokens = word_tokenize(neu_statements)


MAX_LENGTH = 32
BATCH_SIZE = 16
NUM_CLASSES = 3
EPOCHS = 10
DROPOUT_PROB = float(sys.argv[1])
WEIGHT_DECAY = float(sys.argv[2])
NFOLDS = 5
LEARNING_RATE = float(sys.argv[3])

le = LabelEncoder()
statement_df["labels"] = le.fit_transform(statement_df["labels"])
target_strings = le.classes_


df_train, df_test = train_test_split(
    statement_df,
    test_size=0.2,
    random_state=42,
    stratify=statement_df["labels"].values,
)

df_val, df_test = train_test_split(
    df_test, test_size=0.5, random_state=42, stratify=df_test["labels"].values
)

df_train_full = pd.concat([df_train, df_val])


loss_function = nn.CrossEntropyLoss().to(device)

skf = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=42)


DISTILBERT_MODEL_NAME = "distilbert-base-uncased"
distilbert_tokenizer = DistilBertTokenizer.from_pretrained(
    DISTILBERT_MODEL_NAME, return_dict=False
)
distilbert_train_ds = create_dataset(
    df_train, distilbert_tokenizer, MAX_LENGTH
)
distilbert_test_ds = create_dataset(df_test, distilbert_tokenizer, MAX_LENGTH)
distilbert_val_ds = create_dataset(df_val, distilbert_tokenizer, MAX_LENGTH)

distilbert_train_dataloader = create_dataloader(
    distilbert_train_ds, BATCH_SIZE
)
distilbert_test_dataloader = create_dataloader(distilbert_test_ds, BATCH_SIZE)
distilbert_val_dataloader = create_dataloader(distilbert_val_ds, BATCH_SIZE)


distilbert_model = DistilBertForSequenceClassification(
    pretrained_model_name=DISTILBERT_MODEL_NAME, num_classes=NUM_CLASSES
)
distilbert_model = distilbert_model.to(device)

training_steps = len(distilbert_train_dataloader.dataset) * EPOCHS

distilbert_optimizer = AdamW(
    distilbert_model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    correct_bias=True,
)
distilbert_scheduler = get_linear_schedule_with_warmup(
    distilbert_optimizer,
    num_warmup_steps=int(0.1 * training_steps),
    num_training_steps=training_steps,
)

(
    distilbert_history,
    distilbert_preds,
    distilbert_outputs,
) = train_fold(
    epochs=EPOCHS,
    model=distilbert_model,
    device=device,
    train_dataloader=distilbert_train_dataloader,
    val_dataloader=distilbert_val_dataloader,
    test_dataloader=distilbert_test_dataloader,
    loss_fn=loss_function,
    optimizer=distilbert_optimizer,
    scheduler=distilbert_scheduler,
    model_save_name="distilbest_best_model.bin",
    n_train=len(df_train),
    n_val=len(df_val),
    single_model=True,
)


distilbert_history, distilbert_test_outputs = get_oof_and_test_preds(
    model_type="distilbert",
    tokenizer=distilbert_tokenizer,
    train_df=df_train_full,
    test_df=df_test,
    single_model=False,
)

cv_ensemble_performance(distilbert_test_outputs, df_test["labels"].values)
