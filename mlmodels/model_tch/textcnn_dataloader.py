# coding: utf-8
"""


"""

import os
import json
import shutil
from pathlib import Path
from time import sleep
import re

import pandas as pd
from numpy.random import RandomState

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import torchtext
from torchtext.data import Field
from torchtext.data import TabularDataset
from torchtext.vocab import GloVe
from torchtext.data import Iterator, BucketIterator
import torchtext.datasets
from dataloader import PyTorchDataLoader
import spacy


####################################################################################################
import mlmodels.models as M

from mlmodels.util import os_package_root_path, log, path_norm, get_model_uri

VERBOSE = False
MODEL_URI = get_model_uri(__file__)


###########################################################################################################
###########################################################################################################
def _train(m, device, train_itr, optimizer, epoch, max_epoch):
    m.train()
    corrects, train_loss = 0.0, 0
    for batch in train_itr:
        text, target = batch.text, batch.label
        text = torch.transpose(text, 0, 1)
        target.data.sub_(1)
        text, target = text.to(device), target.to(device)
        optimizer.zero_grad()
        logit = m(text)

        loss = F.cross_entropy(logit, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        result = torch.max(logit, 1)[1]
        corrects += (result.view(target.size()).data == target.data).sum()

    size = len(train_itr.dataset)
    train_loss /= size
    accuracy = 100.0 * corrects / size

    return train_loss, accuracy


def _valid(m, device, test_itr):
    m.eval()
    corrects, test_loss = 0.0, 0
    for batch in test_itr:
        text, target = batch.text, batch.label
        text = torch.transpose(text, 0, 1)
        target.data.sub_(1)
        text, target = text.to(device), target.to(device)

        logit = m(text)
        loss = F.cross_entropy(logit, target)

        test_loss += loss.item()
        result = torch.max(logit, 1)[1]
        corrects += (result.view(target.size()).data == target.data).sum()

    size = len(test_itr.dataset)
    test_loss /= size
    accuracy = 100.0 * corrects / size

    return test_loss, accuracy


def _get_device():
    # use GPU if it is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # return device
    return "cpu"


def get_config_file():
    return os.path.join(
        os_package_root_path(__file__, 1), "config", "model_tch", "datatextcnn.json"
    )


def get_data_file():
    return os.path.join(os_package_root_path(__file__), "dataset", "IMDB_Dataset.txt")


###########################################################################################################
###########################################################################################################
class TextCNN(nn.Module):
    def __init__(self, model_pars=None, data_pars=None, compute_pars=None, **kwargs):
        kernel_wins = [int(x) for x in model_pars["kernel_height"]]
        super(TextCNN, self).__init__()

        # load pretrained embedding in embedding layer.
        emb_dim = model_pars.get("emb_dim", 300)
        self.emb_size = model_pars.get("emb_size", 20000)
        vocab = get_dataset(data_pars)[0].dataset.fields["text"].vocab
        # emb_dim = 300
        self.embed = nn.Embedding(len(vocab), emb_dim)
        self.embed.weight.data.copy_(vocab.vectors)

        # Convolutional Layers with different window size kernels
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, model_pars["dim_channel"], (w, emb_dim)) for w in kernel_wins]
        )
        # Dropout layer
        self.dropout = nn.Dropout(model_pars["dropout_rate"])

        # FC layer
        self.fc = nn.Linear(
            len(kernel_wins) * model_pars["dim_channel"], model_pars["num_class"]
        )

    def forward(self, x):
        emb_x = self.embed(x)
        emb_x = emb_x.unsqueeze(1)
        con_x = [conv(emb_x) for conv in self.convs]
        pool_x = [F.max_pool1d(x.squeeze(-1), x.size()[2]) for x in con_x]
        fc_x = torch.cat(pool_x, dim=1)
        fc_x = fc_x.squeeze(-1)
        fc_x = self.dropout(fc_x)
        logit = self.fc(fc_x)
        return logit

    def rebuild_embed(self, vocab_built):
        self.embed = nn.Embedding(*vocab_built.vectors.shape)
        self.embed.weight.data.copy_(vocab_built.vectors)


# Model = TextCNN
class Model:
    def __init__(self, model_pars=None, data_pars=None, compute_pars=None):
        ### Model Structure        ################################
        if model_pars is None:
            self.model = None
            return None

        self.model = TextCNN(model_pars, data_pars, compute_pars)


def fit_metrics(model, data_pars=None, compute_pars=None, out_pars=None, **kwargs):
    # return metrics on full dataset
    device = _get_device()
    data_pars = data_pars.copy()
    # data_pars.update(frac=1)
    test_iter, _ = get_dataset(data_pars, out_pars)
    model.model.rebuild_embed(test_iter.dataset.fields["text"].vocab)

    return _valid(model.model, device, test_iter)


def fit(model, data_pars=None, compute_pars=None, out_pars=None, **kwargs):
    model0 = model.model
    lr = compute_pars["learning_rate"]
    epochs = compute_pars["epochs"]
    device = _get_device()
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    best_test_acc = -1
    optimizer = optim.Adam(model0.parameters(), lr=lr)
    train_iter, valid_iter = get_dataset(data_pars, out_pars)
    vocab = train_iter.dataset.fields["text"].vocab
    # load word embeddings to model
    model0.rebuild_embed(vocab)

    for epoch in range(1, epochs + 1):

        tr_loss, tr_acc = _train(model0, device, train_iter, optimizer, epoch, epochs)
        print(f"Train Epoch: {epoch} \t Loss: {tr_loss} \t Accuracy: {tr_acc}")

        ts_loss, ts_acc = _valid(model0, device, valid_iter)
        print(f"Train Epoch: {epoch} \t Loss: {ts_loss} \t Accuracy: {ts_acc}")

        if ts_acc > best_test_acc:
            best_test_acc = ts_acc
            # save paras(snapshot)
            log(f"model saves at {best_test_acc}% accuracy")
            os.makedirs(out_pars["checkpointdir"], exist_ok=True)
            torch.save(
                model0.state_dict(),
                os.path.join(out_pars["checkpointdir"], "best_accuracy"),
            )

        train_loss.append(tr_loss)
        train_acc.append(tr_acc)
        test_loss.append(ts_loss)
        test_acc.append(ts_acc)

    model.model = model0
    return model, None


def get_dataset(
    data_pars=None, out_pars=None, return_preprocessor_function=False, **kwargs
):
    loader = PyTorchDataLoader(**data_pars)
    return (
        (loader.get_data(), loader.preprocessor)
        if return_preprocessor_function
        else loader.get_data()
    )


def predict(model, session=None, data_pars=None, compute_pars=None, out_pars=None):
    data_pars = data_pars.copy()
    test_iter, _ = get_dataset(data_pars, out_pars)
    # get a batch of data
    x_test = next(iter(test_iter)).text
    model0 = model.model
    ypred = model0(x_test).detach().numpy()
    return ypred


def save(model, session=None, save_pars=None):
    from mlmodels.util import save_tch

    save_tch(model, save_pars=save_pars)
    # return torch.save(model, path)


def load(load_pars=None):
    from mlmodels.util import load_tch

    return load_tch(load_pars), None
    # return torch.load(path)


def get_params(param_pars=None, **kw):
    import json

    pp = param_pars
    choice = pp["choice"]
    config_mode = pp["config_mode"]
    data_path = pp["data_path"]

    if choice == "json":
        data_path = path_norm(data_path)
        cf = json.load(open(data_path, "rb"))
        cf = cf[config_mode]
        return cf["model_pars"], cf["data_pars"], cf["compute_pars"], cf["out_pars"]

    if choice == "test01":
        log("#### Path params   ##########################################")
        data_path = path_norm("dataset/text/imdb.csv")
        out_path = path_norm("ztest/model_tch/textcnn/")
        model_path = os.path.join(out_path, "model")

        data_pars = {
            "data_path": path_norm("dataset/recommender/IMDB_sample.txt"),
            "train_path": path_norm("dataset/recommender/IMDB_train.csv"),
            "valid_path": path_norm("dataset/recommender/IMDB_valid.csv"),
            "split_if_exists": True,
            "frac": 0.99,
            "lang": "en",
            "pretrained_emb": "glove.6B.300d",
            "batch_size": 64,
            "val_batch_size": 64,
        }

        model_pars = {
            "dim_channel": 100,
            "kernel_height": [3, 4, 5],
            "dropout_rate": 0.5,
            "num_class": 2,
        }

        compute_pars = {
            "learning_rate": 0.001,
            "epochs": 1,
            "checkpointdir": out_path + "/checkpoint/",
        }

        out_pars = {"path": model_path, "checkpointdir": out_path + "/checkpoint/"}

        return model_pars, data_pars, compute_pars, out_pars


###########################################################################################################
###########################################################################################################
def test(data_path="dataset/", pars_choice="json", config_mode="test"):
    ### Local test
    from mlmodels.util import path_norm

    data_path = path_norm(data_path)

    log("#### Loading params   ##############################################")
    param_pars = {
        "choice": pars_choice,
        "data_path": data_path,
        "config_mode": config_mode,
    }
    model_pars, data_pars, compute_pars, out_pars = get_params(param_pars)
    print(out_pars)

    log("#### Loading dataset   #############################################")
    Xtuple = get_dataset(data_pars)
    print(len(Xtuple))

    log("#### Model init, fit   #############################################")
    session = None
    model = Model(model_pars, data_pars, compute_pars)
    model, session = fit(model, session, data_pars, compute_pars, out_pars)

    log("#### Save/Load   ###################################################")
    save_pars = {"path": out_pars["model_path"] + "/model.pkl"}
    save(model, session, save_pars=save_pars)
    model2, session2 = load(save_pars)
    ypred = predict(model2, session2, data_pars, compute_pars, out_pars)

    log("#### Predict   #####################################################")
    data_pars["train"] = 0
    ypred = predict(model, session, data_pars, compute_pars, out_pars)

    log("#### metrics   #####################################################")
    metrics_val = fit_metrics(model, session, data_pars, compute_pars, out_pars)
    print(metrics_val)

    log("#### Plot   ########################################################")


if __name__ == "__main__":
    test(data_path="dataset/json_/textcnn_dataloader.json", pars_choice="test01")
