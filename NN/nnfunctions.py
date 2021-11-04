"""
Neural network functions for scene segmentation. Based off of 

"""
from keras.backend.cntk_backend import clear_session
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from proglearn.deciders import SimpleArgmaxAverage
from proglearn.progressive_learner import ProgressiveLearner
from proglearn.transformers import (
    NeuralClassificationTransformer,
)
from proglearn.voters import KNNClassificationVoter

from proglearn.network import LifelongClassificationNetwork

from tensorflow.keras.backend import clear_session

def run_fte_bte_exp(data_x, data_y, which_task, slot, shift=0):

    df_total = []

    unique_y = np.unique(data_y)
    train_x, train_y, test_x, test_y = cross_val_data(data_x, data_y, shift, slot, total_cls=len(unique_y))

    clear_session()  # clear GPU memory before each run, to avoid OOM error

    default_transformer_class = NeuralClassificationTransformer

    network = make_odin()

    default_transformer_kwargs = {
        "network": network,
        "euclidean_layer_idx": -2,
        "loss": "categorical_crossentropy",
        "optimizer": Adam(3e-4),
        "fit_kwargs": {
            "epochs": 100,
            "callbacks": [EarlyStopping(patience=5, monitor="val_loss")],
            "verbose": False,
            "validation_split": 0.33,
            "batch_size": 32,
        },
    }
    default_voter_class = KNNClassificationVoter
    default_voter_kwargs = {"k": int(np.log2(300))}
    default_decider_class = SimpleArgmaxAverage

    p_learner = ProgressiveLearner(
        default_transformer_class=default_transformer_class,
        default_transformer_kwargs=default_transformer_kwargs,
        default_voter_class=default_voter_class,
        default_voter_kwargs=default_voter_kwargs,
        default_decider_class=default_decider_class,
    )

    df = experiment(
        train_x,
        train_y,
        test_x,
        test_y,
        shift,
        slot,
        p_learner,
        which_task,
    )

    df_total.append(df)

    return df_total

def make_odin():
    network = keras.Sequential()
    network.add(
        layers.Conv2D(
            filters=16,
            kernel_size=(3, 3),
            activation="relu",
            input_shape=np.shape(data_x)[1:],
        )
    )
    network.add(layers.BatchNormalization())
    network.add(
        layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=2,
            padding="same",
            activation="relu",
        )
    )
    network.add(layers.BatchNormalization())
    network.add(
        layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=2,
            padding="same",
            activation="relu",
        )
    )
    network.add(layers.BatchNormalization())
    network.add(
        layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            strides=2,
            padding="same",
            activation="relu",
        )
    )
    network.add(layers.BatchNormalization())
    network.add(
        layers.Conv2D(
            filters=254,
            kernel_size=(3, 3),
            strides=2,
            padding="same",
            activation="relu",
        )
    )

    network.add(layers.Flatten())
    network.add(layers.BatchNormalization())
    network.add(layers.Dense(2000, activation="relu"))
    network.add(layers.BatchNormalization())
    network.add(layers.Dense(2000, activation="relu"))
    network.add(layers.BatchNormalization())
    network.add(layers.Dense(units=10, activation="softmax"))
    return network

def cross_val_data(data_x, data_y, shift, slot, total_cls=100):
    # Creates copies of both data_x and data_y so that they can be modified without affecting the original sets
    x = data_x.copy()
    y = data_y.copy()
    # Creates a sorted array of arrays that each contain the indices at which each unique element of data_y can be found
    idx = [np.where(data_y == u)[0] for u in np.unique(data_y)]

    for i in range(total_cls):
        # Chooses the i'th array within the larger idx array
        indx = idx[i]

        # 30 available training data points per class
        # Chooses all samples other than those in the testing batch
        tmp_x = np.concatenate(
            (x[indx[0 : (shift * 10)], :], x[indx[((shift + 1) * 10) : 40], :]),
            axis=0,
        )
        tmp_y = np.concatenate(
            (y[indx[0 : (shift * 10)]], y[indx[((shift + 1) * 10) : 40]]), axis=0
        )

        if i == 0:
            # 30 training data points per class
            # Rotates which set of 30 samples from each class is chosen for training each task
            # With 10 classes per task, total of 300 training samples per task
            train_x = tmp_x[(slot * 30) : ((slot + 1) * 30)]
            train_y = tmp_y[(slot * 30) : ((slot + 1) * 30)]

            # 10 testing data points per class
            # Batch for testing set is rotated each time
            test_x = x[indx[(shift * 10) : ((shift + 1) * 10)], :]
            test_y = y[indx[(shift * 10) : ((shift + 1) * 10)]]
        else:
            # 30 training data points per class
            # Rotates which set of 30 samples from each class is chosen for training each task
            # With 10 classes per task, total of 300 training samples per task
            train_x = np.concatenate(
                (train_x, tmp_x[(slot * 30) : ((slot + 1) * 30)]), axis=0
            )
            train_y = np.concatenate(
                (train_y, tmp_y[(slot * 30) : ((slot + 1) * 30)]), axis=0
            )

            # 10 testing data points per class
            # Batch for testing set is rotated each time
            test_x = np.concatenate(
                (test_x, x[indx[(shift * 10) : ((shift + 1) * 10)], :]), axis=0
            )
            test_y = np.concatenate(
                (test_y, y[indx[(shift * 10) : ((shift + 1) * 10)]]), axis=0
            )

    return train_x, train_y, test_x, test_y

def experiment(X_train, y_train, )