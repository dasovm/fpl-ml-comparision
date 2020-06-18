import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from tensorflow.keras.regularizers import l2

from custom_lstm import CondLSTM

tf.keras.backend.set_floatx('float64')

y_col = "total_points"
point_separator = 4


class CondLSTMModel(Model):
    def __init__(self, time_steps, amount_of_features, n_neurons=64, dropout_rate=0.5, n_hidden_layers=1, n_neurons_last_layer=128, name=None, using_classes=False):
        super(CondLSTMModel, self).__init__(name=name)
        self.n_hidden_layers = n_hidden_layers
        self.masking = Masking(mask_value=0, input_shape=(
            time_steps, amount_of_features), dtype='float32')
        self.lstms = []
        self.dropouts = []
        for _ in range(0, n_hidden_layers):
            self.lstms.append(CondLSTM(
                n_neurons, return_sequences=True, kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001)))
            self.dropouts.append(Dropout(dropout_rate))
        self.last_cond = CondLSTM(
            n_neurons, return_sequences=False, kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001))
        self.last_dropout = Dropout(dropout_rate)
        self.dense = Dense(n_neurons_last_layer, activation='relu',
                           kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001))
        self.out = Dense(
            units=1, activation='sigmoid' if using_classes else 'relu')

    def call(self, inputs, **kwargs):
        o, cond = inputs
        self.masking(o)

        for i in range(0, self.n_hidden_layers):
            o = self.lstms[i]([o, cond])
            o = self.dropouts[i](o)

        o = self.last_cond([o, cond])
        o = self.last_dropout(o)
        o = self.dense(o)
        o = self.out(o)
        return o


def generate_data(time_steps, amount_of_targets, using_classes=False):
    amount_of_data = 100

    X = np.random.uniform(0, 10, size=(amount_of_data, time_steps, 5))
    targets = np.random.uniform(0, 1, size=(amount_of_data, amount_of_targets))
    y = np.random.randint(0, 10, size=(amount_of_data,)).astype("float")
    if using_classes:
        y = get_classes_from_y(y)

    return X, targets, y


def split_df_to_train_test(X, targets, y, split_rate=0.8):
    amount_of_data = X.shape[0]

    split_level = int(amount_of_data * split_rate)

    return X[:split_level], targets[:split_level], y[:split_level], X[split_level:], targets[split_level:], y[split_level:]


def get_classes_from_y(y):
    return (y > point_separator).astype(int)
