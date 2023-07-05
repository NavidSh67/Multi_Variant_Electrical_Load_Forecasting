
"""
Multi-Variant Electiricty Prediction
Using LSTM

@author: Navid Shirzadi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.models import load_model
from sklearn.metrics import mean_squared_error
import math


class ElectricityConsumptionPredictor:
    def __init__(self, data_file):
        self.df = pd.read_csv(data_file)
        self.df.dropna(inplace=True)
        self.training_set = None
        self.test_set = None
        self.training_set_scaled = None
        self.test_set_scaled = None
        self.X_train = None
        self.y_train = None
        self.model = None

    def preprocess_data(self):
        sn.heatmap(self.df.corr())
        self.training_set = self.df.iloc[:8712, 1:4].values
        self.test_set = self.df.iloc[8712:, 1:4].values

        sc = MinMaxScaler(feature_range=(0, 1))
        self.training_set_scaled = sc.fit_transform(self.training_set)
        self.test_set_scaled = sc.fit_transform(self.test_set)
        self.test_set_scaled = self.test_set_scaled[:, 0:2]

    def create_sequences(self, window_size=24):
        self.X_train = []
        self.y_train = []
        for i in range(window_size, len(self.training_set_scaled)):
            self.X_train.append(self.training_set_scaled[i - window_size:i, 0:3])
            self.y_train.append(self.training_set_scaled[i, 2])

        self.X_train, self.y_train = np.array(self.X_train), np.array(self.y_train)
        self.X_train = np.reshape(self.X_train, (self.X_train.shape[0], self.X_train.shape[1], 3))

    def build_model(self):
        self.model = Sequential()
        self.model.add(LSTM(units=70, return_sequences=True, input_shape=(self.X_train.shape[1], 3)))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=70, return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=70, return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=70))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(units=1))
        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def train_model(self, epochs=80, batch_size=32):
        self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size)

    def save_model(self, file_name):
        self.model.save(file_name)

    def load_model(self, file_name):
        self.model = load_model(file_name)

    def predict(self):
        prediction_test = []
        batch_one = self.training_set_scaled[-24:]
        batch_new = batch_one.reshape((1, 24, 3))

        for i in range(48):
            first_pred = self.model.predict(batch_new)[0]
            prediction_test.append(first_pred)

            new_var = self.test_set_scaled[i, :]
            new_var = new_var.reshape(1, 2)
            new_test = np.insert(new_var, 2, [first_pred], axis=1)
            new_test = new_test.reshape(1, 1, 3)
            batch_new = np.append(batch_new[:, 1:, :], new_test, axis=1)

        prediction_test = np.array(prediction_test)
        return prediction
