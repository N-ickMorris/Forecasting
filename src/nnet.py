# -*- coding: utf-8 -*-
"""
Stream data through a neural network model to produce a rolling forecast

@author: Nick
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import TimeSeriesSplit
from keras import Sequential, layers, optimizers, regularizers
from keras.wrappers.scikit_learn import KerasRegressor
from forecast import Forecasting

N_JOBS = -1  # The number of jobs to run in parallel (-1 means use all cores)
MULTI = False  # should a model be built for each step of the forecasting horizon? (otherwise, one model for the entire horizon)


class MLP(Forecasting):
    """
    Builds a Neural Network forecasting model

    Parameters
    ----------
    csv : str
        name of (or path to) CSV file of a data frame

    output : str
        name of column to predict in a model

    inputs : list of str, default=None
        names of columns to use as features in a model

    datetime : str, default=None
        name of column to use as an index for the predictions and engineering time features

    resolution : list of str, default=None
        name of time intervals to use as features in a model
            options: year, quarter, month, week, dayofyear, day, weekday, hour, minute, second

    train_samples : int, default=100
        the number of observations to train the model with

    history_window : int, default=10
        the number of past time periods used as features in the model

    input_history : bool, default=False
        should backward lags be computed for input variables to use as features in the model?
        by default, backward lags are only computed for the output variable

    forecast_window : int, default=10
        the number of time periods in the future to predict

    forecast_frequency : int, default=1
        the number of time periods between predictions

    train_frequency : int, default=5
        the number of predictions between training a new model

    tune_model : bool, default=False
        should the model hyperparameters be optimized with a grid search?

    Attributes
    ----------
    _model : sklearn Pipeline, default=None
        the model to make predictions with

    _data : pandas DataFrame
        the full data set to stream through a model

    _predictions : pandas DataFrame
        the rolling predictions

    _actual : pandas DataFrame
        the known values to be predicted

    _error : pandas DataFrame
        the rolling weighted absolute percent error

    _counter : int
        the counter for scheduling model training
    """

    def predict_ahead(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Make a single forecast with a Neural Network model

        Parameters
        ----------
        df : pandas DataFrame
            the training (streamed) data to model

        Returns
        -------
        predictions : pandas DataFrame
            the forecast -> (1 row, W columns) where W is the forecast_window
        """
        # preprocess the data for supervised machine learning
        X, Y, X_new = self.preprocessing(df, binary=False)

        if self._counter >= self.train_frequency or self._model is None:
            object.__setattr__(self, "_counter", 0)

            # set up a machine learning pipeline
            model = MLPRegressor(
                max_iter=25,
                hidden_layer_sizes=(64, 64),
                learning_rate_init=0.001,
                batch_size=16,
                alpha=0,
                learning_rate="adaptive",
                activation="relu",
                solver="adam",
                warm_start=True,
                shuffle=False,
                random_state=42,
                verbose=False,
            )
            if MULTI:
                model = MultiOutputRegressor(
                    model,
                    n_jobs=N_JOBS,
                )
            pipeline = Pipeline(
                [
                    ("var", VarianceThreshold()),
                    ("scale", MinMaxScaler()),
                    ("model", model),
                ]
            )

            if self.tune_model:
                # set up cross validation for time series
                tscv = TimeSeriesSplit(n_splits=3)
                folds = tscv.get_n_splits(X)

                # set up the tuner
                str_ = ""
                if MULTI:
                    str_ = "estimator__"
                parameters = {
                    f"model__{str_}hidden_layer_sizes": (
                        (32, 32),
                        (64, 64),
                        (128, 128),
                    ),
                    f"model__{str_}batch_size": (16, 32),
                    f"model__{str_}learning_rate_init": (0.0001, 0.001, 0.01),
                }
                grid = RandomizedSearchCV(
                    pipeline,
                    parameters,
                    n_iter=16,
                    cv=folds,
                    random_state=0,
                    n_jobs=1 if MULTI else N_JOBS,
                )

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")  # ignore common warning
                    object.__setattr__(
                        self,
                        "_model",
                        grid.fit(X, Y).best_estimator_,  # search for the best model
                    )
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")  # ignore common warning
                    object.__setattr__(
                        self, "_model", pipeline.fit(X, Y)  # train the model
                    )

        predictions = self._model.predict(X_new)  # forecast
        predictions = pd.DataFrame(predictions)
        object.__setattr__(self, "_counter", self._counter + 1)
        return predictions


class DenseNet(Forecasting):
    """
    Builds a Neural Network forecasting model

    Parameters
    ----------
    csv : str
        name of (or path to) CSV file of a data frame

    output : str
        name of column to predict in a model

    inputs : list of str, default=None
        names of columns to use as features in a model

    datetime : str, default=None
        name of column to use as an index for the predictions and engineering time features

    resolution : list of str, default=None
        name of time intervals to use as features in a model
            options: year, quarter, month, week, dayofyear, day, weekday, hour, minute, second

    train_samples : int, default=100
        the number of observations to train the model with

    history_window : int, default=10
        the number of past time periods used as features in the model

    input_history : bool, default=False
        should backward lags be computed for input variables to use as features in the model?
        by default, backward lags are only computed for the output variable

    forecast_window : int, default=10
        the number of time periods in the future to predict

    forecast_frequency : int, default=1
        the number of time periods between predictions

    train_frequency : int, default=5
        the number of predictions between training a new model

    tune_model : bool, default=False
        should the model hyperparameters be optimized with a grid search?

    Attributes
    ----------
    _model : sklearn Pipeline, default=None
        the model to make predictions with

    _data : pandas DataFrame
        the full data set to stream through a model

    _predictions : pandas DataFrame
        the rolling predictions

    _actual : pandas DataFrame
        the known values to be predicted

    _error : pandas DataFrame
        the rolling weighted absolute percent error

    _counter : int
        the counter for scheduling model training
    """

    def predict_ahead(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Make a single forecast with a Neural Network model

        Parameters
        ----------
        df : pandas DataFrame
            the training (streamed) data to model

        Returns
        -------
        predictions : pandas DataFrame
            the forecast -> (1 row, W columns) where W is the forecast_window
        """
        # preprocess the data for supervised machine learning
        X, Y, X_new = self.preprocessing(df, binary=True)

        if self._counter >= self.train_frequency or self._model is None:
            object.__setattr__(self, "_counter", 0)

            # set up a machine learning pipeline
            model = KerasRegressor(
                build_fn=self.build,
                epochs=25,
                batch_size=16,
                shuffle=False,
                verbose=False,
                learning_rate=0.001,
                h_layers=[64, 64],
                l1_penalty=0,
                dropout=0,
                n_outputs=Y.shape[1],  # 1 if MULTI else Y.shape[1],
            )
            # if MULTI:
            #     model = MultiOutputRegressor(
            #         model, n_jobs=1 if self.tune_model else N_JOBS
            #     )
            pipeline = Pipeline(
                [
                    ("var", VarianceThreshold()),
                    ("scale", MinMaxScaler()),
                    ("model", model),
                ]
            )

            # if self.tune_model:
            #     # set up cross validation for time series
            #     tscv = TimeSeriesSplit(n_splits=3)
            #     folds = tscv.get_n_splits(X)

            #     # set up the tuner
            #     str_ = ""
            #     if MULTI:
            #         str_ = "estimator__"
            #     parameters = {
            #         f"model__{str_}h_layers": ([32, 32], [64, 64], [128, 128]),
            #         f"model__{str_}batch_size": (16, 32),
            #         f"model__{str_}learning_rate": (0.0001, 0.001, 0.01),
            #     }
            #     grid = RandomizedSearchCV(
            #         pipeline,
            #         parameters,
            #         n_iter=16,
            #         cv=folds,
            #         random_state=0,
            #         n_jobs=N_JOBS,
            #     )

            #     object.__setattr__(
            #         self,
            #         "_model",
            #         grid.fit(X, Y).best_estimator_,  # search for the best model
            #     )
            # else:
            object.__setattr__(self, "_model", pipeline.fit(X, Y))  # train the model

        predictions = self._model.predict(X_new)  # forecast
        predictions = pd.DataFrame(predictions).T
        object.__setattr__(self, "_counter", self._counter + 1)
        return predictions

    def build(
        self,
        n_outputs: int,
        h_layers: list = [32, 32],
        learning_rate: float = 0.001,
        l1_penalty: float = 0,
        dropout: float = 0,
    ):
        """
        Build a tensorflow dense neural network with the Adam optimizer and ReLU activation

        Parameters
        ----------
        n_outputs : int
            the number of output variables

        h_layers : list of int, default=[32, 32]
            the size of each hidden layer in the network

        learning_rate : float, default=0.001
            the initial learning rate for the Adam optimizer

        l1_penalty : float, default=0
            the L1-norm penalty for regulating the weights in the network

        dropout : float, default=0
            the fraction of weights to randomly ignore in each hidden layer

        Returns
        -------
        model : keras Sequential
            the neural network model
        """
        # build the network
        model = Sequential()
        for idx, nodes in enumerate(h_layers):
            model.add(
                layers.Dense(
                    units=nodes,
                    activation="relu",
                    kernel_regularizer=regularizers.l1(l1_penalty),
                )
            )
            model.add(layers.Dropout(dropout))
        model.add(layers.Dense(units=n_outputs, activation="linear"))

        # compile the model
        model.compile(
            loss=["mean_squared_error" for _ in h_layers],
            optimizer=optimizers.Adam(lr=learning_rate),
        )
        return model
