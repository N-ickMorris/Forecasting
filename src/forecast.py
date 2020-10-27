# -*- coding: utf-8 -*-
"""
Stream data through an exponential smoothing model to produce a rolling forecast

@author: Nick
"""

import os
import warnings
from typing import Union
from pydantic import BaseModel, root_validator
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing


class Forecasting(BaseModel):
    """
    A base class that can be inherited by a machine learning class to
    produce a model's rolling forecast -> redefine 'predict_ahead' method
    By default this class builds an Exponential Smoothing model

    Parameters
    ----------
    csv : str
        CSV file of a data frame -> "example.csv"

    output : str
        name of column to predict in a model -> "Y"

    inputs : list, default=None
        names of columns to use as features in a model -> ["X1", "X2"]

    datetime : str, default=None
        name of column to use as an index for the predictions

    train_samples : int, default=100
        the number of observations to train the model with

    history_window : int, default=10
        the number of past time periods used as features in the model

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
    _model : statsmodels ExponentialSmoothing, default=None
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

    # input arguments (**kwarg)
    csv: str
    output: str
    inputs: Union[list, None] = None
    datetime: Union[str, None] = None
    train_samples: int = 100
    history_window: int = 10
    forecast_window: int = 10
    forecast_frequency: int = 1
    train_frequency: int = 5
    tune_model: bool = False

    # internal attributes (not inputs)
    __slots__ = ["_model", "_data", "_predictions", "_actual", "_error", "_counter"]

    def __init__(self, **kwarg) -> None:
        """
        Validate the input arguments
        Load the full data set
        """
        super().__init__(**kwarg)  # validate the input arguments
        object.__setattr__(self, "_model", None)  # initial value
        object.__setattr__(self, "_data", pd.read_csv(self.csv))  # load the data
        object.__setattr__(self, "_predictions", pd.DataFrame())  # initial value
        object.__setattr__(self, "_actual", pd.DataFrame())  # initial value
        object.__setattr__(self, "_error", pd.DataFrame())  # initial value
        object.__setattr__(self, "_counter", 0)  # initial value

    @root_validator
    def value_checks(cls, values) -> dict:
        """
        Check that the given csv file exists
        Check that the given column names are in the csv file
        Check that the forecasting parameters have reasonable values
        """
        # validate csv
        csv = values.get("csv")
        if not csv.endswith(".csv"):
            raise ValueError(
                f"'csv' should be a file name with a .csv extension, got a value of '{csv}' instead"
            )
        if not os.path.isfile(csv):
            raise FileNotFoundError(f"There is no csv file at '{os.getcwd()}/{csv}'")
        df = pd.read_csv(csv)

        Y, X, T = values.get("output"), values.get("inputs"), values.get("datetime")

        # validate output
        if not Y in df.columns:
            raise ValueError(
                f"'output' should be a column name in 'csv', got a value of '{Y}' instead"
            )

        # validate inputs
        if not X is None:
            if len(X) == 0:
                raise ValueError(
                    "'inputs' should be a list of column names in 'csv' or a value of None, got an empty list instead"
                )
            not_found = [not x in df.columns for x in X]
            if any(not_found):
                invalid_names = np.array(X)[np.where(not_found)[0]].tolist()
                raise ValueError(
                    f"'inputs' should be a list of column names in 'csv' or a value of None, but the following names are not in 'csv': {invalid_names}"
                )

        # validate datetime
        if not T in df.columns:
            raise ValueError(
                f"'datetime' should be a column name in 'csv', got a value of '{T}' instead"
            )

        samples = values.get("train_samples")
        h_win, f_win = values.get("history_window"), values.get("forecast_window")
        f_freq, t_freq = values.get("forecast_frequency"), values.get("train_frequency")

        # validate forecasting parameters
        if not 1 <= h_win < samples:
            raise ValueError(
                f"'history_window' should be at least 1 and less than 'train_samples' ({samples}), got a value of {h_win} instead"
            )
        if not 1 <= f_win < samples:
            raise ValueError(
                f"'forecast_window' should be at least 1 and less than 'train_samples' ({samples}), got a value of {f_win} instead"
            )
        if not h_win + f_win < samples:
            raise ValueError(
                f"'history_window' + 'forecast_window' should be less than 'train_samples' ({samples}), got a value of {h_win + f_win} instead"
            )
        if not t_freq >= 1:
            raise ValueError(
                f"'train_frequency' should be at least 1, got a value of {t_freq} instead"
            )
        if not f_freq >= 1:
            raise ValueError(
                f"'forecast_frequency' should be at least 1, got a value of {f_freq} instead"
            )
        if not samples + f_win + f_freq < df.shape[0]:
            raise ValueError(
                f"'train_samples' + 'forecast_window' + 'forecast_frequency' should be less than the number of rows in 'csv' ({df.shape[0]}), got a value of {samples + f_win + f_freq} instead"
            )

        return values

    def stream_data(self, df: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
        """
        Query a segment of data for training a model

        Parameters
        ----------
        df : pandas DataFrame
            the available data to query

        start : int
            the first row to pull

        end : int
            the last row to pull

        Returns
        -------
        pandas DataFrame
            the requested data segment
        """
        return df.iloc[start:end, :].copy()

    def series_to_supervised(
        self, df: pd.DataFrame, n_backward: int, n_forward: int, dropnan: bool = False
    ) -> pd.DataFrame:
        """
        Reshape a data frame such that there are time lagged features
        Reference: https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/

        Parameters
        ----------
        df : pandas DataFrame
            the available data to reshape

        n_backward : int
            the number of lags backward to shift the data

        n_forward : int
            the number of lags forward to shift the data

        dropnan : bool, default=False
            should rows with missing values be dropped?

        Returns
        -------
        agg : pandas DataFrame
            the requested data reshaped
        """
        n_vars = 1 if type(df) is list else df.shape[1]
        df = pd.DataFrame(df)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_backward, 0, -1):
            cols.append(df.shift(i))
            names += [(str(df.columns[j]) + "(t-%d)" % (i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_forward):
            cols.append(df.shift(-i))
            if i == 0:
                names += [str(df.columns[j]) + "(t)" for j in range(n_vars)]
            else:
                names += [(str(df.columns[j]) + "(t+%d)" % (i)) for j in range(n_vars)]
        # put it all together
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg

    def reshape_output(self, df: pd.DataFrame) -> tuple:
        """
        Reshape a data frame such that there are time lagged features
        Split up the data frame into inputs and outputs for modeling

        Parameters
        ----------
        df : pandas DataFrame
            the available data to reshape

        Returns
        -------
        x : pandas DataFrame
            the backward lagged features (inputs)

        y : pandas DataFrame
            the forward lagged features (outputs)
        """
        df = self.series_to_supervised(
            df[[self.output]].copy(), self.history_window, self.forecast_window + 1
        )
        x = df.iloc[:, :(self.history_window + 1)]
        y = df.drop(columns=x.columns)
        return x, y

    def predict_ahead(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Make a single forecast with an Exponential Smoothing model

        Parameters
        ----------
        df : pandas DataFrame
            the training (streamed) data to model

        Returns
        -------
        predictions : pandas DataFrame
            the forecast -> (1 row, W columns) where W is the forecast_window
        """
        model = ExponentialSmoothing(
            df[[self.output]],
            trend="add",  # add, mul, None
            damped_trend=False,
            # seasonal="add",        # add, mul
            # seasonal_periods=None  # positve integer
            initialization_method="estimated",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # ignore common warning
            object.__setattr__(self, "_model", model.fit())  # train the model
        predictions = self._model.forecast(self.forecast_window)
        predictions = pd.DataFrame(predictions).reset_index(drop=True).T
        return predictions

    def roll(self, verbose: int = 1) -> None:
        """
        Make rolling forecasts with a model

        Parameters
        ----------
        verbose : int
            should the forecasting error be printed out? (0 - no, 1 - yes)
        """
        for step in range(
            self.train_samples,
            self._data.shape[0] - self.forecast_window,
            self.forecast_frequency,
        ):
            # train a model and make a forecast
            df_train = self.stream_data(self._data, step - self.train_samples, step)
            predicted = self.predict_ahead(df_train)
            actual = (
                self.stream_data(
                    self._data[[self.output]], step + 1, step + 1 + self.forecast_window
                )
                .reset_index(drop=True)
                .T
            )

            # define column and row names for this forecast
            column_names = [
                f"{self.output} (t+{i + 1})" for i in range(self.forecast_window)
            ]
            row_name = (
                step if self.datetime is None else self._data[self.datetime][step]
            )
            predicted.columns, actual.columns = column_names, column_names
            predicted.index, actual.index = [row_name], [row_name]

            # compute the forecast error (weighted absolute percent error)
            error = pd.DataFrame(
                (actual - predicted).abs().sum(axis="columns")
                / actual.abs().sum(axis="columns")
            )
            error.columns = ["percent_error"]

            # save the forecast
            object.__setattr__(
                self,
                "_predictions",
                pd.concat([self._predictions, predicted], axis="index"),
            )
            object.__setattr__(
                self, "_actual", pd.concat([self._actual, actual], axis="index")
            )
            object.__setattr__(
                self, "_error", pd.concat([self._error, error], axis="index")
            )

            # report the percent error
            if verbose != 0:
                print(
                    f"time={error.index[0]}, error={np.round(error.values[0] * 100, 2)[0]}%"
                )
