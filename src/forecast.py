# -*- coding: utf-8 -*-
"""
Stream data through an exponential smoothing model to produce a rolling forecast

@author: Nick
"""

import os
import warnings
from typing import Union, List
from pydantic import BaseModel, root_validator
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing


class Forecasting(BaseModel):
    """
    A base class that can be inherited by a machine learning class to
    produce a model's rolling forecast -> redefine 'predict_ahead' method
    By default this class builds an Exponential Smoothing forecasting model

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
    inputs: Union[List[str], None] = None
    datetime: Union[str, None] = None
    resolution: Union[List[str], None] = None
    train_samples: int = 100
    history_window: int = 10
    input_history: bool = False
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
        object.__setattr__(
            self, "_data", pd.read_csv(self.csv).reset_index(drop=True)
        )  # load the data
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

        Y, X, T, R = (
            values.get("output"),
            values.get("inputs"),
            values.get("datetime"),
            values.get("resolution"),
        )

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

        # validate resolution
        if not R is None:
            not_found = [
                not r
                in [
                    "year",
                    "quarter",
                    "month",
                    "week",
                    "dayofyear",
                    "day",
                    "weekday",
                    "hour",
                    "minute",
                    "second",
                ]
                for r in R
            ]
            if any(not_found):
                invalid_names = np.array(R)[np.where(not_found)[0]].tolist()
                raise ValueError(
                    f"'resolution' should be a list of any of the following time intervals: ['year', 'quarter', 'month', 'week', 'dayofyear', 'day', 'weekday', 'hour', 'minute', 'second'] or a value of None, but the following are not time intervals: {invalid_names}"
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
        df = pd.DataFrame(df.copy())
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
            df[[self.output]], self.history_window, self.forecast_window + 1
        )
        x = df.iloc[:, : (self.history_window + 1)]
        y = df.drop(columns=x.columns)
        return x, y

    def time_features(self, df: pd.DataFrame, binary: bool = True) -> pd.DataFrame:
        """
        Extract features from a datetime column

        Parameters
        ----------
        df : pandas DataFrame
            the available data with a datetime column

        binary : bool, default=True
            should the time features be converted into binary variables?

        Returns
        -------
        T : pandas DataFrame
            the time features
        """
        if self.resolution is None:
            raise ValueError(
                "'resolution' should be a list of any of the following time intervals: ['year', 'quarter', 'month', 'week', 'dayofyear', 'day', 'weekday', 'hour', 'minute', 'second'], got None instead"
            )
        if self.datetime is None:
            raise ValueError(
                "'datetime' should be a column name in 'df', got None instead"
            )
        timestamps = pd.to_datetime(df[self.datetime].copy())

        # extend the timestamps to include the forecast horizon
        delta = timestamps[-1:].values[0] - timestamps[-2:-1].values[0]
        forecast_timestamps = pd.date_range(
            start=timestamps[-1:].values[0] + delta,
            end=timestamps[-1:].values[0] + delta * self.forecast_window,
            periods=self.forecast_window,
        )
        timestamps = pd.Series(np.concatenate((timestamps, forecast_timestamps)))
        T = pd.DataFrame()

        # extract features from the timestamps
        if "year" in self.resolution:
            year = (
                pd.get_dummies(timestamps.dt.year.astype(str))
                if binary
                else pd.DataFrame(timestamps.dt.year)
            )
            year.columns = [f"year_{c}" for c in year.columns] if binary else ["year"]
            T = pd.concat([T, year], axis="columns")
        if "quarter" in self.resolution:
            quarter = (
                pd.get_dummies(timestamps.dt.quarter.astype(str))
                if binary
                else pd.DataFrame(timestamps.dt.quarter)
            )
            quarter.columns = (
                [f"quarter_{c}" for c in quarter.columns] if binary else ["quarter"]
            )
            T = pd.concat([T, quarter], axis="columns")
        if "month" in self.resolution:
            month = (
                pd.get_dummies(timestamps.dt.month.astype(str))
                if binary
                else pd.DataFrame(timestamps.dt.month)
            )
            month.columns = (
                [f"month_{c}" for c in month.columns] if binary else ["month"]
            )
            T = pd.concat([T, month], axis="columns")
        if "week" in self.resolution:
            week = (
                pd.get_dummies(timestamps.dt.isocalendar().week.astype(str))
                if binary
                else pd.DataFrame(timestamps.dt.isocalendar().week)
            )
            week.columns = [f"week_{c}" for c in week.columns] if binary else ["week"]
            T = pd.concat([T, week], axis="columns")
        if "dayofyear" in self.resolution:
            dayofyear = (
                pd.get_dummies(timestamps.dt.dayofyear.astype(str))
                if binary
                else pd.DataFrame(timestamps.dt.dayofyear)
            )
            dayofyear.columns = (
                [f"dayofyear_{c}" for c in dayofyear.columns]
                if binary
                else ["dayofyear"]
            )
            T = pd.concat([T, dayofyear], axis="columns")
        if "day" in self.resolution:
            day = (
                pd.get_dummies(timestamps.dt.day.astype(str))
                if binary
                else pd.DataFrame(timestamps.dt.day)
            )
            day.columns = [f"day_{c}" for c in day.columns] if binary else ["day"]
            T = pd.concat([T, day], axis="columns")
        if "weekday" in self.resolution:
            weekday = (
                pd.get_dummies(timestamps.dt.weekday.astype(str))
                if binary
                else pd.DataFrame(timestamps.dt.weekday)
            )
            weekday.columns = (
                [f"weekday_{c}" for c in weekday.columns] if binary else ["weekday"]
            )
            T = pd.concat([T, weekday], axis="columns")
        if "hour" in self.resolution:
            hour = (
                pd.get_dummies(timestamps.dt.hour.astype(str))
                if binary
                else pd.DataFrame(timestamps.dt.hour)
            )
            hour.columns = [f"hour_{c}" for c in hour.columns] if binary else ["hour"]
            T = pd.concat([T, hour], axis="columns")
        if "minute" in self.resolution:
            minute = (
                pd.get_dummies(timestamps.dt.minute.astype(str))
                if binary
                else pd.DataFrame(timestamps.dt.minute)
            )
            minute.columns = (
                [f"minute_{c}" for c in minute.columns] if binary else ["minute"]
            )
            T = pd.concat([T, minute], axis="columns")
        if "second" in self.resolution:
            second = (
                pd.get_dummies(timestamps.dt.second.astype(str))
                if binary
                else pd.DataFrame(timestamps.dt.second)
            )
            second.columns = (
                [f"second_{c}" for c in second.columns] if binary else ["second"]
            )
            T = pd.concat([T, second], axis="columns")

        # add the forecasting horizon to the timestamp features
        n_backward = self.history_window if self.input_history else 0
        T = self.series_to_supervised(
            T, n_backward=n_backward, n_forward=self.forecast_window + 1, dropnan=False
        )
        T = T.iloc[
            : -self.forecast_window
        ]  # remove the last set of rows with missing values

        return T

    def rolling_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute rolling statistics on a data frame:
            Mean, Min, Max, Std Dev

        Parameters
        ----------
        df : pandas DataFrame
            the available data

        Returns
        -------
        S : pandas DataFrame
            the rolling statistics
        """
        df = df.copy()
        # rolling stats are shifted backward by 1 to enforce that only previous values are used in computation
        Avg = df.rolling(self.history_window).mean().shift(1)
        Avg.columns = [f"{c}_avg" for c in Avg.columns]
        Min = df.rolling(self.history_window).min().shift(1)
        Min.columns = [f"{c}_min" for c in Min.columns]
        Max = df.rolling(self.history_window).max().shift(1)
        Max.columns = [f"{c}_max" for c in Max.columns]
        Std = df.rolling(self.history_window).std().shift(1)
        Std.columns = [f"{c}_stdDev" for c in Std.columns]
        S = pd.concat([Avg, Min, Max, Std], axis="columns")
        return S

    def preprocessing(self, df: pd.DataFrame, binary: bool = True) -> tuple:
        """
        Preprocess the training data with feature engineering to enable
        forecasting with supervised machine learning models

        Parameters
        ----------
        df : pandas DataFrame
            the training (streamed) data to model

        binary : bool, default=True
            should the time features be converted into binary variables?

        Returns
        -------
        X : pandas DataFrame
            the input features and time series features to train the mode

        Y : pandas DataFrame
            the output widened by shifting itself to engineer the forecasting window to train the model

        X_new : pandas DataFrame
            the input features and time series features to produce the forecast
        """
        # split up inputs (X) and outputs (Y)
        X = df[self.inputs].copy() if not self.inputs is None else pd.DataFrame()
        Y = df[[self.output]].copy()

        if self.input_history and not self.inputs is None:
            # add autoregressive terms (on X) to X
            Ax = self.series_to_supervised(
                X, n_backward=self.history_window, n_forward=0
            )

            # add rolling statistics (on X) to X
            Sx = self.rolling_stats(X)
            X = pd.concat([X, Ax, Sx], axis="columns")

        # add rolling statistics (on Y) to X
        Sy = self.rolling_stats(Y)
        X = pd.concat([X, Sy], axis="columns")

        # add autoregressive terms (on Y) to X
        # add forecast horizon to Y
        Ay, Y = self.reshape_output(Y)
        X = pd.concat([X, Ay], axis="columns")

        # add timestamp features to X
        if not self.resolution is None and not self.datetime is None:
            try:
                T = self.time_features(df[[self.datetime]], binary=binary)
                X = pd.concat([X, T], axis="columns")
            except:
                print(
                    "Cannot parse 'datetime' into a datetime object, no time features were added to the model"
                )

        # use the last row to predict the horizon
        X_new = X[-1:].copy()

        # remove missing values
        df = pd.concat([X, Y], axis="columns").dropna()
        X = df[X.columns]
        Y = df[Y.columns]

        return X, Y, X_new

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

    def roll(self, verbose: bool = True) -> None:
        """
        Make rolling forecasts with a model

        Parameters
        ----------
        verbose : bool, default=True
            should the forecasting error be printed out?
        """
        for step in range(
            self.train_samples,
            self._data.shape[0] - self.forecast_window,
            self.forecast_frequency,
        ):
            # train a model and make a forecast
            df = self.stream_data(
                self._data, step - self.train_samples, step
            ).reset_index(drop=True)
            predicted = self.predict_ahead(df)
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
            if verbose:
                print(
                    f"time={error.index[0]}, error={np.round(error.values[0] * 100, 2)[0]}%"
                )
