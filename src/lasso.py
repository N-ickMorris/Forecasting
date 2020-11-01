# -*- coding: utf-8 -*-
"""
Stream data through a lasso regression model to produce a rolling forecast

@author: Nick
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Lasso, LassoCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import TimeSeriesSplit
from forecast import Forecasting


class Regression(Forecasting):
    """
    A lasso regression forecasting model:
        model = Regression(**kwarg)
        model.roll(verbose=1)

    Parameters
    ----------
    csv : str
        name of (or path to) CSV file of a data frame

    output : str
        name of column to predict in a model

    inputs : list of str, default=None
        names of columns to use as features in a model

    datetime : str, default=None
        name of column to use as an index for the predictions

    resolution : list of str, default=None
        name of time intervals to use as features in a model
            options: year, quarter, month, week, dayofyear, day, weekday, hour, minute, second

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
        Make a single forecast with a Lasso Regression model

        Parameters
        ----------
        df : pandas DataFrame
            the training (streamed) data to model

        Returns
        -------
        predictions : pandas DataFrame
            the forecast -> (1 row, W columns) where W is the forecast_window
        """
        # split up inputs (X) and outputs (Y)
        X = df[self.inputs].copy() if not self.inputs is None else pd.DataFrame()
        Y = df[[self.output]].copy()

        # add autoregressive terms to X, add forecast horizon to Y
        X2, Y = self.reshape_output(Y)
        X = pd.concat([X, X2], axis="columns")

        # add the timestamp features to X
        if not self.resolution is None and not self.datetime is None:
            try:
                timestamps = pd.to_datetime(df[self.datetime])

                # extend the timestamps to include the forecast horizon
                delta = timestamps[-1:].values[0] - timestamps[-2:-1].values[0]
                forecast_timestamps = pd.date_range(
                    start=timestamps[-1:].values[0] + delta,
                    end=timestamps[-1:].values[0] + delta * self.forecast_window,
                    periods=self.forecast_window,
                )
                timestamps = pd.Series(
                    np.concatenate((timestamps, forecast_timestamps))
                )
                T = pd.DataFrame()

                # extract features from the timestamps
                if "year" in self.resolution:
                    year = pd.get_dummies(timestamps.dt.year.astype(str))
                    year.columns = [f"year_{c}" for c in year.columns]
                    T = pd.concat([T, year], axis="columns")
                if "quarter" in self.resolution:
                    quarter = pd.get_dummies(timestamps.dt.quarter.astype(str))
                    quarter.columns = [f"quarter_{c}" for c in quarter.columns]
                    T = pd.concat([T, quarter], axis="columns")
                if "month" in self.resolution:
                    month = pd.get_dummies(timestamps.dt.month.astype(str))
                    month.columns = [f"month_{c}" for c in month.columns]
                    T = pd.concat([T, month], axis="columns")
                if "week" in self.resolution:
                    week = pd.get_dummies(timestamps.dt.week.astype(str))
                    week.columns = [f"week_{c}" for c in week.columns]
                    T = pd.concat([T, week], axis="columns")
                if "dayofyear" in self.resolution:
                    dayofyear = pd.get_dummies(timestamps.dt.dayofyear.astype(str))
                    dayofyear.columns = [f"dayofyear_{c}" for c in dayofyear.columns]
                    T = pd.concat([T, dayofyear], axis="columns")
                if "day" in self.resolution:
                    day = pd.get_dummies(timestamps.dt.day.astype(str))
                    day.columns = [f"day_{c}" for c in day.columns]
                    T = pd.concat([T, day], axis="columns")
                if "weekday" in self.resolution:
                    weekday = pd.get_dummies(timestamps.dt.weekday.astype(str))
                    weekday.columns = [f"weekday_{c}" for c in weekday.columns]
                    T = pd.concat([T, weekday], axis="columns")
                if "hour" in self.resolution:
                    hour = pd.get_dummies(timestamps.dt.hour.astype(str))
                    hour.columns = [f"hour_{c}" for c in hour.columns]
                    T = pd.concat([T, hour], axis="columns")
                if "minute" in self.resolution:
                    minute = pd.get_dummies(timestamps.dt.minute.astype(str))
                    minute.columns = [f"minute_{c}" for c in minute.columns]
                    T = pd.concat([T, minute], axis="columns")
                if "second" in self.resolution:
                    second = pd.get_dummies(timestamps.dt.second.astype(str))
                    second.columns = [f"second_{c}" for c in second.columns]
                    T = pd.concat([T, second], axis="columns")

                # add the forecasting horizon to the timestamp features
                T = self.series_to_supervised(
                    T, n_backward=0, n_forward=self.forecast_window + 1, dropnan=True
                )
                X = pd.concat([X, T], axis="columns")

            except:
                print("Cannot parse 'datetime' into a datetime object")

        # use the last row to predict the horizon
        X_new = X[-1:].copy()

        # remove missing values
        df = pd.concat([X, Y], axis="columns").dropna()
        X = df[X.columns]
        Y = df[Y.columns]

        if self._counter >= self.train_frequency or self._model is None:
            object.__setattr__(self, "_counter", 0)

            # set up the machine learning model
            if self.tune_model:
                # set up cross validation for time series
                tscv = TimeSeriesSplit(n_splits=3)
                folds = tscv.get_n_splits(X)
                model = LassoCV(cv=folds, eps=1e-9, n_alphas=16, n_jobs=-1)
            else:
                model = Lasso(alpha=0.1)

            # set up a machine learning pipeline
            pipeline = Pipeline(
                [
                    ("var", VarianceThreshold()),
                    ("scale", MinMaxScaler()),
                    ("model", MultiOutputRegressor(model)),
                ]
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # ignore common warning
                object.__setattr__(
                    self, "_model", pipeline.fit(X, Y)  # train the model
                )

        # forecast
        predictions = self._model.predict(X_new)
        predictions = pd.DataFrame(predictions)
        object.__setattr__(self, "_counter", self._counter + 1)
        return predictions
