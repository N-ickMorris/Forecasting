# -*- coding: utf-8 -*-
"""
Stream data through a extreme gradient boosting tree model to produce a rolling forecast

@author: Nick
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from xgboost.sklearn import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import TimeSeriesSplit
from forecast import Forecasting

N_JOBS = 1  # The number of jobs to run in parallel (-1 means use all cores)


class XGBTree(Forecasting):
    """
    Builds a Extreme Gradient Boosting Tree forecasting model

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
        Make a single forecast with a Extreme Gradient Boosting Tree model

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
            model = XGBRegressor(
                booster="gbtree",
                n_estimators=25,
                learning_rate=0.1,
                max_depth=7,
                min_child_weight=1,
                colsample_bytree=0.8,
                subsample=0.8,
                random_state=42,
                n_jobs=N_JOBS,
            )
            model = MultiOutputRegressor(model, n_jobs=1)
            pipeline = Pipeline(
                [
                    ("var", VarianceThreshold()),
                    ("model", model),
                ]
            )

            if self.tune_model:
                # set up cross validation for time series
                tscv = TimeSeriesSplit(n_splits=3)
                folds = tscv.get_n_splits(X)

                # set up the tuner
                parameters = {
                    "model__estimator__n_estimators": [25, 50, 100],
                    "model__estimator__learning_rate": [0.001, 0.01, 0.1, 1],
                    "model__estimator__max_depth": [3, 6, 9, 12],
                    "model__estimator__min_child_weight": [1, 3, 5],
                    "model__estimator__colsample_bytree": [0.8],
                    "model__estimator__subsample": [0.8],
                }
                grid = RandomizedSearchCV(
                    pipeline,
                    parameters,
                    n_iter=16,
                    cv=folds,
                    random_state=0,
                    n_jobs=1,
                )

                object.__setattr__(
                    self,
                    "_model",
                    grid.fit(X, Y).best_estimator_,  # search for the best model
                )
            else:
                object.__setattr__(
                    self, "_model", pipeline.fit(X, Y)  # train the model
                )

        predictions = self._model.predict(X_new)  # forecast
        predictions = pd.DataFrame(predictions)
        object.__setattr__(self, "_counter", self._counter + 1)
        return predictions
