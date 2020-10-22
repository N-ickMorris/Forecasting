"""
Stream data through a machine learning model to produce a rolling forecast

@author: Nick
"""

from pydantic import BaseModel
import pandas as pd

class Forecasting(BaseModel):
    """
    A base class that is to be inherited by a Machine Learning class to 
    produce a model's rolling forecast

    Parameters
    ----------
    csv : str
        CSV file of a data frame -> "example.csv"

    output : str
        name of column to predict in a model -> "Y"
    
    inputs : list of str
        names of columns to use as features in a model -> ["X1", "X2"]

    tune_model : bool, default=False
        should the model hyperparameters be optimized with a grid search?

    train_samples : int, default=100
        the number of observations to train the model with

    history_window : int, default=10
        the number of past time periods used as features in the model
    
    forecast_window : int, default=10
        the number of time periods ahead that the model predicts
    
    forecast_frequency : int, default=1
        the number of time periods between predictions
    
    train_frequency : int, default=5
        the number of predictions between training a new model

    Attributes
    ----------
    _current_model : sklearn Pipeline, None
        the model to make predictions with
    
    _data : pandas DataFrame
        the full data set to stream through a model
    """
    # input arguments (**kwarg)
    csv: str
    datetime: str
    output: str
    inputs: list
    tune_model: bool=False
    train_samples: int=100
    history_window: int=10
    forecast_window: int=10
    forecast_frequency: int=1
    train_frequency: int=5

    # internal attributes (not inputs)
    __slots__ = ["_current_model", "_data"]

    def __init__(self, **kwarg):
        """
        Load the full data set
        """
        super().__init__(**kwarg) # type check the input arguments
        object.__setattr__(self, "_current_model", None) # initial value
        object.__setattr__(self, "_data", pd.read_csv(self.csv)) # load the data frame

    def pull_data(self, df: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
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

    def series_to_supervised(self, df: pd.DataFrame, n_backward: int, 
                             n_forward: int, dropnan: bool=False) -> pd.DataFrame:
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
        pandas DataFrame
            the requested data reshaped
        """
        n_vars = 1 if type(df) is list else df.shape[1]
        df = pd.DataFrame(df)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_backward, 0, -1):
            cols.append(df.shift(i))
            names += [(str(df.columns[j]) + '(t-%d)' % (i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_forward):
            cols.append(df.shift(-i))
            if i == 0:
                names += [str(df.columns[j]) + '(t)' for j in range(n_vars)]
            else:
                names += [(str(df.columns[j]) + '(t+%d)' % (i)) for j in range(n_vars)]
        # put it all together
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg

    def reshape_output(self, df: pd.DataFrame):
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
        if df.shape[1] != 1:
            raise ValueError("'df' must contain exactly 1 column representing the model output")
        df = self.series_to_supervised(df[[self.output]].copy(), self.history_window,
                                       self.forecast_window + 1)
        y = df.iloc[:, (self.forecast_window + 1):]
        x = df.drop(columns=y.columns)
        return x, y