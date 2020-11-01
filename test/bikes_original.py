# -*- coding: utf-8 -*-
"""
Modeling Bike Share Demand with Lasso Regression
@author: Nick
"""


import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, QuantileTransformer
from sklearn.linear_model import LassoCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot


# define model parameters
ar = 4  # number of autoregressive terms to model with
ma_window = 5  # number of data points to compute a moving average term

# convert series to supervised learning
def series_to_supervised(data, n_backward=1, n_forward=1, dropnan=False):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
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


# In[1]: Prepare the data

# read in the data
data = pd.read_csv("test/bikes.csv")

# define the input variables
inputs = data.drop(columns=["casual", "registered", "count"]).columns.tolist()

# split up the data into inputs (X) and outputs (Y)
X = data[inputs].copy()
Y = data[["count"]].copy()

# add autoregressive terms to X
X = pd.concat([X, series_to_supervised(Y, n_backward=ar, n_forward=0)], axis=1)

# add a moving average term to X
MA = Y.rolling(ma_window).mean()
MA.columns = [c + " ma(" + str(ma_window) + ")" for c in MA.columns]
X = pd.concat([X, MA], axis=1)

# drop rows with missing values
df = pd.concat([X, Y], axis=1).dropna().reset_index(drop=True)
X = df[X.columns]
Y = df[Y.columns]

# split up the data into training and testing
size = int(X.shape[0] / 2)
train_idx = X[:size].index.tolist()
test_idx = X[size:].index.tolist()

# convert datetime to a date time object
X["datetime"] = pd.to_datetime(X["datetime"])

# collect the hour
hours = pd.get_dummies(pd.Series(X["datetime"]).dt.hour.astype(str))
hours.columns = ["Hour " + str(c) for c in hours.columns]

# collect the weekday
weekday = pd.get_dummies(pd.Series(X["datetime"]).dt.weekday.astype(str))
weekday.columns = ["Weekday " + str(c) for c in weekday.columns]

# collect the week
week = pd.get_dummies(pd.Series(X["datetime"]).dt.week.astype(str))
week.columns = ["Week " + str(c) for c in week.columns]

# collect the month
month = pd.get_dummies(pd.Series(X["datetime"]).dt.month.astype(str))
month.columns = ["Month " + str(c) for c in month.columns]

# add the features to the data
X = pd.concat([X, hours, weekday, week, month], axis=1)

# make datetime the index
X.index = X["datetime"]
X = X.drop(columns="datetime")

# convert season and weather to string variables
X["season"] = X["season"].astype(str)
X["weather"] = X["weather"].astype(str)

# determine which columns are strings (for X)
x_columns = X.columns
x_dtypes = X.dtypes
x_str = np.where(x_dtypes == "object")[0]

# convert any string columns to binary columns
X = pd.get_dummies(X, columns=x_columns[x_str])

# In[2]: Model the data

# set up cross validation for time series
tscv = TimeSeriesSplit(n_splits=5)
folds = tscv.get_n_splits(X)

# set up a machine learning pipeline
pipeline = Pipeline(
    [
        ("var1", VarianceThreshold()),
        # ('poly', PolynomialFeatures(2)),
        # ('var2', VarianceThreshold()),
        # ('shape', QuantileTransformer(output_distribution="normal"))
        ("scale", MinMaxScaler()),
        ("model", LassoCV(cv=folds, eps=1e-9, n_alphas=16, n_jobs=-1)),
    ]
)

# train a model
pipeline.fit(X.iloc[train_idx, :], Y.iloc[train_idx, :])

# forecast
predict = pipeline.predict(X.iloc[test_idx, :])
actual = Y.iloc[test_idx, :].to_numpy().T[0]

# score the forecast
print("R2: " + str(r2_score(actual, predict)))

# prepare the data for plotting
df = pd.DataFrame({"Predict": predict, "Actual": actual})
df["index"] = X.iloc[test_idx, :].index

# plot the prediction series
fig = px.line(df, x="index", y="Predict")
fig.add_trace(
    go.Scatter(
        x=df["index"], y=df["Actual"], mode="lines", showlegend=False, name="Actual"
    )
)
fig.update_layout(font=dict(size=16))
plot(fig, filename="Series Predictions.html")

# draw a parity plot
fig1 = px.scatter(df, x="Actual", y="Predict")
fig1.add_trace(
    go.Scatter(
        x=df["Actual"], y=df["Actual"], mode="lines", showlegend=False, name="Actual"
    )
)
fig1.update_layout(font=dict(size=16))
plot(fig1, filename="Parity Plot.html")
