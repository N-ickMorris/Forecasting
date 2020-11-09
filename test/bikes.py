# -*- coding: utf-8 -*-
"""
Build forecasting models to predict bike share demand

@author: Nick
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import r2_score

# add the src directory to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/")))

from forecast import Forecasting
from lasso import Regression
from forest import Forest
from nnet import MLP

# In[1]: Prepare the data for modeling
data = pd.read_csv("test/bikes.csv")

# convert season and weather into binary variables
data["season"] = data["season"].astype(str)
data["weather"] = data["weather"].astype(str)
data = pd.get_dummies(data, columns=["season", "weather"])

# save the data
data[:1440].to_csv("test/bikes_v2.csv", index=False)

# import the configuration for modeling
with open("test/bikes.json") as f:
    config = json.load(f)

# update the configuration for the new version of the data
config["csv"] = "test/bikes_v2.csv"
config["inputs"] = data.drop(
    columns=["datetime", "count", "casual", "registered"]
).columns.tolist()

# test features
# config["inputs"] = None
# config["resolution"] = None
config["input_history"] = False

# In[2]: Model the data

# produce a random forest rolling forecast
print("---- Random Forest ----")
model2 = Forest(**config)
model2.roll(verbose=True)
print(f"Forest Average Error: {np.round(model2._error.mean()[0] * 100, 2)}%")

# produce a lasso regression rolling forecast
print("---- Lasso Regression ----")
model1 = Regression(**config)
model1.roll(verbose=True)
print(f"Lasso Average Error: {np.round(model1._error.mean()[0] * 100, 2)}%")

# produce a neural network rolling forecast
print("---- Neural Network ----")
model3 = MLP(**config)
model3.roll(verbose=True)
print(f"NNet Average Error: {np.round(model3._error.mean()[0] * 100, 2)}%")

# produce a baseline rolling forecast (exponential smoothing)
print("---- Exponential Smoothing ----")
baseline_model = Forecasting(**config)
baseline_model.roll(verbose=True)
print(f"Baseline Average Error: {np.round(baseline_model._error.mean()[0] * 100, 2)}%")

# print(f"Baseline Average Error: {np.round(baseline_model._error.mean()[0] * 100, 2)}%")
# print(f"Lasso Average Error: {np.round(model1._error.mean()[0] * 100, 2)}%")
# print(f"Forest Average Error: {np.round(model2._error.mean()[0] * 100, 2)}%")
# print(f"NNet Average Error: {np.round(model3._error.mean()[0] * 100, 2)}%")

# In[3]: Analyze the model

# pick a model
model = model2

# pick a step ahead to evaluate
step_ahead = 1
df = pd.concat(
    [model._actual.iloc[:, step_ahead - 1], model._predictions.iloc[:, step_ahead - 1]],
    axis="columns",
)
df.columns = ["Actual", "Predict"]
df["index"] = pd.to_datetime(df.index)

# report R2
print(
    f'step ahead={step_ahead}, R2={np.round(r2_score(df["Actual"], df["Predict"]) * 100, 2)}%'
)

# plot the prediction series
fig = px.line(df, x="index", y="Predict")
fig.add_trace(
    go.Scatter(
        x=df["index"], y=df["Actual"], mode="lines", showlegend=False, name="Actual"
    )
)
fig.update_layout(font=dict(size=16))
fig.show()

# draw a parity plot
fig1 = px.scatter(df, x="Actual", y="Predict")
fig1.add_trace(
    go.Scatter(
        x=df["Actual"], y=df["Actual"], mode="lines", showlegend=False, name="Actual"
    )
)
fig1.update_layout(font=dict(size=16))
fig1.show()
