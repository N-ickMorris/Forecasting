# -*- coding: utf-8 -*-
"""
Build forecasting models to predict bike share demand

@author: Nick
"""

import os
import sys

# add the src directory to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/")))

import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from forecast import Forecasting
from lasso import Regression

# import the configuration for modeling
with open("test/config.json") as f:
    config = json.load(f)

# produce a rolling forecast
model = Regression(**config)
model.roll()

# pick a step ahead to evaluate
step_ahead = 1
df = pd.concat(
    [model._actual.iloc[:, step_ahead - 1], model._predictions.iloc[:, step_ahead - 1]],
    axis="columns",
)
df.columns = ["Actual", "Predict"]
df["index"] = pd.to_datetime(df.index)

# plot the prediction series
fig = px.scatter(df, x="index", y="Predict")
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
