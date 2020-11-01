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

# In[1]: Prepare the data for modeling

data = pd.read_csv("test/bikes.csv")

# convert season and weather into binary variables
data["season"] = data["season"].astype(str)
data["weather"] = data["weather"].astype(str)
data = pd.get_dummies(data, columns=["season", "weather"])

# # convert datetime to a date time object
# data["datetime"] = pd.to_datetime(data["datetime"])
# timestamps = pd.Series(data["datetime"])

# # collect the hour as binary variables
# hours = pd.get_dummies(timestamps.dt.hour.astype(str))
# hours.columns = [f"Hour_{c}" for c in hours.columns]

# # collect the weekday as binary variables
# weekday = pd.get_dummies(timestamps.dt.weekday.astype(str))
# weekday.columns = [f"Weekday_{c}" for c in weekday.columns]

# # collect the week as binary variables
# week = pd.get_dummies(timestamps.dt.isocalendar().week.astype(str))
# week.columns = [f"Week_{c}" for c in week.columns]

# # collect the month as binary variables
# month = pd.get_dummies(timestamps.dt.month.astype(str))
# month.columns = [f"Month_{c}" for c in month.columns]

# # compute rolling statistics on the output
# Avg = data[["count"]].rolling(6).mean()
# Avg.columns = ["avg_count"]
# Std = data[["count"]].rolling(6).std()
# Std.columns = ["std_count"]
# Min = data[["count"]].rolling(6).min()
# Min.columns = ["min_count"]
# Max = data[["count"]].rolling(6).max()
# Max.columns = ["max_count"]

# # add the features to the data
# data = pd.concat(
#     [data, hours, weekday, week, month, Avg, Std, Min, Max],
#     axis="columns",
# )

# save the data
data.to_csv("test/bikes_v2.csv", index=False)

# import the configuration for modeling
with open("test/bikes.json") as f:
    config = json.load(f)

# update the configuration for the new version of the data
config["csv"] = "test/bikes_v2.csv"
config["inputs"] = data.drop(
    columns=["datetime", "count", "casual", "registered"]
).columns.tolist()
# config["resolution"] = None

# In[2]: Model the data

# produce a rolling forecast
model = Regression(**config)
model.roll(verbose=True)
print(f"Average Error: {np.round(model._error.mean()[0] * 100, 2)}%")

# In[3]: Analyze the model

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
