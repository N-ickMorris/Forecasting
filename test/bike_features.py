# -*- coding: utf-8 -*-
"""
Engineering features for bike share demand

@author: Nick
"""

import re
import numpy as np
import pandas as pd
import datetime
import calendar
import scipy.cluster.hierarchy as sch


def week_of_month(tgtdate):
    """
    Get the week of the current month for a given timestamp "tgtdate"
    Weeks of month = 1, 2, 3, 4 = first week, second week, third week, forth week

    Reference: https://stackoverflow.com/questions/25249033/week-of-a-month-pandas
    """
    # tgtdate = tgtdate.to_datetime()

    days_this_month = calendar.mdays[tgtdate.month]
    for i in range(1, days_this_month):
        d = datetime.datetime(tgtdate.year, tgtdate.month, i)
        if d.day - d.weekday() > 0:
            startdate = d
            break
    # now we can use the modulo 7 approach
    return (tgtdate - startdate).days // 7 + 1 + 1


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


def corr_cluster(df, method="ward"):
    # reorder columns based on hierarchical clustering
    X = df.corr().values
    d = sch.distance.pdist(X)
    L = sch.linkage(d, method=method)
    ind = sch.fcluster(L, 0.5 * d.max(), "distance")
    columns = [df.columns.tolist()[i] for i in list((np.argsort(ind)))]
    df = df.reindex(columns, axis=1)

    # compute the correlation matrix
    return df.corr()


# read in the data
data = pd.read_csv("test/bikes.csv")

# convert season into a string
conditions = [
    (data["season"] == 1),
    (data["season"] == 2),
    (data["season"] == 3),
    (data["season"] == 4),
]
choices = ["spring", "summer", "fall", "winter"]
data["season"] = np.select(conditions, choices, default=None)

# convert weather into a string
conditions = [
    (data["weather"] == 1),
    (data["weather"] == 2),
    (data["weather"] == 3),
    (data["weather"] == 4),
]
choices = [
    "Clear, Few clouds, Partly cloudy, Partly cloudy",
    "Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist",
    "Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds",
    "Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog",
]
data["weather"] = np.select(conditions, choices, default=None)

# convert holiday and workingday into booleans
data["holiday"] = data["holiday"] == 1
data["workingday"] = data["workingday"] == 1

# compute the fraction of users that are registered users
data["registered_fraction"] = data["registered"] / (data["casual"] + data["registered"])

# extract features from the time stamp
timestamps = pd.to_datetime(data["datetime"])
data["hour"] = timestamps.dt.hour
data["midday"] = np.where(data["hour"] < 12, "AM", "PM")
data["weekday"] = timestamps.dt.day_name()
data["week of month"] = timestamps.apply(week_of_month)
data["week of year"] = timestamps.dt.isocalendar().week
data["month"] = timestamps.dt.month_name()
data["quarter"] = timestamps.dt.quarter
data["year"] = timestamps.dt.year

# compute rolling statistics on the output
window = 3
data[f"count: {window} hour average"] = data[["count"]].rolling(window).mean()
data[f"count: {window} hour standard deviation"] = data[["count"]].rolling(window).std()
data[f"count: {window} hour minimum"] = data[["count"]].rolling(window).min()
data[f"count: {window} hour maximum"] = data[["count"]].rolling(window).max()

# compute lag terms on the output
lags = series_to_supervised(
    data[["count"]], n_backward=24 * 7, n_forward=0
)  # lagging up to a week

# compute autocorrealtions
autocorr = lags.corrwith(data["count"]).reset_index()
autocorr.columns = ["Lag", "Correlation"]

# compute correlation matrix
corr = corr_cluster(
    data[["temp", "atemp", "humidity", "windspeed", "count"]].astype(float)
).reset_index()
corr = pd.melt(corr, id_vars="index")
corr.columns = ["variable 1", "variable 2", "correlation"]

# export the data
data = pd.concat([data, lags], axis="columns").dropna()
data.to_csv("test/bike features.csv", index=False)
autocorr.to_csv("test/bike autocorrelation.csv", index=False)
corr.to_csv("test/bike correlation.csv", index=False)
