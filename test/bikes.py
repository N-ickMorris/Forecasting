"""
Build forecasting models to predict bike share demand

@author: Nick
"""

# add the src directory to the system path
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/') ))

import json
import pandas as pd
from forecast import Forecasting

# import the configuration for modeling
with open('test/config.json') as f:
    config = json.load(f)

# initialize the model
model = Forecasting(**config)
model.test()
x, y = model.reshape_output(model._data[[model.output]])
print(x)
print(y)
