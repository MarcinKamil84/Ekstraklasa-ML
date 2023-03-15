print("<<< START >>>")

# IMPORTS

print("Importing modules...")

# HIDING PYCARET LOGS

import webbrowser
import os
os.environ["PYCARET_CUSTOM_LOGGING_LEVEL"] = "CRITICAL"

# General and feature engineering

import pandas as pd
import numpy as np
import re
from datetime import date, datetime, timedelta

# Machine learning

from imblearn.over_sampling import RandomOverSampler
from pycaret.classification import *


# # CRAWLER

# Executing crawler file

question1 = input("Crawl for recent data? (y/n): ")

if question1 == "y":
    
    print("Crawling for updates...")
    
    import scripts.crawler as crawler

    crawler.run_spider()

else:
    
    print("Skipping updates...")
        
    pass


# # Setting date span

print("Chose games dates to predict (yyyy-mm-dd):")

start_d_input = input("Start date: ")
end_d_input = input("End date: ")

start_d = np.datetime64(start_d_input)
end_d = np.datetime64(end_d_input)


# FEATURE ENGINEERING

print("Feature engineering...")

exec(open('./scripts/feature_engineering.py').read())


# MACHINE LEARNING

# ## Result model building

question2 = input("Train a new model? (y/n): ")

if question2 == "y":
    
    print("Training new model...")
    
    exec(open('./scripts/model_training.py').read())

else:
    
    print("Skipping new model training...")


# PREDICTIONS

print("Making predictions...")

exec(open('./scripts/predict_models.py').read())

print("<<< END >>>")