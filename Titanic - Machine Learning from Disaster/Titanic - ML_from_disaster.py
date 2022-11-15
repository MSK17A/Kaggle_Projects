"""
This is a python application to solve this competition problem https://www.kaggle.com/competitions/titanic/overview
Titanic - Machine Learning from Disaster
"""
# Import dependencies
import numpy as np
import tensorflow as tf
import pandas as pd

df = pd.read_csv("dataset/train.csv")