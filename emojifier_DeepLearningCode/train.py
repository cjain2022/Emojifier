# Preparing Dataset
import numpy as np 
import pandas as pd

train=pd.read_csv('dataset/train_emoji.csv',header=None)
test=pd.read_csv('dataset/test_emoji.csv',header=None)

print(train.head())
# - collumn 0 :-> stores the textual data which was input by user
# - collumn 1 :-> stores the id for the emoji which was predicted for the input text
# - Other collumns :-> can be ignored
