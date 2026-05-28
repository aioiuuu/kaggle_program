import pandas as pd
import numpy as np
import os

print("Testing basic imports...")
try:
    import lightgbm as lgb
    print("[OK] LightGBM OK")
except Exception as e:
    print("[FAIL] LightGBM failed:", e)

print("\nTesting data loading...")
DATA_PATH = r"d:\Trae program\python program\python program\kaggle\Store Sales (Time Series Forecasting)"

try:
    train = pd.read_csv(f"{DATA_PATH}/train.csv", parse_dates=['date'])
    print("[OK] Train data loaded:", train.shape)
except Exception as e:
    print("[FAIL] Train data failed:", e)

try:
    stores = pd.read_csv(f"{DATA_PATH}/stores.csv")
    print("[OK] Stores data loaded:", stores.shape)
except Exception as e:
    print("[FAIL] Stores data failed:", e)

print("\nTesting feature creation...")
train['month'] = train['date'].dt.month
train['dayofweek'] = train['date'].dt.dayofweek
print("[OK] Time features created")

print("\nTesting merge...")
train_merged = train.merge(stores, on='store_nbr', how='left')
print("[OK] Merged data:", train_merged.shape)

print("\nTesting encoding...")
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_merged['family_encoded'] = le.fit_transform(train_merged['family'])
print("[OK] Label encoding OK")

print("\nAll tests passed! Ready to train model.")
