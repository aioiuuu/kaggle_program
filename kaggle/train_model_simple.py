import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

try:
    import lightgbm as lgb
    print("[INFO] LightGBM imported successfully")
except ImportError as e:
    print(f"[ERROR] LightGBM import failed: {e}")
    raise

DATA_PATH = r"d:\Trae program\python program\python program\kaggle\Store Sales (Time Series Forecasting)"
OUTPUT_PATH = r"d:\Trae program\python program\python program\kaggle\model_results"

os.makedirs(OUTPUT_PATH, exist_ok=True)
print(f"[INFO] Output directory: {OUTPUT_PATH}")

print("[STEP 1] Loading data...")
train = pd.read_csv(f"{DATA_PATH}/train.csv", parse_dates=['date'])
test = pd.read_csv(f"{DATA_PATH}/test.csv", parse_dates=['date'])
stores = pd.read_csv(f"{DATA_PATH}/stores.csv")
oil = pd.read_csv(f"{DATA_PATH}/oil.csv", parse_dates=['date'])
holidays = pd.read_csv(f"{DATA_PATH}/holidays_events.csv", parse_dates=['date'])

print(f"  Train shape: {train.shape}")
print(f"  Test shape: {test.shape}")

print("[STEP 2] Creating time features...")
def create_time_features(df):
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['quarter'] = df['date'].dt.quarter
    return df

train = create_time_features(train)
test = create_time_features(test)

print("[STEP 3] Merging data...")
train = train.merge(stores, on='store_nbr', how='left')
test = test.merge(stores, on='store_nbr', how='left')

oil['dcoilwtico'] = oil['dcoilwtico'].ffill().bfill()
train = train.merge(oil, on='date', how='left')
test = test.merge(oil, on='date', how='left')

print("[STEP 4] Encoding categorical features...")
cat_cols = ['family', 'city', 'state', 'type']
label_encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([train[col], test[col]], axis=0)
    le.fit(combined)
    train[col] = le.transform(train[col])
    test[col] = le.transform(test[col])
    label_encoders[col] = le

print("[STEP 5] Preparing features...")
feature_columns = [
    'store_nbr', 'family', 'onpromotion', 'city', 'state', 'type', 'cluster',
    'year', 'month', 'day', 'dayofweek', 'is_weekend', 'quarter', 'dcoilwtico'
]

X = train[feature_columns]
y = np.log1p(train['sales'])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
print(f"  Train: {X_train.shape}, Validation: {X_val.shape}")

print("[STEP 6] Training LightGBM model...")
lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_val = lgb.Dataset(X_val, label=y_val)

params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 63,
    'learning_rate': 0.1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'random_state': 42,
    'verbosity': 1
}

model = lgb.train(params, lgb_train, num_boost_round=200,
                 valid_sets=[lgb_train, lgb_val],
                 valid_names=['train', 'val'],
                 early_stopping_rounds=30, verbose_eval=20)

print("[STEP 7] Evaluating model...")
y_pred = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"  Validation RMSE: {rmse:.4f}")

print("[STEP 8] Making predictions...")
test_features = test[feature_columns]
predictions = model.predict(test_features)
predictions = np.expm1(predictions)
predictions = np.maximum(predictions, 0)

submission = pd.DataFrame({
    'id': test['id'],
    'sales': predictions
})

submission_path = f"{OUTPUT_PATH}/submission.csv"
submission.to_csv(submission_path, index=False)
print(f"  Predictions saved to {submission_path}")

print("[STEP 9] Saving model...")
model_path = f"{OUTPUT_PATH}/sales_model.pkl"
joblib.dump({
    'model': model,
    'label_encoders': label_encoders,
    'feature_columns': feature_columns
}, model_path)
print(f"  Model saved to {model_path}")

print("\n[COMPLETE] Model training and prediction finished!")
