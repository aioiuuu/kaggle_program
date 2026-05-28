import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime

DATA_PATH = r"d:\Trae program\python program\python program\kaggle\Store Sales (Time Series Forecasting)"
OUTPUT_PATH = r"d:\Trae program\python program\python program\kaggle\model_results"

os.makedirs(OUTPUT_PATH, exist_ok=True)

print("[INFO] Starting robust model training...")

print("[STEP 1] Loading data...")
train = pd.read_csv(f"{DATA_PATH}/train.csv", parse_dates=['date'])
test = pd.read_csv(f"{DATA_PATH}/test.csv", parse_dates=['date'])
stores = pd.read_csv(f"{DATA_PATH}/stores.csv")
oil = pd.read_csv(f"{DATA_PATH}/oil.csv", parse_dates=['date'])
holidays = pd.read_csv(f"{DATA_PATH}/holidays_events.csv", parse_dates=['date'])

print(f"  Train: {train.shape}, Test: {test.shape}")

print("[STEP 2] Creating time features (no lag/rolling)...")
for df in [train, test]:
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['weekofyear'] = df['date'].dt.isocalendar().week.astype(int)
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['quarter'] = df['date'].dt.quarter
    df['day_of_year'] = df['date'].dt.dayofyear
    df['is_month_start'] = (df['day'] <= 7).astype(int)
    df['is_month_end'] = (df['day'] >= 25).astype(int)

print("[STEP 3] Merging store and oil data...")
train = train.merge(stores, on='store_nbr', how='left')
test = test.merge(stores, on='store_nbr', how='left')

oil['dcoilwtico'] = oil['dcoilwtico'].ffill().bfill()
train = train.merge(oil, on='date', how='left')
test = test.merge(oil, on='date', how='left')

print("[STEP 4] Processing holidays...")
holidays['is_holiday'] = 1
national_holidays = holidays[holidays['locale_name'] == 'Ecuador'][['date', 'is_holiday']].drop_duplicates()
train = train.merge(national_holidays, on='date', how='left')
test = test.merge(national_holidays, on='date', how='left')
train['is_holiday'] = train['is_holiday'].fillna(0)
test['is_holiday'] = test['is_holiday'].fillna(0)

print("[STEP 5] Computing historical statistics (train set)...")
family_stats = train.groupby('family')['sales'].agg(['mean', 'std', 'median']).reset_index()
family_stats.columns = ['family', 'family_mean', 'family_std', 'family_median']

store_stats = train.groupby('store_nbr')['sales'].agg(['mean', 'std', 'median']).reset_index()
store_stats.columns = ['store_nbr', 'store_mean', 'store_std', 'store_median']

store_family_stats = train.groupby(['store_nbr', 'family'])['sales'].agg(['mean', 'median']).reset_index()
store_family_stats.columns = ['store_nbr', 'family', 'store_family_mean', 'store_family_median']

dayofweek_stats = train.groupby(['store_nbr', 'family', 'dayofweek'])['sales'].mean().reset_index()
dayofweek_stats.columns = ['store_nbr', 'family', 'dayofweek', 'dayofweek_sales_mean']

print("[STEP 6] Merging historical statistics...")
train = train.merge(store_stats, on='store_nbr', how='left')
test = test.merge(store_stats, on='store_nbr', how='left')

train = train.merge(family_stats, on='family', how='left')
test = test.merge(family_stats, on='family', how='left')

train = train.merge(store_family_stats, on=['store_nbr', 'family'], how='left')
test = test.merge(store_family_stats, on=['store_nbr', 'family'], how='left')

train = train.merge(dayofweek_stats, on=['store_nbr', 'family', 'dayofweek'], how='left')
test = test.merge(dayofweek_stats, on=['store_nbr', 'family', 'dayofweek'], how='left')

print("[STEP 7] Encoding categorical features...")
from sklearn.preprocessing import LabelEncoder

cat_cols = ['family', 'city', 'state', 'type']
label_encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([train[col], test[col]], axis=0)
    le.fit(combined)
    train[col] = le.transform(train[col])
    test[col] = le.transform(test[col])
    label_encoders[col] = le

print("[STEP 8] Preparing features...")
feature_columns = [
    'store_nbr', 'family', 'onpromotion', 'city', 'state', 'type', 'cluster',
    'year', 'month', 'day', 'dayofweek', 'weekofyear', 'is_weekend', 'quarter',
    'day_of_year', 'is_month_start', 'is_month_end',
    'dcoilwtico', 'is_holiday',
    'store_mean', 'store_std', 'store_median',
    'family_mean', 'family_std', 'family_median',
    'store_family_mean', 'store_family_median',
    'dayofweek_sales_mean'
]

train = train.fillna(0)
test = test.fillna(0)

X = train[feature_columns]
y = np.log1p(train['sales'])
X_test = test[feature_columns]

print(f"  Features: {len(feature_columns)}")
print(f"  Train shape: {X.shape}")

print("[STEP 9] Training LightGBM...")
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 63,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'random_state': 42,
    'verbosity': -1,
    'max_depth': 10,
    'min_child_samples': 20
}

lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_val = lgb.Dataset(X_val, label=y_val)
callbacks = [lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=50)]

model = lgb.train(params, lgb_train, num_boost_round=500,
                 valid_sets=[lgb_train, lgb_val],
                 valid_names=['train', 'val'],
                 callbacks=callbacks)

print("[STEP 10] Evaluating...")
y_pred = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"  Validation RMSE: {rmse:.4f}")

print("[STEP 11] Feature importance:")
importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': model.feature_importance(importance_type='gain')
}).sort_values('importance', ascending=False)
print(importance.head(10).to_string())

print("[STEP 12] Making predictions...")
predictions = model.predict(X_test)
predictions = np.expm1(predictions)
predictions = np.maximum(predictions, 0)

submission = pd.DataFrame({'id': test['id'], 'sales': predictions})
submission.to_csv(f"{OUTPUT_PATH}/submission_robust.csv", index=False)
print(f"  Saved to: {OUTPUT_PATH}/submission_robust.csv")

joblib.dump({
    'model': model,
    'label_encoders': label_encoders,
    'feature_columns': feature_columns
}, f"{OUTPUT_PATH}/sales_model_robust.pkl")
print(f"  Model saved to: {OUTPUT_PATH}/sales_model_robust.pkl")

print("\n[DONE] Robust model training complete!")
