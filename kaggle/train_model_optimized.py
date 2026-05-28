import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

DATA_PATH = r"d:\Trae program\python program\python program\kaggle\Store Sales (Time Series Forecasting)"
OUTPUT_PATH = r"d:\Trae program\python program\python program\kaggle\model_results"

os.makedirs(OUTPUT_PATH, exist_ok=True)

def load_all_data():
    print("[STEP 1] Loading data...")
    train = pd.read_csv(f"{DATA_PATH}/train.csv", parse_dates=['date'])
    test = pd.read_csv(f"{DATA_PATH}/test.csv", parse_dates=['date'])
    stores = pd.read_csv(f"{DATA_PATH}/stores.csv")
    oil = pd.read_csv(f"{DATA_PATH}/oil.csv", parse_dates=['date'])
    holidays = pd.read_csv(f"{DATA_PATH}/holidays_events.csv", parse_dates=['date'])
    transactions = pd.read_csv(f"{DATA_PATH}/transactions.csv", parse_dates=['date'])
    
    print("  Train:", train.shape)
    print("  Test:", test.shape)
    return train, test, stores, oil, holidays, transactions

def preprocess_holidays(holidays):
    print("[STEP 2] Preprocessing holidays...")
    holidays['is_holiday'] = 1
    holidays['is_national'] = (holidays['locale'] == 'National').astype(int)
    holidays['is_regional'] = (holidays['locale'] == 'Regional').astype(int)
    holidays['is_local'] = (holidays['locale'] == 'Local').astype(int)
    holidays['is_transferred'] = holidays['transferred'].astype(int)
    return holidays[['date', 'locale_name', 'is_holiday', 'is_national', 
                     'is_regional', 'is_local', 'is_transferred']]

def preprocess_oil(oil):
    print("[STEP 3] Preprocessing oil prices...")
    oil['dcoilwtico'] = oil['dcoilwtico'].ffill().bfill()
    oil['oil_change'] = oil['dcoilwtico'].diff()
    oil['oil_pct_change'] = oil['dcoilwtico'].pct_change() * 100
    oil['oil_ma7'] = oil['dcoilwtico'].rolling(7).mean()
    oil['oil_ma30'] = oil['dcoilwtico'].rolling(30).mean()
    oil = oil.fillna(0)
    return oil

def create_time_features(df):
    print("[STEP 4] Creating time features...")
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['weekofyear'] = df['date'].dt.isocalendar().week
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['quarter'] = df['date'].dt.quarter
    df['month_start'] = (df['day'] <= 7).astype(int)
    df['month_middle'] = ((df['day'] > 7) & (df['day'] <= 21)).astype(int)
    df['month_end'] = (df['day'] > 21).astype(int)
    
    df['is_january'] = (df['month'] == 1).astype(int)
    df['is_december'] = (df['month'] == 12).astype(int)
    
    return df

def add_lag_features(df, target_col='sales', lags=[7, 14, 21, 30]):
    print("[STEP 5] Adding lag features...")
    for lag in lags:
        df[f'lag_{lag}'] = df.groupby(['store_nbr', 'family'])[target_col].shift(lag)
    return df

def add_rolling_features(df, target_col='sales', windows=[7, 14, 30]):
    print("[STEP 6] Adding rolling features...")
    for window in windows:
        df[f'rolling_mean_{window}'] = df.groupby(['store_nbr', 'family'])[target_col].rolling(window).mean().reset_index(0, drop=True)
        df[f'rolling_std_{window}'] = df.groupby(['store_nbr', 'family'])[target_col].rolling(window).std().reset_index(0, drop=True)
        df[f'rolling_max_{window}'] = df.groupby(['store_nbr', 'family'])[target_col].rolling(window).max().reset_index(0, drop=True)
        df[f'rolling_min_{window}'] = df.groupby(['store_nbr', 'family'])[target_col].rolling(window).min().reset_index(0, drop=True)
    return df

def merge_all_data(train, test, stores, oil, holidays, transactions):
    print("[STEP 7] Merging datasets...")
    
    train = train.merge(stores, on='store_nbr', how='left')
    test = test.merge(stores, on='store_nbr', how='left')
    
    train = train.merge(oil, on='date', how='left')
    test = test.merge(oil, on='date', how='left')
    
    train = train.merge(transactions, on=['date', 'store_nbr'], how='left')
    test['transactions'] = np.nan
    
    holiday_city = holidays.copy()
    holiday_city.rename(columns={'locale_name': 'city'}, inplace=True)
    train = train.merge(holiday_city, on=['date', 'city'], how='left')
    test = test.merge(holiday_city, on=['date', 'city'], how='left')
    
    national_holidays = holidays[holidays['locale_name'] == 'Ecuador'][['date', 'is_national']]
    national_holidays.drop_duplicates(inplace=True)
    train = train.merge(national_holidays, on='date', how='left', suffixes=('', '_national'))
    test = test.merge(national_holidays, on='date', how='left', suffixes=('', '_national'))
    
    fill_values = {
        'is_holiday': 0, 'is_national': 0, 'is_regional': 0, 'is_local': 0,
        'is_transferred': 0, 'is_national_national': 0,
        'oil_change': 0, 'oil_pct_change': 0, 'oil_ma7': oil['dcoilwtico'].mean(),
        'oil_ma30': oil['dcoilwtico'].mean(),
        'transactions': train['transactions'].median()
    }
    
    train.fillna(fill_values, inplace=True)
    test.fillna(fill_values, inplace=True)
    
    return train, test

def encode_categorical(train, test):
    print("[STEP 8] Encoding categorical features...")
    cat_cols = ['family', 'city', 'state', 'type']
    label_encoders = {}
    
    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([train[col], test[col]], axis=0)
        le.fit(combined)
        train[col] = le.transform(train[col])
        test[col] = le.transform(test[col])
        label_encoders[col] = le
    
    return label_encoders

def prepare_features(train, test):
    print("[STEP 9] Preparing features...")
    
    feature_columns = [
        'store_nbr', 'family', 'onpromotion', 'city', 'state', 'type', 'cluster',
        'year', 'month', 'day', 'dayofweek', 'weekofyear', 'is_weekend', 'quarter',
        'month_start', 'month_middle', 'month_end', 'is_january', 'is_december',
        'dcoilwtico', 'oil_change', 'oil_pct_change', 'oil_ma7', 'oil_ma30',
        'transactions', 'is_holiday', 'is_national', 'is_regional', 'is_local',
        'is_transferred', 'is_national_national',
        'lag_7', 'lag_14', 'lag_21', 'lag_30',
        'rolling_mean_7', 'rolling_std_7', 'rolling_max_7', 'rolling_min_7',
        'rolling_mean_14', 'rolling_std_14', 'rolling_max_14', 'rolling_min_14',
        'rolling_mean_30', 'rolling_std_30', 'rolling_max_30', 'rolling_min_30'
    ]
    
    X_train = train[feature_columns]
    y_train = np.log1p(train['sales'])
    X_test = test[feature_columns]
    
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
    
    return X_train, X_test, y_train, feature_columns

def train_lgbm_cv(X, y, feature_columns, n_splits=5):
    print("[STEP 10] Training LightGBM with Cross Validation...")
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    models = []
    val_scores = []
    
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 127,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'random_state': 42,
        'verbosity': -1,
        'max_depth': -1,
        'min_child_samples': 20,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1
    }
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"  Fold {fold + 1}/{n_splits}...")
        
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        lgb_train = lgb.Dataset(X_tr, label=y_tr)
        lgb_val = lgb.Dataset(X_val, label=y_val)
        
        callbacks = [lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=50)]
        
        model = lgb.train(params, lgb_train, num_boost_round=500,
                         valid_sets=[lgb_train, lgb_val],
                         valid_names=['train', 'val'],
                         callbacks=callbacks)
        
        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        val_scores.append(rmse)
        models.append(model)
        
        print(f"  Fold {fold + 1} RMSE: {rmse:.4f}")
    
    print(f"\n  Average CV RMSE: {np.mean(val_scores):.4f} (std: {np.std(val_scores):.4f})")
    
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': np.mean([model.feature_importance(importance_type='gain') for model in models], axis=0)
    }).sort_values('importance', ascending=False)
    
    print("\n  Top 10 Features:")
    print(feature_importance.head(10).to_string())
    
    return models, feature_importance

def predict_ensemble(models, X_test):
    print("[STEP 11] Making ensemble predictions...")
    
    predictions = []
    for model in models:
        pred = model.predict(X_test)
        predictions.append(pred)
    
    final_pred = np.mean(predictions, axis=0)
    final_pred = np.expm1(final_pred)
    final_pred = np.maximum(final_pred, 0)
    
    return final_pred

def main():
    print("=" * 60)
    print("[OPTIMIZED MODEL TRAINING]")
    print("=" * 60)
    print("Start time:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    train, test, stores, oil, holidays, transactions = load_all_data()
    holidays = preprocess_holidays(holidays)
    oil = preprocess_oil(oil)
    
    train = create_time_features(train)
    test = create_time_features(test)
    
    train = add_lag_features(train)
    train = add_rolling_features(train)
    
    train, test = merge_all_data(train, test, stores, oil, holidays, transactions)
    label_encoders = encode_categorical(train, test)
    
    test['sales'] = 0
    test = add_lag_features(test)
    test = add_rolling_features(test)
    
    X_train, X_test, y_train, feature_columns = prepare_features(train, test)
    
    models, feature_importance = train_lgbm_cv(X_train, y_train, feature_columns)
    
    predictions = predict_ensemble(models, X_test)
    
    submission = pd.DataFrame({
        'id': test['id'],
        'sales': predictions
    })
    
    submission_path = f"{OUTPUT_PATH}/submission_optimized.csv"
    submission.to_csv(submission_path, index=False)
    print("\n  Predictions saved to:", submission_path)
    
    model_path = f"{OUTPUT_PATH}/sales_model_optimized.pkl"
    joblib.dump({
        'models': models,
        'label_encoders': label_encoders,
        'feature_columns': feature_columns,
        'feature_importance': feature_importance
    }, model_path)
    print("  Model saved to:", model_path)
    
    print("\nEnd time:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("=" * 60)

if __name__ == "__main__":
    main()
