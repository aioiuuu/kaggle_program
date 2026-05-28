import pandas as pd
import numpy as np
import os
import joblib
import traceback
from datetime import datetime

def log_message(msg, log_file):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_line = f"[{timestamp}] {msg}\n"
    print(msg)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(log_line)

def main():
    log_file = r"d:\Trae program\python program\python program\kaggle\training_log_optimized.txt"
    
    try:
        log_message("=" * 60, log_file)
        log_message("[OPTIMIZED MODEL TRAINING STARTED]", log_file)
        log_message("=" * 60, log_file)
        
        DATA_PATH = r"d:\Trae program\python program\python program\kaggle\Store Sales (Time Series Forecasting)"
        OUTPUT_PATH = r"d:\Trae program\python program\python program\kaggle\model_results"
        os.makedirs(OUTPUT_PATH, exist_ok=True)
        
        log_message("[STEP 1] Loading data...", log_file)
        train = pd.read_csv(f"{DATA_PATH}/train.csv", parse_dates=['date'])
        test = pd.read_csv(f"{DATA_PATH}/test.csv", parse_dates=['date'])
        stores = pd.read_csv(f"{DATA_PATH}/stores.csv")
        oil = pd.read_csv(f"{DATA_PATH}/oil.csv", parse_dates=['date'])
        holidays = pd.read_csv(f"{DATA_PATH}/holidays_events.csv", parse_dates=['date'])
        transactions = pd.read_csv(f"{DATA_PATH}/transactions.csv", parse_dates=['date'])
        log_message(f"  Train: {train.shape}, Test: {test.shape}", log_file)
        
        log_message("[STEP 2] Preprocessing holidays...", log_file)
        holidays['is_holiday'] = 1
        holidays['is_national'] = (holidays['locale'] == 'National').astype(int)
        holidays = holidays[['date', 'locale_name', 'is_holiday', 'is_national']]
        
        log_message("[STEP 3] Preprocessing oil...", log_file)
        oil['dcoilwtico'] = oil['dcoilwtico'].ffill().bfill()
        
        log_message("[STEP 4] Creating time features...", log_file)
        for df in [train, test]:
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df['day'] = df['date'].dt.day
            df['dayofweek'] = df['date'].dt.dayofweek
            df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
            df['quarter'] = df['date'].dt.quarter
        
        log_message("[STEP 5] Adding lag features...", log_file)
        lags = [7, 14, 21, 30]
        for lag in lags:
            train[f'lag_{lag}'] = train.groupby(['store_nbr', 'family'])['sales'].shift(lag)
        
        log_message("[STEP 6] Adding rolling features...", log_file)
        for window in [7, 14, 30]:
            rolling_result = train.groupby(['store_nbr', 'family'])['sales'].rolling(window).mean()
            train[f'rolling_mean_{window}'] = rolling_result.values
        
        log_message("[STEP 7] Merging data...", log_file)
        train = train.merge(stores, on='store_nbr', how='left')
        test = test.merge(stores, on='store_nbr', how='left')
        train = train.merge(oil, on='date', how='left')
        test = test.merge(oil, on='date', how='left')
        train = train.merge(transactions, on=['date', 'store_nbr'], how='left')
        test['transactions'] = train['transactions'].median()
        
        log_message("[STEP 8] Encoding categorical features...", log_file)
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
        
        log_message("[STEP 9] Preparing features...", log_file)
        feature_columns = [
            'store_nbr', 'family', 'onpromotion', 'city', 'state', 'type', 'cluster',
            'year', 'month', 'day', 'dayofweek', 'is_weekend', 'quarter',
            'dcoilwtico', 'transactions',
            'lag_7', 'lag_14', 'lag_21', 'lag_30',
            'rolling_mean_7', 'rolling_mean_14', 'rolling_mean_30'
        ]
        
        train = train.dropna(subset=feature_columns + ['sales'])
        X = train[feature_columns]
        y = np.log1p(train['sales'])
        
        log_message("[STEP 10] Training LightGBM model...", log_file)
        import lightgbm as lgb
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_squared_error
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
        
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'num_leaves': 127,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'random_state': 42,
            'verbosity': -1
        }
        
        lgb_train = lgb.Dataset(X_train, label=y_train)
        lgb_val = lgb.Dataset(X_val, label=y_val)
        callbacks = [lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=50)]
        
        model = lgb.train(params, lgb_train, num_boost_round=500,
                         valid_sets=[lgb_train, lgb_val],
                         valid_names=['train', 'val'],
                         callbacks=callbacks)
        
        log_message("[STEP 11] Evaluating model...", log_file)
        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        log_message(f"  Validation RMSE: {rmse:.4f}", log_file)
        
        log_message("[STEP 12] Making predictions...", log_file)
        test['sales'] = 0
        for lag in lags:
            test[f'lag_{lag}'] = test.groupby(['store_nbr', 'family'])['sales'].shift(lag)
        for window in [7, 14, 30]:
            rolling_result = test.groupby(['store_nbr', 'family'])['sales'].rolling(window).mean()
            test[f'rolling_mean_{window}'] = rolling_result.values
        
        test = test.fillna(0)
        X_test = test[feature_columns]
        predictions = model.predict(X_test)
        predictions = np.expm1(predictions)
        predictions = np.maximum(predictions, 0)
        
        submission = pd.DataFrame({'id': test['id'], 'sales': predictions})
        submission_path = f"{OUTPUT_PATH}/submission_optimized.csv"
        submission.to_csv(submission_path, index=False)
        log_message(f"  Predictions saved to: {submission_path}", log_file)
        
        model_path = f"{OUTPUT_PATH}/sales_model_optimized.pkl"
        joblib.dump({'model': model, 'label_encoders': label_encoders, 'feature_columns': feature_columns}, model_path)
        log_message(f"  Model saved to: {model_path}", log_file)
        
        log_message("=" * 60, log_file)
        log_message("[TRAINING COMPLETED SUCCESSFULLY]", log_file)
        log_message("=" * 60, log_file)
        
    except Exception as e:
        error_msg = f"ERROR: {str(e)}\n{traceback.format_exc()}"
        log_message(error_msg, log_file)
        raise

if __name__ == "__main__":
    main()
