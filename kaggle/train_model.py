import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder

try:
    import xgboost as xgb
    XGB_INSTALLED = True
except ImportError:
    XGB_INSTALLED = False
    print("[WARNING] XGBoost not installed, will use LightGBM")

try:
    import lightgbm as lgb
    LGB_INSTALLED = True
except ImportError:
    LGB_INSTALLED = False
    print("[WARNING] LightGBM not installed")

DATA_PATH = r"d:\Trae program\python program\python program\kaggle\Store Sales (Time Series Forecasting)"
OUTPUT_PATH = r"d:\Trae program\python program\python program\kaggle\model_results"

os.makedirs(OUTPUT_PATH, exist_ok=True)

class SalesPredictor:
    def __init__(self):
        self.models = {}
        self.label_encoders = {}
        self.feature_columns = []
        self.best_model = None
        
    def load_data(self):
        print(">>> Loading data...")
        self.train = pd.read_csv(f"{DATA_PATH}/train.csv", parse_dates=['date'])
        self.test = pd.read_csv(f"{DATA_PATH}/test.csv", parse_dates=['date'])
        self.stores = pd.read_csv(f"{DATA_PATH}/stores.csv")
        self.oil = pd.read_csv(f"{DATA_PATH}/oil.csv", parse_dates=['date'])
        self.holidays = pd.read_csv(f"{DATA_PATH}/holidays_events.csv", parse_dates=['date'])
        self.transactions = pd.read_csv(f"{DATA_PATH}/transactions.csv", parse_dates=['date'])
        self.sample_submission = pd.read_csv(f"{DATA_PATH}/sample_submission.csv")
        print(f"  Train: {self.train.shape}")
        print(f"  Test: {self.test.shape}")
        
    def preprocess_holidays(self):
        print(">>> Preprocessing holidays...")
        self.holidays['is_holiday'] = 1
        self.holidays['is_national_holiday'] = (self.holidays['locale'] == 'National').astype(int)
        self.holidays['is_regional_holiday'] = (self.holidays['locale'] == 'Regional').astype(int)
        self.holidays['is_local_holiday'] = (self.holidays['locale'] == 'Local').astype(int)
        self.holidays = self.holidays[['date', 'locale_name', 'is_holiday', 'is_national_holiday', 
                                      'is_regional_holiday', 'is_local_holiday']]
        
    def preprocess_oil(self):
        print(">>> Preprocessing oil prices...")
        self.oil['dcoilwtico'] = self.oil['dcoilwtico'].ffill()
        self.oil['dcoilwtico'] = self.oil['dcoilwtico'].bfill()
        
    def create_time_features(self, df):
        print(">>> Creating time features...")
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
        
        return df
    
    def merge_data(self):
        print(">>> Merging datasets...")
        
        self.train = self.train.merge(self.stores, on='store_nbr', how='left')
        self.train = self.train.merge(self.oil, on='date', how='left')
        self.train = self.train.merge(self.transactions, on=['date', 'store_nbr'], how='left')
        
        train_holidays = self.holidays.copy()
        train_holidays.rename(columns={'locale_name': 'city'}, inplace=True)
        self.train = self.train.merge(train_holidays, on=['date', 'city'], how='left')
        
        national_holidays = self.holidays[self.holidays['locale_name'] == 'Ecuador'].copy()
        national_holidays = national_holidays[['date', 'is_national_holiday']]
        national_holidays.drop_duplicates(inplace=True)
        self.train = self.train.merge(national_holidays, on='date', how='left', suffixes=('', '_national'))
        
        self.test = self.test.merge(self.stores, on='store_nbr', how='left')
        self.test = self.test.merge(self.oil, on='date', how='left')
        
        test_holidays = self.holidays.copy()
        test_holidays.rename(columns={'locale_name': 'city'}, inplace=True)
        self.test = self.test.merge(test_holidays, on=['date', 'city'], how='left')
        
        self.test = self.test.merge(national_holidays, on='date', how='left', suffixes=('', '_national'))
        
        self.train.fillna({
            'is_holiday': 0,
            'is_national_holiday': 0,
            'is_regional_holiday': 0,
            'is_local_holiday': 0,
            'is_national_holiday_national': 0,
            'transactions': self.train['transactions'].median()
        }, inplace=True)
        
        self.test.fillna({
            'is_holiday': 0,
            'is_national_holiday': 0,
            'is_regional_holiday': 0,
            'is_local_holiday': 0,
            'is_national_holiday_national': 0,
            'dcoilwtico': self.oil['dcoilwtico'].mean()
        }, inplace=True)
        
    def encode_categorical(self):
        print(">>> Encoding categorical features...")
        
        cat_cols = ['family', 'city', 'state', 'type']
        
        for col in cat_cols:
            le = LabelEncoder()
            combined = pd.concat([self.train[col], self.test[col]], axis=0)
            le.fit(combined)
            self.train[col] = le.transform(self.train[col])
            self.test[col] = le.transform(self.test[col])
            self.label_encoders[col] = le
            
    def prepare_features(self):
        print(">>> Preparing features...")
        
        self.feature_columns = [
            'store_nbr', 'family', 'onpromotion', 'city', 'state', 'type', 'cluster',
            'year', 'month', 'day', 'dayofweek', 'weekofyear', 'is_weekend', 'quarter',
            'month_start', 'month_middle', 'month_end', 'dcoilwtico', 'transactions',
            'is_holiday', 'is_national_holiday', 'is_regional_holiday', 'is_local_holiday'
        ]
        
        X = self.train[self.feature_columns]
        y = np.log1p(self.train['sales'] + 1)
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=True)
        
        return X_train, X_val, y_train, y_val
    
    def train_xgboost(self, X_train, X_val, y_train, y_val):
        print(">>> Training XGBoost model...")
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 8,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'tree_method': 'hist',
            'random_state': 42
        }
        
        watchlist = [(dtrain, 'train'), (dval, 'val')]
        model = xgb.train(params, dtrain, num_boost_round=1000, 
                         evals=watchlist, early_stopping_rounds=50, verbose_eval=50)
        
        y_pred = model.predict(dval)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)
        print(f"  XGBoost RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        
        return model, rmse
    
    def train_lightgbm(self, X_train, X_val, y_train, y_val):
        print(">>> Training LightGBM model...")
        
        lgb_train = lgb.Dataset(X_train, label=y_train)
        lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
        
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 63,
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'random_state': 42
        }
        
        model = lgb.train(params, lgb_train, num_boost_round=1000,
                         valid_sets=[lgb_train, lgb_val], 
                         valid_names=['train', 'val'],
                         early_stopping_rounds=50, verbose_eval=50)
        
        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)
        print(f"  LightGBM RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        
        return model, rmse
    
    def train_models(self):
        X_train, X_val, y_train, y_val = self.prepare_features()
        
        best_rmse = float('inf')
        
        if XGB_INSTALLED:
            xgb_model, xgb_rmse = self.train_xgboost(X_train, X_val, y_train, y_val)
            self.models['xgboost'] = xgb_model
            if xgb_rmse < best_rmse:
                best_rmse = xgb_rmse
                self.best_model = ('xgboost', xgb_model)
        
        if LGB_INSTALLED:
            lgb_model, lgb_rmse = self.train_lightgbm(X_train, X_val, y_train, y_val)
            self.models['lightgbm'] = lgb_model
            if lgb_rmse < best_rmse:
                best_rmse = lgb_rmse
                self.best_model = ('lightgbm', lgb_model)
        
        print(f"\n>>> Best model: {self.best_model[0]} with RMSE: {best_rmse:.4f}")
        
    def predict(self):
        print(">>> Making predictions...")
        
        test_features = self.test[self.feature_columns]
        
        if self.best_model[0] == 'xgboost':
            dtest = xgb.DMatrix(test_features)
            predictions = self.best_model[1].predict(dtest)
        else:
            predictions = self.best_model[1].predict(test_features)
        
        predictions = np.expm1(predictions) - 1
        predictions = np.maximum(predictions, 0)
        
        submission = pd.DataFrame({
            'id': self.test['id'],
            'sales': predictions
        })
        
        submission.to_csv(f"{OUTPUT_PATH}/submission.csv", index=False)
        print(f"  Predictions saved to {OUTPUT_PATH}/submission.csv")
        
        return submission
    
    def save_model(self):
        print(">>> Saving model...")
        joblib.dump({
            'models': self.models,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'best_model_name': self.best_model[0]
        }, f"{OUTPUT_PATH}/sales_predictor.pkl")
        print(f"  Model saved to {OUTPUT_PATH}/sales_predictor.pkl")
    
    def run(self):
        print("=" * 60)
        print(">>> Sales Prediction Model Training")
        print("=" * 60)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        self.load_data()
        self.preprocess_holidays()
        self.preprocess_oil()
        self.train = self.create_time_features(self.train)
        self.test = self.create_time_features(self.test)
        self.merge_data()
        self.encode_categorical()
        self.train_models()
        self.predict()
        self.save_model()
        
        print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

if __name__ == "__main__":
    predictor = SalesPredictor()
    predictor.run()
