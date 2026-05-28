# ===================== 1. 导入库 =====================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb

# ===================== 2. 加载数据 =====================
df = pd.read_csv('train.csv')
df = df.drop('Id', axis=1)

# 去掉异常值
df = df[df['SalePrice'] < df['SalePrice'].quantile(0.99)]
df = df[df['SalePrice'] > df['SalePrice'].quantile(0.01)]

X = df.drop('SalePrice', axis=1)
y = df['SalePrice']

# 取对数（大幅提升精度）
y_log = np.log(y)

# 划分训练集测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y_log, test_size=0.3, random_state=42
)

# ===================== 3. 自动区分特征 =====================
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object', 'string']).columns

# ===================== 4. 预处理流水线 =====================
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# ===================== 5. 直接训练 XGBoost（不做 Lasso 筛选） =====================
print("🔹 训练 XGBoost 模型中...")

# 预处理
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# XGBoost 模型
model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=1200,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    random_state=42
)
model.fit(X_train_processed, y_train)

# ===================== 6. 评估 =====================
def metrics(y_true, y_pred):
    return (
        r2_score(y_true, y_pred),
        mean_absolute_error(y_true, y_pred),
        np.sqrt(mean_squared_error(y_true, y_pred))
    )

y_tr_pred = model.predict(X_train_processed)
y_te_pred = model.predict(X_test_processed)

tr_r2, tr_mae, tr_rmse = metrics(y_train, y_tr_pred)
te_r2, te_mae, te_rmse = metrics(y_test, y_te_pred)

print("\n" * 2)
print("=" * 55)
print("🏆 模型训练完成（无报错版）")
print("=" * 55)
print(f"训练集 R²: {tr_r2:.4f} | MAE: {tr_mae:.4f} | RMSE: {tr_rmse:.4f}")
print(f"测试集 R²: {te_r2:.4f} | MAE: {te_mae:.4f} | RMSE: {te_rmse:.4f}")

# ===================== 7. 测试集预测 =====================
print("\n🔹 正在预测 test.csv...")
test_df = pd.read_csv('test.csv')
ids = test_df['Id']
test_df = test_df.drop('Id', axis=1)

test_processed = preprocessor.transform(test_df)
preds_log = model.predict(test_processed)
preds_real = np.exp(preds_log)

submit = pd.DataFrame({
    'Id': ids,
    'SalePrice': preds_real
})
submit.to_csv('房价预测结果_最终版.csv', index=False)

print("✅ 预测完成！已保存：房价预测结果_最终版.csv")
print(submit.head())

# ===================== 8. 画图 =====================
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.figure(figsize=(10,5))
plt.scatter(np.exp(y_test), np.exp(y_te_pred), alpha=0.6, c='crimson')
plt.plot([np.exp(y_test).min(), np.exp(y_test).max()],
         [np.exp(y_test).min(), np.exp(y_test).max()], 'b--', lw=2)
plt.xlabel("真实房价")
plt.ylabel("预测房价")
plt.title(f"R² = {te_r2:.4f}")
plt.grid(alpha=0.3)
plt.show()