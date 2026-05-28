import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier  # 使用 XGBoost

# ======================
# 1. 读取训练数据
# ======================
df = pd.read_csv('train.csv')
y = df['Transported'].astype(int)
x = df.drop(columns=['PassengerId', 'Name', 'Transported'])

# ======================
# 2. 舱位特征拆分
# ======================
def extract_cabin_features(cabin):
    if pd.isna(cabin):
        return pd.Series(['Unknown', -1, 'Unknown'])
    parts = cabin.split('/')
    if len(parts) == 3:
        deck, room, side = parts
        return pd.Series([deck, int(room), side])
    else:
        return pd.Series(['Unknown', -1, 'Unknown'])

cabin_features = x['Cabin'].apply(extract_cabin_features)
cabin_features.columns = ['Deck', 'RoomNumber', 'Side']
x = pd.concat([x.drop(columns=['Cabin']), cabin_features], axis=1)

# ======================
# 3. 布尔值转数字
# ======================
x['CryoSleep'] = x['CryoSleep'].map({True: 1, False: 0})
x['VIP'] = x['VIP'].map({True: 1, False: 0})

# ======================
# 4. 消费特征处理
# ======================
expense_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
x[expense_cols] = x[expense_cols].fillna(0)
x['TotalSpending'] = x[expense_cols].sum(axis=1)

# ======================
# 5. 缺失值填充
# ======================
x['Age'] = x['Age'].fillna(x['Age'].median())
x['HomePlanet'] = x['HomePlanet'].fillna(x['HomePlanet'].mode()[0])
x['Destination'] = x['Destination'].fillna(x['Destination'].mode()[0])
x['RoomNumber'] = x['RoomNumber'].astype(int)

# ======================
# 6. 类别特征编码
# ======================
categorical_cols = ['HomePlanet', 'Destination', 'Deck', 'Side']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    x[col] = x[col].fillna('Unknown')
    x[col] = le.fit_transform(x[col])
    label_encoders[col] = le

# ======================
# 7. 数值特征标准化
# ======================
numeric_cols = ['Age', 'RoomNumber', 'RoomService', 'FoodCourt',
                'Spa', 'ShoppingMall', 'VRDeck', 'TotalSpending']
scaler = StandardScaler()
x[numeric_cols] = scaler.fit_transform(x[numeric_cols])

# ======================
# 8. 划分训练集 & 测试集
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

# ======================
# 9. 训练 XGBoost 模型（准确率更高）
# ======================
xgb_model = XGBClassifier(
    n_estimators=500,  # 更多树
    max_depth=6,  # 深度适中，防止过拟合
    learning_rate=0.03,  # 更小学习率，更精准
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=1,  # 正则化，防过拟合
    reg_lambda=1,
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False
)
xgb_model.fit(X_train, y_train)

# 评估
y_pred = xgb_model.predict(X_test)
print(f'✅ XGBoost 验证集准确率: {accuracy_score(y_test, y_pred) * 100:.2f}%')

# 特征重要性
importance = pd.Series(xgb_model.feature_importances_, index=X_train.columns)
print("\n🔥 Top 10 特征重要性:")
print(importance.sort_values(ascending=False).head(10))

# ======================
# 10. 读取 test.csv 并预测
# ======================
test_df = pd.read_csv('test.csv')
test_ids = test_df['PassengerId']
x_test = test_df.drop(columns=['PassengerId', 'Name'])

# ----------------------
# 对测试集做完全一样的预处理
# ----------------------
# 舱位处理
cabin_features_test = x_test['Cabin'].apply(extract_cabin_features)
cabin_features_test.columns = ['Deck', 'RoomNumber', 'Side']
x_test = pd.concat([x_test.drop(columns=['Cabin']), cabin_features_test], axis=1)

# 布尔转换
x_test['CryoSleep'] = x_test['CryoSleep'].map({True: 1, False: 0})
x_test['VIP'] = x_test['VIP'].map({True: 1, False: 0})

# 消费特征
x_test[expense_cols] = x_test[expense_cols].fillna(0)
x_test['TotalSpending'] = x_test[expense_cols].sum(axis=1)

# 缺失值
x_test['Age'] = x_test['Age'].fillna(df['Age'].median())
x_test['HomePlanet'] = x_test['HomePlanet'].fillna(df['HomePlanet'].mode()[0])
x_test['Destination'] = x_test['Destination'].fillna(df['Destination'].mode()[0])
x_test['RoomNumber'] = x_test['RoomNumber'].astype(int)

# 类别编码（使用训练集的编码器）
for col in categorical_cols:
    le = label_encoders[col]
    x_test[col] = x_test[col].fillna('Unknown')
    x_test[col] = le.transform(x_test[col])

# 标准化
x_test[numeric_cols] = scaler.transform(x_test[numeric_cols])

# 确保列顺序一致
x_test = x_test[X_train.columns]

# ======================
# 11. 预测并生成提交文件
# ======================
test_predictions = xgb_model.predict(x_test)
predicted_bool = test_predictions.astype(bool)

submission = pd.DataFrame({
    'PassengerId': test_ids,
    'Transported': predicted_bool
})

submission.to_csv('submission.csv', index=False)
print("\n🎉 预测完成！最终提交文件已保存为：submission.csv")