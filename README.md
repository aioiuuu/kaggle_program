

---

##  比赛概览 / Competitions Overview

### 1. Titanic 生存预测
- **任务类型 / Task**: 二分类问题（预测乘客是否存活）
- **核心亮点 / Highlights**:
  - 特征工程：乘客头衔提取、家庭规模构造、票价分箱
  - 模型对比：逻辑回归、随机森林、XGBoost 等多种算法
  - 交叉验证与超参数调优

### 2. 房价预测 / House Price Prediction
- **任务类型 / Task**: 回归问题（预测房屋最终售价）
- **核心亮点 / Highlights**:
  - 缺失值处理、偏态分布修正、异常值清洗
  - 特征缩放、编码与交互特征构建
  - 树模型与线性回归的集成方案

### 3. Spaceship Titanic
- **任务类型 / Task**: 二分类问题（预测乘客是否被传送）
- **核心亮点 / Highlights**:
  - 复杂特征解析：舱位、乘客分组、冷冻睡眠状态等信息提取
  - 自定义预处理流水线，适配比赛特殊数据格式
  - 模型堆叠与融合，提升最终提交成绩

---

##  技术栈与工作流 / Tech Stack & Workflow

### 核心依赖库 / Core Libraries
- `pandas`, `numpy` – 数据处理与预处理
- `matplotlib`, `seaborn` – 探索性数据分析与可视化
- `scikit-learn` – 基准模型、预处理与交叉验证
- `xgboost`, `lightgbm` – 梯度提升模型
- `optuna` / `GridSearchCV` – 超参数优化工具

### 标准工作流 / Standard Workflow
1.  **EDA 探索性分析**：分析数据分布、相关性、缺失值与特征关系
2.  **数据预处理**：数据清洗、编码、标准化与特征工程
3.  **模型构建**：从基准模型开始，逐步迭代优化
4.  **验证与调优**：使用分层 K 折交叉验证，避免过拟合
5.  **生成提交文件**：按比赛要求格式输出预测结果

---

##  运行方式 / How to Run

1.  安装依赖：
    ```bash
    pip install -r requirements.txt
