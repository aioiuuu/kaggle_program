import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

DATA_PATH = r"d:\Trae program\python program\python program\kaggle\Store Sales (Time Series Forecasting)"
OUTPUT_PATH = r"d:\Trae program\python program\python program\kaggle\eda_results"

os.makedirs(OUTPUT_PATH, exist_ok=True)

def load_data():
    print("=" * 60)
    print(">>> Loading data...")
    print("=" * 60)

    train = pd.read_csv(f"{DATA_PATH}/train.csv", parse_dates=['date'])
    test = pd.read_csv(f"{DATA_PATH}/test.csv", parse_dates=['date'])
    stores = pd.read_csv(f"{DATA_PATH}/stores.csv")
    oil = pd.read_csv(f"{DATA_PATH}/oil.csv", parse_dates=['date'])
    holidays = pd.read_csv(f"{DATA_PATH}/holidays_events.csv", parse_dates=['date'])
    transactions = pd.read_csv(f"{DATA_PATH}/transactions.csv", parse_dates=['date'])
    sample_submission = pd.read_csv(f"{DATA_PATH}/sample_submission.csv")

    return {
        'train': train,
        'test': test,
        'stores': stores,
        'oil': oil,
        'holidays': holidays,
        'transactions': transactions,
        'sample_submission': sample_submission
    }

def explore_basic_info(data):
    print("\n" + "=" * 60)
    print(">>> Dataset Basic Info")
    print("=" * 60)

    for name, df in data.items():
        print(f"\n[{name}]")
        print(f"  Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
        print(f"  Columns: {list(df.columns)}")

def explore_missing_values(data):
    print("\n" + "=" * 60)
    print(">>> Missing Value Analysis")
    print("=" * 60)

    for name, df in data.items():
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        missing_df = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
        missing_df = missing_df[missing_df['Missing Count'] > 0]

        if len(missing_df) > 0:
            print(f"\n[{name}]")
            print(missing_df.to_string())
        else:
            print(f"\n[{name}] No missing values")

def explore_train_data(train):
    print("\n" + "=" * 60)
    print(">>> Train Data Detailed Analysis")
    print("=" * 60)

    print("\n[Data Types]")
    print(train.dtypes)

    print("\n[Numerical Statistics]")
    print(train.describe().round(2).to_string())

    print("\n[Product Families]")
    print(f"  Total: {train['family'].nunique()} families")
    family_counts = train['family'].value_counts()
    for family, count in family_counts.items():
        print(f"    - {family}: {count:,} records")

    print("\n[Number of Stores]")
    print(f"  Total: {train['store_nbr'].nunique()} stores")

    print("\n[Date Range]")
    print(f"  Start: {train['date'].min()}")
    print(f"  End: {train['date'].max()}")
    print(f"  Days: {(train['date'].max() - train['date'].min()).days} days")

    print("\n[Sales Analysis by Family]")
    sales_stats = train.groupby('family')['sales'].agg(['mean', 'std', 'min', 'max', 'sum'])
    sales_stats = sales_stats.sort_values('sum', ascending=False)
    print(sales_stats.round(2).to_string())

    print("\n[Sales = 0 Analysis]")
    zero_sales = train[train['sales'] == 0]
    print(f"  Records with sales = 0: {len(zero_sales):,} ({len(zero_sales)/len(train)*100:.1f}%)")

def explore_stores_data(stores):
    print("\n" + "=" * 60)
    print(">>> Stores Data Detailed Analysis")
    print("=" * 60)

    print("\n[Store Type Distribution]")
    print(stores['type'].value_counts().to_string())

    print("\n[City Distribution]")
    print(stores['city'].value_counts().to_string())

    print("\n[State Distribution]")
    print(stores['state'].value_counts().to_string())

    print("\n[Cluster Distribution]")
    print(f"  Total: {stores['cluster'].nunique()} clusters")
    print(stores['cluster'].value_counts().sort_index().to_string())

def explore_oil_data(oil):
    print("\n" + "=" * 60)
    print(">>> Oil Price Data Detailed Analysis")
    print("=" * 60)

    print(f"\n[Date Range]")
    print(f"  Start: {oil['date'].min()}")
    print(f"  End: {oil['date'].max()}")

    print(f"\n[Oil Price Statistics]")
    oil_clean = oil.dropna()
    print(f"  Valid records: {len(oil_clean)}")
    print(f"  Missing records: {len(oil) - len(oil_clean)}")
    print(f"  Average price: ${oil_clean['dcoilwtico'].mean():.2f}")
    print(f"  Min price: ${oil_clean['dcoilwtico'].min():.2f}")
    print(f"  Max price: ${oil_clean['dcoilwtico'].max():.2f}")

def explore_holidays_data(holidays):
    print("\n" + "=" * 60)
    print(">>> Holidays Data Detailed Analysis")
    print("=" * 60)

    print("\n[Holiday Type Distribution]")
    print(holidays['type'].value_counts().to_string())

    print("\n[Locale Type Distribution]")
    print(holidays['locale'].value_counts().to_string())

    print("\n[Transferred Holidays]")
    transferred = holidays[holidays['transferred'] == True]
    print(f"  Transferred holidays: {len(transferred)}")

def explore_transactions_data(transactions):
    print("\n" + "=" * 60)
    print(">>> Transactions Data Detailed Analysis")
    print("=" * 60)

    print("\n[Daily Transaction Statistics]")
    daily_stats = transactions.groupby('date')['transactions'].agg(['mean', 'std', 'min', 'max'])
    print(f"  Avg daily total transactions: {daily_stats['mean'].mean():.0f}")
    print(f"  Max single day transactions: {daily_stats['max'].max():,}")

    print("\n[Store Transaction Ranking (Top 10)]")
    store_totals = transactions.groupby('store_nbr')['transactions'].sum().sort_values(ascending=False)
    print(store_totals.head(10).to_string())

def create_visualizations(data):
    print("\n" + "=" * 60)
    print(">>> Generating Visualizations...")
    print("=" * 60)

    train = data['train']
    stores = data['stores']
    oil = data['oil']
    transactions = data['transactions']
    holidays = data['holidays']

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    ax1 = axes[0, 0]
    daily_sales = train.groupby('date')['sales'].sum()
    ax1.plot(daily_sales.index, daily_sales.values, linewidth=0.5, alpha=0.7)
    ax1.set_title('Daily Total Sales Over Time')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Total Sales')
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    top_families = train.groupby('family')['sales'].sum().sort_values(ascending=False).head(10)
    top_families.plot(kind='barh', ax=ax2, color='steelblue')
    ax2.set_title('Top 10 Product Families by Total Sales')
    ax2.set_xlabel('Total Sales')
    ax2.invert_yaxis()

    ax3 = axes[1, 0]
    store_sales = train.groupby('store_nbr')['sales'].sum().reset_index()
    store_sales = store_sales.merge(stores[['store_nbr', 'city']], on='store_nbr')
    city_sales = store_sales.groupby('city')['sales'].sum().sort_values(ascending=False)
    city_sales.plot(kind='bar', ax=ax3, color='coral')
    ax3.set_title('Sales by City')
    ax3.set_xlabel('City')
    ax3.set_ylabel('Total Sales')
    ax3.tick_params(axis='x', rotation=45)

    ax4 = axes[1, 1]
    oil_clean = oil.dropna()
    ax4.plot(oil_clean['date'], oil_clean['dcoilwtico'], linewidth=0.5, alpha=0.7, color='green')
    ax4.set_title('Oil Price Over Time')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Oil Price (USD)')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PATH}/sales_overview.png", dpi=150, bbox_inches='tight')
    print(f"  [OK] Saved: {OUTPUT_PATH}/sales_overview.png")
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    family_monthly = train.groupby([train['date'].dt.to_period('M'), 'family'])['sales'].sum().reset_index()
    family_monthly['date'] = family_monthly['date'].astype(str)
    top5_families = train.groupby('family')['sales'].sum().nlargest(5).index
    for family in top5_families:
        family_data = family_monthly[family_monthly['family'] == family]
        ax1.plot(family_data['date'], family_data['sales'], label=family, linewidth=1)
    ax1.set_title('Top 5 Product Families - Monthly Sales Trend')
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Sales')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.tick_params(axis='x', rotation=45)

    ax2 = axes[1]
    transactions_monthly = transactions.copy()
    transactions_monthly['month'] = transactions_monthly['date'].dt.to_period('M')
    monthly_trans = transactions_monthly.groupby('month')['transactions'].sum()
    ax2.bar(range(len(monthly_trans)), monthly_trans.values, color='purple', alpha=0.7)
    ax2.set_title('Monthly Total Transactions')
    ax2.set_xlabel('Month')
    ax2.set_ylabel('Total Transactions')
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PATH}/trends_analysis.png", dpi=150, bbox_inches='tight')
    print(f"  [OK] Saved: {OUTPUT_PATH}/trends_analysis.png")
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax1 = axes[0, 0]
    stores['type'].value_counts().plot(kind='pie', ax=ax1, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Store Type Distribution')
    ax1.set_ylabel('')

    ax2 = axes[0, 1]
    stores['cluster'].value_counts().sort_index().plot(kind='bar', ax=ax2, color='teal')
    ax2.set_title('Store Cluster Distribution')
    ax2.set_xlabel('Cluster')
    ax2.set_ylabel('Count')

    ax3 = axes[1, 0]
    holidays['type'].value_counts().plot(kind='bar', ax=ax3, color='orange')
    ax3.set_title('Holiday Type Distribution')
    ax3.set_xlabel('Type')
    ax3.set_ylabel('Count')
    ax3.tick_params(axis='x', rotation=45)

    ax4 = axes[1, 1]
    holidays['locale'].value_counts().plot(kind='bar', ax=ax4, color='crimson')
    ax4.set_title('Holiday Locale Distribution')
    ax4.set_xlabel('Locale')
    ax4.set_ylabel('Count')
    ax4.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PATH}/store_holiday_analysis.png", dpi=150, bbox_inches='tight')
    print(f"  [OK] Saved: {OUTPUT_PATH}/store_holiday_analysis.png")
    plt.close()

    print("\n>>> All visualizations generated successfully!")

def main():
    print("\n" + "=" * 60)
    print(">>> Store Sales - Exploratory Data Analysis (EDA)")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    data = load_data()
    explore_basic_info(data)
    explore_missing_values(data)
    explore_train_data(data['train'])
    explore_stores_data(data['stores'])
    explore_oil_data(data['oil'])
    explore_holidays_data(data['holidays'])
    explore_transactions_data(data['transactions'])

    try:
        create_visualizations(data)
    except Exception as e:
        print(f"\n[WARNING] Visualization error: {e}")
        print("  (matplotlib/seaborn may not be installed)")

    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

if __name__ == "__main__":
    main()
