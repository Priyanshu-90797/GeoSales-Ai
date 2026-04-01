

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv('C:\\GeoSales Ai\\sales_data.csv')
df.head()

 #Data Understanding

# Shape of dataset
print("Shape:", df.shape)

# Column names
print("Columns:", df.columns)

# Data types and info
df.info()

# Statistical summary
df.describe()

#Data Cleaning

# Check missing values
print("Missing values:\n", df.isnull().sum())

# Remove missing values
df.dropna(inplace=True)

# Check duplicates
print("Duplicates:", df.duplicated().sum())

# Remove duplicates
df.drop_duplicates(inplace=True)
# Convert Date column to datetime

df['Order Date'] = pd.to_datetime(df['Order Date'])
# Extract date features

df['year'] = df['Order Date'].dt.year
df['month'] = df['Order Date'].dt.month
df['day'] = df['Order Date'].dt.day

# Profit Margin
df['profit_margin'] = df['Profit'] / df['Sales']

#EDA:-Exploratory Data Analysis

sales_by_region=df.groupby('Region')['Sales'].sum()
print(sales_by_region)

sales_by_region.plot(kind='bar',title='Total Sales by Region')
plt.show()
df.groupby('Region')['Profit'].sum().plot(kind='bar', title="Profit by Region")
plt.show()
df.groupby('Order Date')['Sales'].sum().plot(title="Sales Trend")
plt.show()
df.groupby('Product Name')['Sales'].sum().sort_values(ascending=False).head(20).plot(kind='barh', title="Top 20 Products by Sales")
plt.show()
plt.scatter(df['Sales'], df['Profit'])
plt.xlabel("Sales")
plt.ylabel("Profit")
plt.show()
numeric_df = df.select_dtypes(include=['number'])

sns.heatmap(numeric_df.corr(), annot=True)
plt.show()

#Advanced Analysis

df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

print(df.columns)
monthly_sales = df.groupby('month')['sales'].sum()

monthly_sales.plot(kind='bar',title="Monthly Sales")
plt.show()
monthly_profit = df.groupby('month')['profit'].sum()

monthly_profit.plot(kind='bar', title="Monthly Profit")
plt.show()
pivot = pd.pivot_table(df, values='sales', index='region', columns='category', aggfunc='sum')

print(pivot)
profit_margin_region = df.groupby('region')['profit_margin'].mean()

print(profit_margin_region)
top_products = df.groupby('product_name')['sales'].sum().sort_values(ascending=False).head(10)

worst_products = df.groupby('product_name')['sales'].sum().sort_values().head(10)

print("Top Products:\n", top_products)
print("Worst Products:\n", worst_products)
plt.scatter(df['discount'], df['profit'])
plt.xlabel("Discount")
plt.ylabel("Profit")
plt.title("Discount vs Profit")
plt.show()

#which region is best 
region_sales = df.groupby('region')['sales'].sum().sort_values(ascending=False)

print(region_sales)
region_profit = df.groupby('region')['profit'].sum().sort_values(ascending=False)

print(region_profit)

#Product loss 
product_profit = df.groupby('product_name')['profit'].sum().sort_values()

# print(product_profit.head(10))

loss_products = product_profit[product_profit < 0]

print(loss_products)

#Discount Effect
plt.scatter(df['discount'], df['profit'])
plt.xlabel("Discount")
plt.ylabel("Profit")
plt.title("Discount vs Profit")
plt.show()
df.groupby('discount')['profit'].mean()
df['discount_bin'] = pd.cut(df['discount'], bins=5)

df.groupby('discount_bin')['profit'].mean()

#Machine learning 

print(df.head())
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
# Convert date
df['order_date'] = pd.to_datetime(df['order_date'])

# Extract features
df['year'] = df['order_date'].dt.year
df['month'] = df['order_date'].dt.month

# Profit margin
df['profit_margin'] = df['profit'] / df['sales']
df = df.drop(['order_id', 'customer_name', 'product_name'], axis=1)
df_encoded=pd.get_dummies(df,drop_first=True)

X = df_encoded.drop(['sales', 'order_date'], axis=1)
X = X.select_dtypes(include=['number'])

y = df_encoded['sales']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(X_train, y_train)

pred_lr = lr.predict(X_test)
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100)

rf.fit(X_train, y_train)

pred_rf = rf.predict(X_test)
from sklearn.metrics import mean_absolute_error, r2_score

print("Linear Regression:")
print("MAE:", mean_absolute_error(y_test, pred_lr))
print("R2:", r2_score(y_test, pred_lr))

print("\nRandom Forest:")
print("MAE:", mean_absolute_error(y_test, pred_rf))
print("R2:", r2_score(y_test, pred_rf))
best_model= rf
import pandas as pd

importance=pd.Series(rf.feature_importances_, index=X.columns)
importance.sort_values(ascending=False).head(10).plot(kind='barh', title="Top 10 Feature Importances")
plt.show()
sample = X_test[0:1]

prediction = best_model.predict(sample)

print("Predicted Sales:", prediction)
import pandas as pd
import joblib

df = pd.read_csv("data/sales_data.csv", encoding='latin1')

df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

df['order_date'] = pd.to_datetime(df['order_date'], dayfirst=True)
df['month'] = df['order_date'].dt.month
df['year'] = df['order_date'].dt.year

# ❗ IMPORTANT: only selected columns
df = df[['month', 'year', 'profit', 'discount', 'sales']]

# Features
X = df.drop('sales', axis=1)
y = df['sales']

# Save columns
joblib.dump(X.columns, "model/model_columns.pkl")

# Train model
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()
model.fit(X, y)

# ===== FEATURE IMPORTANCE =====
importance = pd.Series(model.feature_importances_, index=X.columns)
importance = importance.sort_values(ascending=False)

print("Feature Importance:")
print(importance)

import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
importance.plot(kind='bar')

plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")

plt.show()

from xgboost import XGBRegressor

xgb = XGBRegressor(n_estimators=100, max_depth=5)
xgb.fit(X, y)
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor()
xgb=XGBRegressor()
rf.fit(X, y)
xgb.fit(X, y)
print("RF:", r2_score(y, rf.predict(X)))
print("XGB:", r2_score(y, xgb.predict(X)))
# 4.Forecasting 
monthly = df.groupby('month')['sales'].sum().reset_index()

from sklearn.linear_model import LinearRegression

forecast_model = LinearRegression()
forecast_model.fit(monthly[['month']], monthly['sales'])

future = pd.DataFrame({'month': [13,14,15]})
future_pred = forecast_model.predict(future)

print("Future Sales:", future_pred)


import joblib

best_model = rf

joblib.dump(model, "model/sales_model.pkl")
joblib.dump(X.columns, "model/model_columns.pkl")
joblib.dump(xgb, "model/xgb_sales_model.pkl")
joblib.dump(forecast_model, "model/forecast_model.pkl")