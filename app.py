import os

os.system("pip install joblib scikit-learn")

import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="GeoSales AI", layout="wide")
st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
.card {
    background-color: #161b22;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
}
.big-font {
    font-size: 25px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;'>🚀 GeoSales AI</h1>", unsafe_allow_html=True)


df = pd.read_csv("data/sales_data.csv", encoding='latin1')

df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

df['order_date'] = pd.to_datetime(df['order_date'], dayfirst=True)
df['month'] = df['order_date'].dt.month
df['year'] = df['order_date'].dt.year

st.sidebar.title("Filters")

region = st.sidebar.selectbox("Region", df['region'].unique())
category = st.sidebar.selectbox("Category", df['category'].unique())

filtered_df = df[(df['region']==region) & (df['category']==category)]


col1, col2, col3 = st.columns(3)

col1.markdown(f"<div class='card'><div class='big-font'>💰 {int(filtered_df['sales'].sum())}</div>Total Sales</div>", unsafe_allow_html=True)
col2.markdown(f"<div class='card'><div class='big-font'>📈 {int(filtered_df['profit'].sum())}</div>Total Profit</div>", unsafe_allow_html=True)
col3.markdown(f"<div class='card'><div class='big-font'>📦 {filtered_df.shape[0]}</div>Orders</div>", unsafe_allow_html=True)

st.markdown("---")


st.subheader("📊 Sales vs Profit by Region")

sales_region = df.groupby('region')['sales'].sum().reset_index()
profit_region = df.groupby('region')['profit'].sum().reset_index()

merged = sales_region.merge(profit_region, on='region')

fig = px.line(merged, x='region', y=['sales','profit'], markers=True)
st.plotly_chart(fig, use_container_width=True)


st.subheader("📈 Monthly Trend")

monthly = filtered_df.groupby('month')['sales'].sum().reset_index()

fig = px.line(monthly, x='month', y='sales', markers=True)
st.plotly_chart(fig, use_container_width=True)


st.subheader("🔮 Future Forecast")

monthly_full = df.groupby('month')['sales'].sum().reset_index()

model_f = LinearRegression()
model_f.fit(monthly_full[['month']], monthly_full['sales'])

future = pd.DataFrame({'month':[13,14,15]})
pred = model_f.predict(future)

forecast_df = pd.DataFrame({
    "month":[13,14,15],
    "forecast":pred
})

fig = px.line(forecast_df, x='month', y='forecast', markers=True)
st.plotly_chart(fig, use_container_width=True)


st.subheader("💡 Insights")

best_region = df.groupby('region')['profit'].sum().idxmax()
worst_region = df.groupby('region')['profit'].sum().idxmin()

st.success(f"🔥 Best Region: {best_region}")
st.error(f"⚠️ Worst Region: {worst_region}")


model = joblib.load("model/sales_model.pkl")
cols = joblib.load("model/model_columns.pkl")


st.subheader("🧠 Feature Importance")

importance = pd.Series(model.feature_importances_, index=cols).sort_values()

fig = px.bar(importance, orientation='h')
st.plotly_chart(fig, use_container_width=True)


st.subheader("🤖 AI Prediction")

col1, col2, col3 = st.columns(3)

month = col1.slider("Month",1,12,5)
profit = col2.number_input("Profit",100.0)
discount = col3.number_input("Discount",0.1)

input_df = pd.DataFrame({
    'month':[month],
    'year':[2023],
    'profit':[profit],
    'discount':[discount]
})

input_df = input_df.reindex(columns=cols, fill_value=0)

if st.button("Predict"):
    result = model.predict(input_df)   
    st.success(f"💰 Sales Prediction: {result[0]:.2f}")