import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression

# -------------------------
# CONFIG
# -------------------------
st.set_page_config(page_title="GeoSales AI", layout="wide")

# -------------------------
# CSS
# -------------------------
st.markdown("""
<style>
body {background: linear-gradient(to right, #0e1117, #1c1f26);}
.card {
    background: rgba(255,255,255,0.05);
    padding: 20px;
    border-radius: 15px;
    text-align: center;
}
.big {font-size: 30px; font-weight: bold; color: #00ffcc;}
.title {text-align:center; font-size:40px; font-weight:bold;}
</style>
""", unsafe_allow_html=True)

# -------------------------
# TITLE
# -------------------------
st.markdown("<div class='title'>🚀 GeoSales AI Dashboard </div>", unsafe_allow_html=True)

# -------------------------
# LOAD DATA
# -------------------------
df = pd.read_csv("sales_data.csv", encoding='latin1')

df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
df['order_date'] = pd.to_datetime(df['order_date'], dayfirst=True)
df['month'] = df['order_date'].dt.month
df['year'] = df['order_date'].dt.year

# -------------------------
# FILTERS
# -------------------------
st.sidebar.title("⚙️ Filters")

region = st.sidebar.selectbox("Region", ["All"] + list(df['region'].unique()))
category = st.sidebar.selectbox("Category", ["All"] + list(df['category'].unique()))

filtered_df = df.copy()

if region != "All":
    filtered_df = filtered_df[filtered_df['region'] == region]

if category != "All":
    filtered_df = filtered_df[filtered_df['category'] == category]

# -------------------------
# KPI CARDS
# -------------------------
col1, col2, col3 = st.columns(3)

col1.markdown(f"<div class='card'><div>Total Sales</div><div class='big'>₹ {int(filtered_df['sales'].sum())}</div></div>", unsafe_allow_html=True)
col2.markdown(f"<div class='card'><div>Total Profit</div><div class='big'>₹ {int(filtered_df['profit'].sum())}</div></div>", unsafe_allow_html=True)
col3.markdown(f"<div class='card'><div>Orders</div><div class='big'>{filtered_df.shape[0]}</div></div>", unsafe_allow_html=True)

st.markdown("---")

# -------------------------
# GRAPH SECTION
# -------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Sales vs Profit Trend")
    trend = filtered_df.groupby('month')[['sales','profit']].sum().reset_index()
    fig = px.line(trend, x='month', y=['sales','profit'], markers=True)
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("📦 Category Sales")
    cat = filtered_df.groupby('category')['sales'].sum().reset_index()
    fig = px.bar(cat, x='category', y='sales', color='category')
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

col1, col2 = st.columns(2)

with col1:
    st.subheader("🌍 Region Distribution")
    reg = filtered_df.groupby('region')['sales'].sum().reset_index()
    fig = px.pie(reg, names='region', values='sales')
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("📈 Monthly Trend")
    monthly = filtered_df.groupby('month')['sales'].sum().reset_index()
    fig = px.line(monthly, x='month', y='sales', markers=True)
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

col1, col2 = st.columns(2)

with col1:
    st.subheader("📉 Profit vs Discount")
    fig = px.scatter(filtered_df, x='discount', y='profit', color='category')
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("📆 Yearly Growth")
    yearly = filtered_df.groupby('year')['sales'].sum().reset_index()
    fig = px.line(yearly, x='year', y='sales', markers=True)
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

# -------------------------
# INSIGHTS
# -------------------------
st.subheader("💡 Smart Insights")

if not filtered_df.empty:
    best_region = filtered_df.groupby('region')['sales'].sum().idxmax()
    st.success(f"🔥 {best_region} is the top-performing region.")

# -------------------------
# CHATBOT SECTION
# -------------------------
st.markdown("---")
st.subheader("🤖 GeoSales AI Chatbot")

user_input = st.text_input("Ask your question about sales...")

if user_input:
    query = user_input.lower()

    if "total sales" in query:
        st.write(f"💰 Total Sales: ₹ {int(filtered_df['sales'].sum())}")

    elif "profit" in query:
        st.write(f"📈 Total Profit: ₹ {int(filtered_df['profit'].sum())}")

    elif "best region" in query:
        best_region = filtered_df.groupby('region')['sales'].sum().idxmax()
        st.write(f"🏆 Best Region: {best_region}")

    elif "orders" in query:
        st.write(f"📦 Total Orders: {filtered_df.shape[0]}")

    else:
        st.write("🤖 Sorry, I can answer questions about sales, profit, region, and orders.")