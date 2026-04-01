
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="GeoSales AI", layout="wide")

# ---------------- PREMIUM UI ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}

div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.05);
    border-radius: 15px;
    padding: 15px;
    backdrop-filter: blur(10px);
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}

section[data-testid="stSidebar"] {
    background: rgba(0,0,0,0.3);
}

button[kind="primary"] {
    background: linear-gradient(90deg,#00c6ff,#0072ff);
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown("<h1 style='text-align:center;'>🚀 GeoSales AI Dashboard</h1>", unsafe_allow_html=True)
st.caption("AI-powered Sales Analytics & Forecasting")

# ---------------- DATA ----------------
df = pd.read_csv("sales_data.csv", encoding='latin1')
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

df['order_date'] = pd.to_datetime(df['order_date'], dayfirst=True)
df['month'] = df['order_date'].dt.month
df['year'] = df['order_date'].dt.year

# ---------------- SIDEBAR ----------------
st.sidebar.title("🔍 Filters")
region = st.sidebar.selectbox("Region", df['region'].unique())
category = st.sidebar.selectbox("Category", df['category'].unique())

filtered_df = df[(df['region']==region) & (df['category']==category)]

st.caption(f"📍 Showing: {region} | {category}")

# ---------------- KPI ----------------
prev_sales = df['sales'].sum() * 0.9
prev_profit = df['profit'].sum() * 0.9

total_sales = filtered_df['sales'].sum()
total_profit = filtered_df['profit'].sum()
orders = filtered_df.shape[0]

sales_change = ((total_sales-prev_sales)/prev_sales)*100
profit_change = ((total_profit-prev_profit)/prev_profit)*100

c1,c2,c3 = st.columns(3)

c1.metric("💰 Sales", int(total_sales), f"{sales_change:.2f}%")
c2.metric("📈 Profit", int(total_profit), f"{profit_change:.2f}%")
c3.metric("📦 Orders", orders)

st.markdown("---")

# ---------------- REGION COMPARISON ----------------
st.subheader(f"📊 Sales vs Profit by Region ({category})")

chart_df = df[df['category']==category]

sales_r = chart_df.groupby('region')['sales'].sum().reset_index()
profit_r = chart_df.groupby('region')['profit'].sum().reset_index()

merged = sales_r.merge(profit_r,on='region')

fig = px.line(merged,x='region',y=['sales','profit'],markers=True)
fig.update_layout(template="plotly_dark",
                  paper_bgcolor="rgba(0,0,0,0)",
                  plot_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig,use_container_width=True)

# ---------------- MONTHLY ----------------
st.subheader("📈 Monthly Trend")

monthly = filtered_df.groupby('month')['sales'].sum().reset_index()

fig = px.line(monthly,x='month',y='sales',markers=True)
fig.update_layout(template="plotly_dark",
                  paper_bgcolor="rgba(0,0,0,0)",
                  plot_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig,use_container_width=True)

# ---------------- FORECAST (FIXED) ----------------
st.subheader("🔮 Future Forecast")

monthly_full = filtered_df.groupby('month')['sales'].sum().reset_index()

if len(monthly_full) > 3:
    model_f = LinearRegression()
    model_f.fit(monthly_full[['month']], monthly_full['sales'])

    future = pd.DataFrame({'month':[13,14,15]})
    pred = model_f.predict(future)

    forecast_df = pd.DataFrame({
        "month":[13,14,15],
        "forecast":pred
    })

    monthly_full['type'] = "Actual"
    forecast_df['type'] = "Forecast"

    combined = pd.concat([
        monthly_full.rename(columns={'sales':'value'}),
        forecast_df.rename(columns={'forecast':'value'})
    ])

    fig = px.line(combined, x='month', y='value', color='type', markers=True)
    fig.update_layout(template="plotly_dark",
                      paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig,use_container_width=True)

else:
    st.warning("Not enough data for forecasting")

# ---------------- INSIGHTS ----------------
st.markdown("## 💡 Business Insights")

best_region = df.groupby('region')['profit'].sum().idxmax()
worst_region = df.groupby('region')['profit'].sum().idxmin()
growth = ((total_sales-prev_sales)/prev_sales)*100

st.markdown(f"""
<div style='background: rgba(255,255,255,0.05);
padding:20px;border-radius:15px;'>

🔥 Best Region: {best_region}<br><br>
⚠️ Worst Region: {worst_region}<br><br>
📈 Growth: {growth:.2f}%  

</div>
""",unsafe_allow_html=True)

# ---------------- MODEL ----------------
model = joblib.load("model/sales_model.pkl")
cols = joblib.load("model/model_columns.pkl")

# ---------------- FEATURE ----------------
st.subheader("🧠 Feature Importance")

imp = pd.Series(model.feature_importances_,index=cols).sort_values()

fig = px.bar(imp,orientation='h')
fig.update_layout(template="plotly_dark",
                  paper_bgcolor="rgba(0,0,0,0)",
                  plot_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig,use_container_width=True)

st.info("💡 Profit has highest impact on sales prediction")

# ---------------- AI ----------------
st.markdown("## 🤖 AI Sales Prediction Engine")

c1,c2,c3 = st.columns(3)

month = c1.slider("Month",1,12,5)
profit = c2.number_input("Profit",100.0)
discount = c3.number_input("Discount",0.1)

input_df = pd.DataFrame({
    'month':[month],
    'year':[2023],
    'profit':[profit],
    'discount':[discount]
})

input_df = input_df.reindex(columns=cols,fill_value=0)

if st.button("Predict Sales"):
    result = model.predict(input_df)
    st.success(f"💰 Predicted Sales: {result[0]:.2f}")