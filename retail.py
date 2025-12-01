import streamlit as st
import pandas as pd

st.title("Retail Demand Forecasting & Inventory Optimization App")

df = pd.read_csv("FMCG.csv")

# Make SKU string
df["sku"] = df["sku"].astype(str).str.strip()

# Get unique SKUs
sku_list = sorted(df["sku"].unique())

selected_sku = st.selectbox("Select Product (SKU)", sku_list)

st.write("You selected:", selected_sku)

row = df[df["sku"] == selected_sku].sort_values("date").iloc[-1]

st.subheader("Product Details (Auto-filled)")
st.write("Brand:", row["brand"])
st.write("Category:", row["category"])
st.write("Segment:", row["segment"])
st.write("Pack Type:", row["pack_type"])
st.write("Price:", row["price_unit"])

import datetime

date_input = st.date_input("Select Date", datetime.date.today())
stock_input = st.number_input("Current Stock", min_value=0, value=100)

# ----------------------------------------
# Create input DF (correct feature order)
# ----------------------------------------
df_input = pd.DataFrame([{
    "sku": row["sku"],
    "brand": row["brand"],
    "segment": row["segment"],
    "category": row["category"],
    "channel": row["channel"],
    "region": row["region"],
    "pack_type": row["pack_type"],
    "price_unit": row["price_unit"],
    "promotion_flag": row["promotion_flag"],
    "delivery_days": row["delivery_days"],
    "stock_available": stock_input,
    "year": date_input.year,
    "month": date_input.month,
    "week": date_input.isocalendar()[1],
    "day": date_input.day,
    "weekday": date_input.weekday(),
    "is_weekend": 1 if date_input.weekday() >= 5 else 0
}])

# ----------------------------------------
# Apply label encoders EXACTLY as training
# ----------------------------------------
import joblib
cat_cols = ["sku", "brand", "segment", "category", "channel", "region", "pack_type"]

encoders = joblib.load("encoders.pkl")
for col in cat_cols:
    df_input[col] = encoders[col].transform(df_input[col])


model = joblib.load("model.pkl")

prediction = model.predict(df_input)[0]

st.success(f"Predicted Demand: {prediction:.2f}")

# -----------------------------
# INVENTORY CALCULATIONS
# -----------------------------

# Safety Stock (global, constant for all SKUs)
demand_std = df["units_sold"].std()
safety_stock = demand_std * 1.65   # 95% service level

# Notebook formulas
predicted_units = prediction
lead_time = row["delivery_days"]

reorder_point = (predicted_units * lead_time) + safety_stock
recommended_inventory = predicted_units + safety_stock

st.subheader("📦 Inventory Recommendations")

st.write(f"**Safety Stock:** {safety_stock:.2f} units")
st.write(f"**Reorder Point (ROP):** {reorder_point:.2f} units")
st.write(f"**Recommended Inventory Level:** {recommended_inventory:.2f} units")

# Should we reorder?
if stock_input < reorder_point:
    st.error("⚠️ Stock is below Reorder Point — Reorder Required!")
else:
    st.success("Stock is sufficient — No immediate reorder needed.")


hist = df[df["sku"] == selected_sku].sort_values("date")

st.line_chart(hist[["units_sold"]])
