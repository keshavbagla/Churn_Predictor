import streamlit as st
import pickle
import pandas as pd
import numpy as np

model = pickle.load(open('churn_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")

st.title("📉 Customer Churn Prediction App")
st.write("Enter customer details to predict churn and reasons")

tenure = st.number_input("Tenure (months)", 0, 100, 10)
city_tier = st.selectbox("City Tier", [1, 2, 3])
warehouse_to_home = st.number_input("Warehouse to Home Distance (km)", 1, 100, 15)
hour_spend_on_app = st.slider("Hours Spent on App", 0.0, 10.0, 2.0)
satisfaction_score = st.selectbox("Satisfaction Score", [1, 2, 3, 4, 5])
complain_score = st.slider("Number of Complaints", 0, 5, 1)
order_amount_hike = st.slider("Order Amount Change (%)", -50, 50, 5)
coupon_used = st.number_input("Coupons Used", 0, 50, 5)
order_count = st.number_input("Order Count", 0, 100, 10)
days_since_last_order = st.number_input("Days Since Last Order", 0, 365, 20)
cashback_amount = st.number_input("Cashback Amount (₹)", 0, 1000, 100)

if st.button("Predict Churn"):

    avg_cashbk_per_order = cashback_amount / order_count if order_count > 0 else 0

    data = pd.DataFrame([{
        'Tenure': tenure,
        'CityTier': city_tier,
        'WarehouseToHome': warehouse_to_home,
        'HourSpendOnApp': hour_spend_on_app,
        'SatisfactionScore': satisfaction_score,
        'Complain': complain_score,
        'OrderAmountHikeFromlastYear': order_amount_hike,
        'CouponUsed': coupon_used,
        'OrderCount': order_count,
        'DaySinceLastOrder': days_since_last_order,
        'CashbackAmount': cashback_amount,
        'avg_cashbk_per_order': avg_cashbk_per_order
    }])

    data['CityTier'] = data['CityTier'].astype('category')
    data = pd.get_dummies(data, drop_first=True)


    num_cols = [
        'Tenure', 'WarehouseToHome', 'HourSpendOnApp',
        'SatisfactionScore', 'Complain',
        'OrderAmountHikeFromlastYear', 'CouponUsed',
        'OrderCount', 'DaySinceLastOrder',
        'CashbackAmount', 'avg_cashbk_per_order'
    ]
    data[num_cols] = scaler.transform(data[num_cols])

    for col in model.feature_names_in_:
        if col not in data.columns:
            data[col] = 0

    data = data[model.feature_names_in_]

    pred = model.predict(data)[0]
    prob = model.predict_proba(data)[0][1] * 100

    if pred == 1:
        st.error(f"Customer WILL churn ({prob:.1f}% probability)")
    else:
        st.success(f"Customer will NOT churn ({100 - prob:.1f}% confidence)")
        
    st.write("Pred:", pred)
    st.write("Prob:", prob)