import streamlit as st
from src.inference import predict

st.title("AI-Powered UEBA – Bank of Baroda Hackathon")

st.sidebar.header("Enter Session Details")

login_time = st.sidebar.text_input("Login Time", "2025-09-29 03:00:00")
location = st.sidebar.text_input("Location", "Delhi")
device = st.sidebar.selectbox("Device", ["Windows", "Android", "iPhone"])
amount = st.sidebar.number_input("Transaction Amount", min_value=100, value=5000)

if st.sidebar.button("Analyze"):
    session = {
        "login_time": login_time,
        "location": location,
        "device": device,
        "transaction_amount": amount
    }
    result = predict(session)
    st.write("### Result")
    st.json(result)
