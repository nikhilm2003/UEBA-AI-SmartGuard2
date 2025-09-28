# UEBA-AI-SmartGuard2
This repository provides a complete project skeleton for the Bank of Baroda Hackathon (AI-powered UEBA). It showcases how to build a User &amp; Entity Behavior Analytics system for banking security under Zero Trust, covering raw data, feature engineering, model training, inference, and a demo app.

# AI-Powered UEBA for Bank of Baroda – Hackathon 2025

##  Problem
Traditional authentication fails when credentials are compromised.
Attackers who steal valid credentials can mimic normal users.
This makes detection very difficult for banks.

##  Solution
An AI-powered User & Entity Behavior Analytics (UEBA) system that:
- Learns normal user behavior patterns
- Detects anomalies in login or transactions
- Flags or blocks suspicious sessions in real time

##  Technology Stack
- **Python** for core logic
- **scikit-learn** for anomaly detection (Isolation Forest)
- **Pandas / NumPy** for feature engineering
- **Streamlit** for demo application
- **Joblib** for model persistence

##  Demo
Run the Streamlit app:
```bash
streamlit run app.py
