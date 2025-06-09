# 🟡 vibe-ml-G-predic

> Predict gold price movement using Machine Learning (XGBoost + LSTM) and technical indicators like EMA & RSI.

This project is a lightweight ML pipeline for **daily gold price trend prediction**, ideal for backtesting and learning about ML in financial markets.

---

## 📦 Features

- 📥 Pulls gold price data from Yahoo Finance (`GC=F`)
- 📊 Uses technical indicators:
  - Exponential Moving Average (EMA)
  - Relative Strength Index (RSI)
- 🧠 Models:
  - `XGBoostClassifier` – fast, tabular baseline
  - `LSTM` – sequential deep learning model
- 🧪 Predicts if **gold will go UP (1) or DOWN (0)** tomorrow
- ⚙️ Built in Python, with `pandas`, `ta`, `yfinance`, `xgboost`, and `tensorflow`

---

## 🚀 Quickstart

### 1. Clone this repo

```bash
git clone https://github.com/your-username/vibe-ml-G-predic.git
cd vibe-ml-G-predic

2. Setup environment (Anaconda recommended)

conda create -n gold-ml-trade python=3.10
conda activate gold-ml-trade
pip install -r requirements.txt

3. Run XGBoost model

python main_xgb.py

4. Run LSTM model

python main_lstm.py

📊 Sample Output

📊 Accuracy: 58.97%
📈 Probability of gold price going UP tomorrow: 36.60%

🛠 Requirements

    Python 3.8+

    Tensorflow

    XGBoost

    yfinance

    ta

    pandas

    scikit-learn

    numpy

Install with:

pip install -r requirements.txt

🧪 TODO

    Add more technical indicators (MACD, Bollinger Bands)

    Add early-stopping and callbacks in LSTM

    Improve train-test splitting (e.g., time-aware CV)

    Add backtesting & strategy simulation

    Add real-time price updates from Binance or AlphaVantage

📌 Project Goal (THAI 🇹🇭)

โปรเจกต์นี้สร้างขึ้นเพื่อ ทดลองใช้ Machine Learning ทำนายราคาทองคำล่วงหน้าในวันถัดไป โดยอิงจากข้อมูลย้อนหลังและอินดิเคเตอร์พื้นฐาน เช่น EMA และ RSI เหมาะสำหรับ:

    นักพัฒนา/นักศึกษา ที่สนใจด้าน AI และการเงิน

    คนที่อยากเข้าใจการสร้างโมเดลพยากรณ์ราคาตลาดแบบง่าย

    ใช้เป็น baseline หรือโครงร่างสำหรับต่อยอดระบบเทรดอัตโนมัติ

คำเตือน: โมเดลนี้ยังไม่มีระบบ backtest อย่างจริงจัง การใช้งานเพื่อการเทรดจริงมีความเสี่ยงสูง
```
