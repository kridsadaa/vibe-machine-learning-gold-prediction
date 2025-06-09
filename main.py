import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, SMAIndicator
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time
import warnings
warnings.filterwarnings('ignore')

print("🏆 Simple XGBoost Gold Prediction (Fixed Version)")
print("=" * 50)

# 1. ดึงข้อมูลราคาทองด้วยการจัดการ Rate Limit
def download_gold_data_safe():
    """Download gold data with error handling"""
    try:
        print("📥 กำลังโหลดข้อมูลราคาทอง...")
        # Add small delay to avoid rate limiting
        time.sleep(1)
        
        gold = yf.download("GC=F", start="2022-01-01", progress=False)
        
        if gold.empty:
            raise ValueError("No data downloaded")
            
        df = gold[['Close']].copy()
        print(f"✅ โหลดข้อมูลสำเร็จ: {len(df)} วัน")
        
        # Show basic info
        current_price = float(df['Close'].iloc[-1])
        print(f"💰 ราคาล่าสุด: ${current_price:.2f}")
        
        return df
        
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {e}")
        print("🔄 ใช้ข้อมูลจำลองแทน...")
        
        # Create sample data as fallback
        dates = pd.date_range(start="2022-01-01", freq='D')
        np.random.seed(42)
        
        # Generate realistic price movement
        returns = np.random.normal(0.0005, 0.02, len(dates))  # Small positive drift
        prices = [2000]  # Starting price
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        df = pd.DataFrame({'Close': prices}, index=dates)
        print(f"✅ สร้างข้อมูลจำลอง: {len(df)} วัน")
        
        return df

# โหลดข้อมูล
df = download_gold_data_safe()

# 🛠 แก้ปัญหา shape ของ df['Close']
close = df['Close'].squeeze()

# 2. สร้าง Technical Indicators
print("🔍 กำลังคำนวณ Technical Indicators...")

try:
    # Basic indicators
    df['EMA20'] = EMAIndicator(close, window=20).ema_indicator()
    df['RSI'] = RSIIndicator(close, window=14).rsi()
    df['SMA10'] = SMAIndicator(close, window=10).sma_indicator()
    df['SMA50'] = SMAIndicator(close, window=50).sma_indicator()
    
    # Price-based features
    df['Price_Change'] = close.pct_change()
    df['Price_MA_Ratio'] = close / df['EMA20']
    df['Volatility'] = close.rolling(10).std()
    
    # Momentum indicators
    df['ROC_5'] = ((close - close.shift(5)) / close.shift(5)) * 100
    
    print("✅ คำนวณ Technical Indicators สำเร็จ")
    
except Exception as e:
    print(f"❌ เกิดข้อผิดพลาดในการคำนวณ indicators: {e}")
    exit(1)

# 3. สร้าง Target: ขึ้น (1) หรือ ลง (0) ในวันถัดไป
df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

# 4. เตรียมข้อมูล
df.dropna(inplace=True)

# เลือก features
features = ['EMA20', 'RSI', 'SMA10', 'SMA50', 'Price_Change', 'Price_MA_Ratio', 'Volatility', 'ROC_5']

print(f"\n📋 ใช้ Features: {len(features)} ตัว")
for i, feature in enumerate(features, 1):
    print(f"   {i}. {feature}")

X = df[features]
y = df['Target']

# แสดง target distribution
up_days = int((y == 1).sum())
down_days = int((y == 0).sum())
print(f"\n📊 Target Distribution:")
print(f"   📈 ขึ้น (1): {up_days} วัน ({up_days/len(y)*100:.1f}%)")
print(f"   📉 ลง (0): {down_days} วัน ({down_days/len(y)*100:.1f}%)")

# 5. แบ่ง train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False, random_state=42
)

print(f"\n📊 Data Split:")
print(f"   🔧 Training: {len(X_train)} samples")
print(f"   🧪 Testing: {len(X_test)} samples")

# 6. เทรนโมเดล XGBoost (เวอร์ชันง่าย ๆ ที่ทำงานได้แน่นอน)
print("\n🚀 กำลังเทรนโมเดล XGBoost...")

# ใช้พารามิเตอร์ที่เข้ากันได้กับทุกเวอร์ชัน
model = XGBClassifier(
    n_estimators=100,  # ลดลงเพื่อความเร็ว
    max_depth=4,
    learning_rate=0.1,
    random_state=42
)

# เทรนแบบง่าย ๆ
try:
    model.fit(X_train, y_train)
    print("✅ การเทรนเสร็จสมบูรณ์")
except Exception as e:
    print(f"❌ เกิดข้อผิดพลาดในการเทรน: {e}")
    exit(1)

# 7. ทดสอบและประเมินผล
print("\n🧪 กำลังทดสอบโมเดล...")

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
print(f"\n📊 ผลการประเมิน:")
print(f"   🎯 Accuracy: {acc:.2%}")

# 8. Feature Importance
print(f"\n🔍 Feature Importance:")
try:
    feature_importance = model.feature_importances_
    
    # Create importance dataframe
    importance_df = pd.DataFrame({
        'feature': features,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    print("   Top 5 สำคัญที่สุด:")
    for i, row in importance_df.head(5).iterrows():
        print(f"   {row['feature']}: {row['importance']:.3f}")
        
except Exception as e:
    print(f"⚠️ ไม่สามารถแสดง feature importance: {e}")

# 9. Visualization
print("\n📊 กำลังสร้างกราฟ...")

try:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Feature Importance
    if 'importance_df' in locals():
        top_features = importance_df.head(6)
        axes[0,0].barh(range(len(top_features)), top_features['importance'], color='skyblue')
        axes[0,0].set_yticks(range(len(top_features)))
        axes[0,0].set_yticklabels(top_features['feature'])
        axes[0,0].set_title('🔍 Feature Importance')
        axes[0,0].set_xlabel('Importance Score')
    
    # Plot 2: Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    im = axes[0,1].imshow(cm, cmap='Blues')
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            axes[0,1].text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=14)
    
    axes[0,1].set_xticks([0, 1])
    axes[0,1].set_yticks([0, 1])
    axes[0,1].set_xticklabels(['Predicted ลง', 'Predicted ขึ้น'])
    axes[0,1].set_yticklabels(['Actual ลง', 'Actual ขึ้น'])
    axes[0,1].set_title('🔍 Confusion Matrix')
    
    # Plot 3: Prediction Distribution
    axes[1,0].hist(y_pred_proba, bins=15, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[1,0].axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Threshold')
    axes[1,0].set_title('📊 Prediction Probability')
    axes[1,0].set_xlabel('Probability')
    axes[1,0].set_ylabel('Count')
    axes[1,0].legend()
    
    # Plot 4: Recent Price Movement
    recent_data = df.tail(50)
    axes[1,1].plot(recent_data.index, recent_data['Close'], 'b-', linewidth=2, label='Gold Price')
    axes[1,1].set_title('📈 Recent Gold Price (50 Days)')
    axes[1,1].set_ylabel('Price ($)')
    axes[1,1].legend()
    axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
except Exception as e:
    print(f"⚠️ ไม่สามารถสร้างกราฟได้: {e}")

# 10. พยากรณ์ล่าสุด
print(f"\n🔮 การพยากรณ์:")

try:
    latest = df.iloc[-1:][features]
    prob = model.predict_proba(latest)[0]
    pred_class = model.predict(latest)[0]
    
    current_price = float(df['Close'].iloc[-1])
    rsi_current = float(df['RSI'].iloc[-1])
    ema20_current = float(df['EMA20'].iloc[-1])
    
    pred_direction = "📈 ขึ้น" if pred_class == 1 else "📉 ลง"
    confidence = max(prob) * 100
    
    print(f"   💎 ราคาทองล่าสุด: ${current_price:.2f}")
    print(f"   🎯 ความน่าจะเป็นราคา 'ขึ้น' พรุ่งนี้: {prob[1]*100:.2f}%")
    print(f"   📊 ทิศทางที่คาดการณ์: {pred_direction}")
    print(f"   🔥 ความมั่นใจ: {confidence:.1f}%")
    
    print(f"\n📋 ข้อมูลเพิ่มเติม:")
    print(f"   📊 RSI ล่าสุด: {rsi_current:.1f}")
    print(f"   📈 EMA20 ล่าสุด: ${ema20_current:.2f}")
    print(f"   ⚖️ ราคา vs EMA20: {'Above' if current_price > ema20_current else 'Below'}")
    
except Exception as e:
    print(f"❌ เกิดข้อผิดพลาดในการพยากรณ์: {e}")

# 11. Performance Summary
correct_predictions = int((y_pred == y_test).sum())
total_predictions = len(y_test)

print(f"\n📈 สรุปประสิทธิภาพโมเดล:")
print(f"   🎯 ความแม่นยำ: {acc:.2%}")
print(f"   ✅ ทำนายถูก: {correct_predictions}/{total_predictions}")

if 'importance_df' in locals():
    print(f"   🌟 Feature ที่สำคัญที่สุด: {importance_df.iloc[0]['feature']}")

print(f"\n🎉 การวิเคราะห์เสร็จสิ้น!")
print("=" * 50)

# 12. Model Information
import xgboost as xgb
print(f"\n💡 ข้อมูลระบบ:")
print(f"   🔍 XGBoost version: {xgb.__version__}")
print(f"   📊 Total features: {len(features)}")
print(f"   📅 Data points: {len(df)}")
print(f"   🎯 Model type: XGBClassifier")

print(f"\n💾 หากต้องการบันทึกโมเดล:")
print("""
import joblib
joblib.dump(model, 'gold_xgb_model.pkl')

# โหลดโมเดล:
model = joblib.load('gold_xgb_model.pkl')
""")

import joblib

joblib.dump(model, 'gold_xgb_model.pkl')
model = joblib.load('gold_xgb_model.pkl')

