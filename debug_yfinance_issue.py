import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

# Try to import optional packages with error handling
try:
    import seaborn as sns
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    plt.style.use('default')

try:
    from ta.trend import EMAIndicator
    from ta.momentum import RSIIndicator
    HAS_TA = True
except ImportError:
    HAS_TA = False
    print("❌ TA library not found. Please install: pip install ta")
    exit(1)

print("🏆 Gold Price Prediction with LSTM (Robust Version)")
print("=" * 50)

# 1. โหลดข้อมูล (Robust method)
print("📥 กำลังโหลดข้อมูลราคาทอง...")
try:
    # Method 1: Try standard download
    try:
        ticker = yf.Ticker("GC=F")
        df = ticker.history(start="2020-01-01", end="2024-12-31")
        df = df[['Close']].copy()
        df.columns = ['Close']
        
        if df.empty or len(df) < 100:
            raise ValueError("Insufficient data")
            
        print(f"✅ โหลดข้อมูลสำเร็จ (Method 1): {len(df)} วัน")
        
    except:
        # Method 2: Fallback with different parameters
        print("⚠️ ลองวิธีอื่น...")
        df = yf.download("GC=F", start="2020-01-01", end="2024-12-31", progress=False)
        df = df[['Close']].copy()
        df.columns = ['Close']
        
        if df.empty:
            raise ValueError("No data downloaded")
            
        print(f"✅ โหลดข้อมูลสำเร็จ (Method 2): {len(df)} วัน")
    
    # แสดงสถิติข้อมูล
    current_price = float(df['Close'].iloc[-1])
    min_price = float(df['Close'].min())
    max_price = float(df['Close'].max())
    
    print(f"💰 ราคาล่าสุด: ${current_price:.2f}")
    print(f"📊 ช่วงราคา: ${min_price:.2f} - ${max_price:.2f}")
    
except Exception as e:
    print(f"❌ ไม่สามารถโหลดข้อมูลได้: {e}")
    print("💡 กรุณาตรวจสอบการเชื่อมต่ออินเทอร์เน็ต")
    exit(1)

# 2. คำนวณ Technical Indicators
print("🔍 กำลังคำนวณ Technical Indicators...")
try:
    df['EMA20'] = EMAIndicator(df['Close'], window=20).ema_indicator()
    df['RSI'] = RSIIndicator(df['Close'], window=14).rsi()
    
    # เพิ่ม indicators เพิ่มเติม
    df['SMA10'] = df['Close'].rolling(window=10).mean()
    df['Price_Change'] = df['Close'].pct_change()
    df['Volatility'] = df['Price_Change'].rolling(window=10).std()
    
    # ลบ NaN values
    df.dropna(inplace=True)
    
    if len(df) < 50:
        raise ValueError("Insufficient data after cleaning")
    
    print(f"✅ คำนวณ Technical Indicators สำเร็จ")
    print(f"📊 ข้อมูลหลังทำความสะอาด: {len(df)} วัน")

except Exception as e:
    print(f"❌ เกิดข้อผิดพลาดในการคำนวณ indicators: {e}")
    exit(1)

# 3. สร้าง Target
print("🎯 กำลังสร้าง Target...")
df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
df.dropna(inplace=True)

up_days = int((df['Target'] == 1).sum())
down_days = int((df['Target'] == 0).sum())
up_pct = up_days / len(df) * 100
down_pct = down_days / len(df) * 100

print(f"📊 Target Distribution:")
print(f"   📈 ขึ้น (1): {up_days} วัน ({up_pct:.1f}%)")
print(f"   📉 ลง (0): {down_days} วัน ({down_pct:.1f}%)")

# 4. สร้าง Sequence Data
print("🔄 กำลังสร้าง Sequence Data...")

def create_sequences(data, target, window=10):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window])
        y.append(target[i+window])
    return np.array(X), np.array(y)

features = ['Close', 'EMA20', 'RSI', 'SMA10', 'Price_Change', 'Volatility']
print(f"📋 ใช้ Features: {features}")

# Scale features
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[features])

# สร้าง sequences
X, y = create_sequences(scaled_data, df['Target'].values, window=10)
print(f"✅ สร้าง Sequences สำเร็จ: {X.shape}")

# 5. แบ่ง Train/Test
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"📊 Data Split:")
print(f"   🔧 Training: {len(X_train)} sequences")
print(f"   🧪 Testing: {len(X_test)} sequences")

# 6. สร้าง LSTM Model
print("🧠 กำลังสร้าง LSTM Model...")

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(32, return_sequences=False),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)

print("📋 Model Summary:")
model.summary()

# 7. เทรนโมเดล
print("\n🚀 กำลังเทรนโมเดล...")

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, verbose=1)
]

try:
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.15,
        callbacks=callbacks,
        verbose=1
    )
    print("✅ การเทรนเสร็จสมบูรณ์")
except Exception as e:
    print(f"❌ เกิดข้อผิดพลาดในการเทรน: {e}")
    exit(1)

# 8. ทดสอบโมเดล
print("\n🧪 กำลังทดสอบโมเดล...")
y_pred = model.predict(X_test, verbose=0).flatten()
y_pred_class = (y_pred > 0.5).astype(int)

# คำนวณ metrics
acc = accuracy_score(y_test, y_pred_class)
print(f"\n📊 ผลการประเมิน:")
print(f"   🎯 Accuracy: {acc:.2%}")

# 9. Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Training Loss
axes[0,0].plot(history.history['loss'], 'b-', label='Training Loss', linewidth=2)
axes[0,0].plot(history.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
axes[0,0].set_title('📈 Model Loss')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# Training Accuracy
axes[0,1].plot(history.history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
axes[0,1].plot(history.history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
axes[0,1].set_title('🎯 Model Accuracy')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_class)
if HAS_SEABORN:
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1,0])
else:
    im = axes[1,0].imshow(cm, cmap='Blues')
    for i in range(2):
        for j in range(2):
            axes[1,0].text(j, i, str(cm[i, j]), ha='center', va='center')

axes[1,0].set_title('🔍 Confusion Matrix')
axes[1,0].set_xlabel('Predicted')
axes[1,0].set_ylabel('Actual')

# Prediction Distribution
axes[1,1].hist(y_pred, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
axes[1,1].axvline(x=0.5, color='red', linestyle='--', linewidth=2)
axes[1,1].set_title('📊 Prediction Distribution')
axes[1,1].set_xlabel('Predicted Probability')

plt.tight_layout()
plt.show()

# 10. การพยากรณ์
print(f"\n🔮 การพยากรณ์:")
latest_seq = X[-1:]
pred_prob = float(model.predict(latest_seq, verbose=0)[0][0])
pred_direction = "📈 ขึ้น" if pred_prob > 0.5 else "📉 ลง"

# ข้อมูลล่าสุด (safe conversion)
current_price = float(df['Close'].iloc[-1])
ema20_current = float(df['EMA20'].iloc[-1])
rsi_current = float(df['RSI'].iloc[-1])
volatility_current = float(df['Volatility'].iloc[-1])

print(f"   💎 ราคาทองล่าสุด: ${current_price:.2f}")
print(f"   🎯 ความน่าจะเป็นราคา 'ขึ้น' พรุ่งนี้: {pred_prob*100:.2f}%")
print(f"   📊 ทิศทางที่คาดการณ์: {pred_direction}")
print(f"   🔥 ความมั่นใจ: {abs(pred_prob - 0.5) * 200:.1f}%")

print(f"\n📋 ข้อมูลเพิ่มเติม:")
print(f"   📊 RSI ล่าสุด: {rsi_current:.1f}")
print(f"   📈 EMA20 ล่าสุด: ${ema20_current:.2f}")
print(f"   ⚖️ ราคา vs EMA20: {'Above' if current_price > ema20_current else 'Below'}")
print(f"   📊 Volatility ล่าสุด: {volatility_current:.4f}")

# 11. สรุปประสิทธิภาพ
correct_predictions = int((y_pred_class == y_test).sum())
total_predictions = len(y_test)

print(f"\n📈 สรุปประสิทธิภาพโมเดล:")
print(f"   🎯 ความแม่นยำ: {acc:.2%}")
print(f"   ✅ ทำนายถูก: {correct_predictions}/{total_predictions}")
print(f"   📊 จำนวน epochs ที่เทรน: {len(history.history['loss'])}")

print(f"\n🎉 การวิเคราะห์เสร็จสิ้น!")
print("=" * 50)