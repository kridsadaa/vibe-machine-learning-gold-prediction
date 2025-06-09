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
    print("✅ Seaborn loaded successfully")
except ImportError:
    HAS_SEABORN = False
    plt.style.use('default')
    print("⚠️ Seaborn not found - using matplotlib default style")

try:
    from ta.trend import EMAIndicator
    from ta.momentum import RSIIndicator
    HAS_TA = True
    print("✅ TA library loaded successfully")
except ImportError:
    HAS_TA = False
    print("❌ TA library not found. Please install: pip install ta")
    exit(1)

print("\n🏆 Gold Price Prediction with LSTM")
print("=" * 50)

# 1. โหลดข้อมูล
print("📥 กำลังโหลดข้อมูลราคาทอง...")
try:
    df = yf.download("GC=F", start="2020-01-01", end="2024-12-31")[['Close']]
    print(f"✅ โหลดข้อมูลสำเร็จ: {len(df)} วัน")
    print(f"📊 ช่วงราคา: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")
except Exception as e:
    print(f"❌ เกิดข้อผิดพลาดในการโหลดข้อมูล: {e}")
    exit(1)

# 2. คำนวณ Technical Indicators
print("🔍 กำลังคำนวณ Technical Indicators...")
df['EMA20'] = EMAIndicator(df['Close'], window=20).ema_indicator()
df['RSI'] = RSIIndicator(df['Close'], window=14).rsi()

# เพิ่ม technical indicators เพิ่มเติม
df['SMA10'] = df['Close'].rolling(window=10).mean()
df['Price_Change'] = df['Close'].pct_change()
df['Volatility'] = df['Price_Change'].rolling(window=10).std()

df.dropna(inplace=True)
print(f"✅ คำนวณ Technical Indicators สำเร็จ")

# 3. สร้าง Target: ราคาวันถัดไปมากกว่าวันนี้ = 1 (ขึ้น), ไม่ใช่ = 0
df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
df.dropna(inplace=True)

print(f"📊 Target Distribution:")
up_days = (df['Target'] == 1).sum()
down_days = (df['Target'] == 0).sum()
print(f"   📈 ขึ้น (1): {up_days} วัน ({(df['Target'] == 1).mean():.1%})")
print(f"   📉 ลง (0): {down_days} วัน ({(df['Target'] == 0).mean():.1%})")

# 4. สร้าง Sequence Data (window = 10 วัน)
def create_sequences(data, target, window=10):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window])
        y.append(target[i+window])
    return np.array(X), np.array(y)

# ใช้ features มากขึ้น
features = ['Close', 'EMA20', 'RSI', 'SMA10', 'Price_Change', 'Volatility']
print(f"🔄 ใช้ Features: {features}")

scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[features])

print("🔄 กำลังสร้าง Sequence Data...")
X, y = create_sequences(scaled, df['Target'].values, window=10)
print(f"✅ สร้าง Sequences สำเร็จ: {X.shape[0]} sequences, {X.shape[1]} timesteps, {X.shape[2]} features")

# 5. แบ่ง Train/Test
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"📊 Data Split:")
print(f"   🔧 Training: {len(X_train)} sequences")
print(f"   🧪 Testing: {len(X_test)} sequences")

# 6. สร้าง LSTM model
print("🧠 กำลังสร้าง LSTM Model...")
model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(32, return_sequences=False),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# เพิ่ม learning rate scheduler
initial_learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=100,
    decay_rate=0.96,
    staircase=True)

model.compile(
    loss='binary_crossentropy', 
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    metrics=['accuracy']
)

print("📋 Model Architecture:")
model.summary()

# 7. เทรนโมเดล
print("\n🚀 กำลังเทรนโมเดล...")

# สร้าง callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        patience=15, 
        restore_best_weights=True, 
        verbose=1,
        monitor='val_loss'
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        min_lr=1e-7,
        verbose=1
    )
]

# เทรนโมเดล
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

# 8. ทดสอบและประเมินผล
print("\n🧪 กำลังทดสอบโมเดล...")
y_pred = model.predict(X_test, verbose=0).flatten()
y_pred_class = (y_pred > 0.5).astype(int)

# คำนวณ metrics
acc = accuracy_score(y_test, y_pred_class)
print(f"\n📊 ผลการประเมิน:")
print(f"   🎯 Accuracy: {acc:.2%}")

# Additional metrics
from sklearn.metrics import precision_score, recall_score, f1_score

precision = precision_score(y_test, y_pred_class)
recall = recall_score(y_test, y_pred_class)
f1 = f1_score(y_test, y_pred_class)

print(f"   📈 Precision: {precision:.2%}")
print(f"   📉 Recall: {recall:.2%}")
print(f"   🎯 F1-Score: {f1:.2%}")

# Classification Report
print(f"\n📈 Classification Report:")
print(classification_report(y_test, y_pred_class, target_names=['ลง', 'ขึ้น']))

# 9. Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Training History
axes[0,0].plot(history.history['loss'], label='Training Loss', linewidth=2)
axes[0,0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
axes[0,0].set_title('📈 Model Loss During Training')
axes[0,0].set_xlabel('Epoch')
axes[0,0].set_ylabel('Loss')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# Plot 2: Accuracy History
axes[0,1].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
axes[0,1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
axes[0,1].set_title('🎯 Model Accuracy During Training')
axes[0,1].set_xlabel('Epoch')
axes[0,1].set_ylabel('Accuracy')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# Plot 3: Confusion Matrix
cm = confusion_matrix(y_test, y_pred_class)
if HAS_SEABORN:
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted ลง', 'Predicted ขึ้น'],
                yticklabels=['Actual ลง', 'Actual ขึ้น'],
                ax=axes[1,0])
else:
    im = axes[1,0].imshow(cm, interpolation='nearest', cmap='Blues')
    axes[1,0].figure.colorbar(im, ax=axes[1,0])
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[1,0].text(j, i, format(cm[i, j], 'd'),
                         ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black")
    axes[1,0].set_xticks([0, 1])
    axes[1,0].set_yticks([0, 1])
    axes[1,0].set_xticklabels(['Predicted ลง', 'Predicted ขึ้น'])
    axes[1,0].set_yticklabels(['Actual ลง', 'Actual ขึ้น'])
axes[1,0].set_title('🔍 Confusion Matrix')

# Plot 4: Prediction Probability Distribution
axes[1,1].hist(y_pred, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
axes[1,1].set_title('📊 Prediction Probability Distribution')
axes[1,1].set_xlabel('Predicted Probability')
axes[1,1].set_ylabel('Frequency')
axes[1,1].axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Threshold')
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 10. พยากรณ์วันล่าสุด
print(f"\n🔮 การพยากรณ์:")
latest_seq = X[-1:]
pred_prob = model.predict(latest_seq, verbose=0)[0][0]
pred_direction = "📈 ขึ้น" if pred_prob > 0.5 else "📉 ลง"

print(f"   💎 ราคาทองล่าสุด: ${df['Close'].iloc[-1]:.2f}")
print(f"   🎯 ความน่าจะเป็นราคา 'ขึ้น' พรุ่งนี้: {pred_prob*100:.2f}%")
print(f"   📊 ทิศทางที่คาดการณ์: {pred_direction}")
print(f"   🔥 ความมั่นใจ: {abs(pred_prob - 0.5) * 200:.1f}%")

# 11. Feature Importance Analysis (ข้อมูลเพิ่มเติม)
print(f"\n📋 ข้อมูลเพิ่มเติม:")
print(f"   📊 RSI ล่าสุด: {df['RSI'].iloc[-1]:.1f}")
print(f"   📈 EMA20 ล่าสุด: ${df['EMA20'].iloc[-1]:.2f}")
print(f"   📈 SMA10 ล่าสุด: ${df['SMA10'].iloc[-1]:.2f}")
print(f"   ⚖️ ราคา vs EMA20: {'Above' if df['Close'].iloc[-1] > df['EMA20'].iloc[-1] else 'Below'}")
print(f"   📊 Volatility ล่าสุด: {df['Volatility'].iloc[-1]:.4f}")
print(f"   📈 Price Change ล่าสุด: {df['Price_Change'].iloc[-1]:.4f}")

# 12. Model Performance Summary
print(f"\n📈 สรุปประสิทธิภาพโมเดล:")
print(f"   🎯 Test Accuracy: {acc:.2%}")
print(f"   📊 ทำนายถูก: {(y_pred_class == y_test).sum()} / {len(y_test)}")
print(f"   📈 True Positives: {((y_pred_class == 1) & (y_test == 1)).sum()}")
print(f"   📉 True Negatives: {((y_pred_class == 0) & (y_test == 0)).sum()}")

# 13. บันทึกโมเดล (optional)
try:
    model_save = input("\n💾 ต้องการบันทึกโมเดลหรือไม่? (y/n): ")
    if model_save.lower() in ['y', 'yes']:
        model.save('gold_lstm_model.h5')
        # บันทึก scaler ด้วย
        import joblib
        joblib.dump(scaler, 'gold_scaler.pkl')
        print("✅ บันทึกโมเดลและ scaler เรียบร้อยแล้ว:")
        print("   📁 gold_lstm_model.h5")
        print("   📁 gold_scaler.pkl")
except:
    print("⚠️ ข้ามการบันทึกโมเดล")

print(f"\n🎉 การวิเคราะห์เสร็จสิ้น!")
print("=" * 50)

# 14. การใช้งานโมเดลที่บันทึกไว้ (คำแนะนำ)
print(f"\n💡 วิธีการโหลดโมเดลที่บันทึกไว้:")
print("""
# โหลดโมเดลและ scaler
import tensorflow as tf
import joblib

model = tf.keras.models.load_model('gold_lstm_model.h5')
scaler = joblib.load('gold_scaler.pkl')

# ใช้งานทำนาย
# predictions = model.predict(new_data)
""")