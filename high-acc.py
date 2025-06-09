import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
import json
import os
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, SMAIndicator
from ta.volatility import BollingerBands
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("🔴 Real-time Gold Prediction with Binance API")
print("=" * 60)

class BinanceGoldPredictor:
    def __init__(self):
        self.symbol = "PAXGUSDT"  # Paxos Gold - backed by real gold
        self.data_file = "binance_gold_history.csv"
        self.model_file = "gold_model.pkl"
        self.scaler_file = "gold_scaler.pkl"
        self.base_url = "https://api.binance.com/api/v3"
        self.model = None
        self.scaler = None
        self.feature_columns = None
        
    def get_current_price(self):
        """ดึงราคาปัจจุบัน"""
        try:
            url = f"{self.base_url}/ticker/price"
            params = {"symbol": self.symbol}
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            return float(data['price'])
        except Exception as e:
            print(f"❌ ไม่สามารถดึงราคาได้: {e}")
            return None
    
    def get_historical_klines(self, interval="1h", total_records=365*24*3):
        """ดึงข้อมูลประวัติศาสตร์จาก Binance (รองรับข้อมูลจำนวนมาก)"""
        
        max_limit = 1000  # Binance limit ต่อ request
        all_data = []
        
        print(f"📥 ดึงข้อมูลประวัติศาสตร์ {self.symbol} ({total_records:,} records)...")
        
        try:
            # คำนวณจำนวน requests ที่ต้องทำ
            num_requests = (total_records + max_limit - 1) // max_limit
            print(f"🔄 ต้องทำ {num_requests} requests...")
            
            end_time = None  # เริ่มจากปัจจุบัน
            
            for i in range(num_requests):
                print(f"   📡 Request {i+1}/{num_requests}...", end=" ")
                
                url = f"{self.base_url}/klines"
                params = {
                    "symbol": self.symbol,
                    "interval": interval,
                    "limit": max_limit
                }
                
                # เพิ่ม endTime สำหรับ request ที่ 2 เป็นต้นไป
                if end_time:
                    params["endTime"] = end_time
                
                response = requests.get(url, params=params, timeout=30)
                data = response.json()
                
                if not data or len(data) == 0:
                    print("❌ ไม่มีข้อมูล")
                    break
                
                # เพิ่มข้อมูลลง list
                all_data.extend(data)
                
                # ตั้ง endTime สำหรับ request ถัดไป (ลบ 1ms เพื่อไม่ให้ซ้ำ)
                end_time = data[0][0] - 1
                
                print(f"✅ ได้ {len(data)} records")
                
                # หยุดถ้าได้ข้อมูลครบแล้ว
                if len(all_data) >= total_records:
                    break
                
                # รอสักครู่เพื่อหลีกเลี่ยง rate limit
                time.sleep(0.1)
            
            # ตัดข้อมูลส่วนเกิน
            if len(all_data) > total_records:
                all_data = all_data[-total_records:]
            
            # แปลงเป็น DataFrame
            df = pd.DataFrame(all_data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # แปลง timestamp เป็น datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # เรียงตาม timestamp (เก่าไปใหม่)
            df = df.sort_index()
            
            # แปลงราคาเป็น float
            price_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in price_columns:
                df[col] = df[col].astype(float)
            
            # เลือกเฉพาะคอลัมน์ที่ต้องการ
            df = df[['open', 'high', 'low', 'close', 'volume']]
            
            # ลบข้อมูลซ้ำ (ถ้ามี)
            df = df[~df.index.duplicated(keep='last')]
            
            print(f"✅ ดึงข้อมูลรวมสำเร็จ: {len(df):,} records")
            print(f"📊 ช่วงเวลา: {df.index[0]} ถึง {df.index[-1]}")
            print(f"💰 ราคาเก่าสุด: ${df['close'].iloc[0]:.2f}")
            print(f"💰 ราคาล่าสุด: ${df['close'].iloc[-1]:.2f}")
            
            # แสดงสถิติเพิ่มเติม
            price_change = ((df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]) * 100
            print(f"📈 การเปลี่ยนแปลงรวม: {price_change:+.2f}%")
            
            return df
            
        except Exception as e:
            print(f"❌ ไม่สามารถดึงข้อมูลประวัติศาสตร์: {e}")
            
            # ถ้าดึงข้อมูลมากไม่ได้ ลองดึงข้อมูลน้อยลง
            print("🔄 ลองดึงข้อมูล 1000 records แทน...")
            return self.get_simple_klines(interval)
    
    def get_simple_klines(self, interval="1h", limit=1000):
        """ดึงข้อมูลแบบง่าย (fallback method)"""
        try:
            url = f"{self.base_url}/klines"
            params = {
                "symbol": self.symbol,
                "interval": interval,
                "limit": limit
            }
            
            response = requests.get(url, params=params, timeout=30)
            data = response.json()
            
            # แปลงเป็น DataFrame
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # แปลง timestamp เป็น datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # แปลงราคาเป็น float
            price_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in price_columns:
                df[col] = df[col].astype(float)
            
            # เลือกเฉพาะคอลัมน์ที่ต้องการ
            df = df[['open', 'high', 'low', 'close', 'volume']]
            
            print(f"✅ ได้ข้อมูล (fallback): {len(df)} records")
            
            return df
            
        except Exception as e:
            print(f"❌ Fallback method ก็ล้มเหลว: {e}")
            return None
    
    def save_data(self, df):
        """บันทึกข้อมูลลง CSV"""
        try:
            df.to_csv(self.data_file)
            print(f"💾 บันทึกข้อมูลลง {self.data_file}")
        except Exception as e:
            print(f"⚠️ ไม่สามารถบันทึกข้อมูล: {e}")
    
    def load_data(self):
        """โหลดข้อมูลจาก CSV"""
        try:
            if os.path.exists(self.data_file):
                df = pd.read_csv(self.data_file, index_col=0, parse_dates=True)
                print(f"📁 โหลดข้อมูลจาก {self.data_file}: {len(df)} records")
                return df
        except Exception as e:
            print(f"⚠️ ไม่สามารถโหลดข้อมูล: {e}")
        return None
    
    def create_features(self, df):
        """สร้าง technical features"""
        print("🔧 กำลังสร้าง Technical Features...")
        
        # ใช้ราคา close เป็นหลัก
        close_prices = df['close']
        high_prices = df['high']
        low_prices = df['low']
        volume = df['volume']
        
        # === Technical Indicators ===
        
        # Moving Averages (รองรับข้อมูลมาก)
        df['EMA12'] = EMAIndicator(close_prices, window=12).ema_indicator()
        df['EMA26'] = EMAIndicator(close_prices, window=26).ema_indicator()
        df['EMA50'] = EMAIndicator(close_prices, window=50).ema_indicator()
        df['SMA20'] = SMAIndicator(close_prices, window=20).sma_indicator()
        df['SMA50'] = SMAIndicator(close_prices, window=50).sma_indicator()
        df['SMA100'] = SMAIndicator(close_prices, window=100).sma_indicator()
        
        # เพิ่ม SMA200 ถ้ามีข้อมูลเพียงพอ
        if len(close_prices) >= 200:
            df['SMA200'] = SMAIndicator(close_prices, window=200).sma_indicator()
            print("   ✅ เพิ่ม SMA200")
        else:
            print(f"   ⚠️ ข้อมูลไม่พอสำหรับ SMA200 (มี {len(close_prices)}, ต้องการ 200)")
        
        # RSI (multiple timeframes)
        df['RSI'] = RSIIndicator(close_prices, window=14).rsi()
        df['RSI_30'] = RSIIndicator(close_prices, window=30).rsi()
        df['RSI_50'] = RSIIndicator(close_prices, window=50).rsi()
        
        # Bollinger Bands
        bb = BollingerBands(close_prices, window=20)
        df['BB_Upper'] = bb.bollinger_hband()
        df['BB_Lower'] = bb.bollinger_lband()
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        df['BB_Position'] = (close_prices - df['BB_Lower']) / df['BB_Width']
        
        # === Price Features ===
        
        # Returns
        df['Return_1'] = close_prices.pct_change()
        df['Return_6'] = close_prices.pct_change(6)  # 6 hours
        df['Return_24'] = close_prices.pct_change(24)  # 24 hours
        
        # Volatility
        df['Volatility_6'] = close_prices.rolling(6).std()
        df['Volatility_24'] = close_prices.rolling(24).std()
        
        # High-Low features
        df['HL_Ratio'] = (high_prices - low_prices) / close_prices
        df['Price_Position'] = (close_prices - low_prices) / (high_prices - low_prices)
        
        # Price vs Moving Averages (ปรับปรุงให้รองรับ SMA ใหม่)
        df['Price_EMA12_Ratio'] = close_prices / df['EMA12']
        df['Price_EMA50_Ratio'] = close_prices / df['EMA50']
        df['Price_SMA20_Ratio'] = close_prices / df['SMA20']
        df['Price_SMA50_Ratio'] = close_prices / df['SMA50']
        df['Price_SMA100_Ratio'] = close_prices / df['SMA100']
        df['EMA12_EMA26_Ratio'] = df['EMA12'] / df['EMA26']
        df['SMA20_SMA50_Ratio'] = df['SMA20'] / df['SMA50']
        
        # เพิ่ม SMA200 ratios ถ้ามี
        if 'SMA200' in df.columns:
            df['Price_SMA200_Ratio'] = close_prices / df['SMA200']
            df['SMA50_SMA200_Ratio'] = df['SMA50'] / df['SMA200']
            
        # Long-term trend indicators (ใช้ข้อมูลมาก)
        if len(close_prices) >= 100:
            df['Trend_Score_100'] = (close_prices > df['SMA100']).astype(int)
            df['Above_SMA_Count'] = (
                (close_prices > df['SMA20']).astype(int) +
                (close_prices > df['SMA50']).astype(int) +
                (close_prices > df['SMA100']).astype(int)
            )
        
        # Long-term momentum
        if len(close_prices) >= 200:
            df['ROC_100'] = ((close_prices - close_prices.shift(100)) / close_prices.shift(100)) * 100
            df['ROC_200'] = ((close_prices - close_prices.shift(200)) / close_prices.shift(200)) * 100
        
        # === Volume Features ===
        df['Volume_MA'] = volume.rolling(24).mean()
        df['Volume_Ratio'] = volume / df['Volume_MA']
        df['Price_Volume'] = close_prices * volume
        
        # === Momentum Features ===
        df['ROC_6'] = ((close_prices - close_prices.shift(6)) / close_prices.shift(6)) * 100
        df['ROC_24'] = ((close_prices - close_prices.shift(24)) / close_prices.shift(24)) * 100
        df['Momentum'] = close_prices / close_prices.shift(12)
        
        # === Time Features ===
        df['Hour'] = df.index.hour
        df['Day_of_Week'] = df.index.dayofweek
        df['Is_Weekend'] = (df.index.dayofweek >= 5).astype(int)
        df['Is_US_Trading'] = ((df.index.hour >= 14) & (df.index.hour <= 21)).astype(int)  # US market hours
        df['Is_Asian_Trading'] = ((df.index.hour >= 1) & (df.index.hour <= 9)).astype(int)  # Asian market hours
        
        # === Advanced Features (ใช้ข้อมูลมาก) ===
        
        # Support/Resistance levels (multiple timeframes)
        df['Resistance_24'] = high_prices.rolling(24).max()
        df['Support_24'] = low_prices.rolling(24).min()
        df['Resistance_168'] = high_prices.rolling(168).max()  # 1 สัปดาห์
        df['Support_168'] = low_prices.rolling(168).min()
        
        if len(close_prices) >= 720:  # 1 เดือน
            df['Resistance_720'] = high_prices.rolling(720).max()
            df['Support_720'] = low_prices.rolling(720).min()
        
        df['Near_Resistance_24'] = (close_prices / df['Resistance_24'] > 0.98).astype(int)
        df['Near_Support_24'] = (close_prices / df['Support_24'] < 1.02).astype(int)
        
        # Trend strength (multiple timeframes)
        df['Trend_6'] = (df['EMA12'] > df['EMA12'].shift(6)).astype(int)
        df['Trend_24'] = (df['SMA20'] > df['SMA20'].shift(24)).astype(int)
        df['Trend_168'] = (df['SMA50'] > df['SMA50'].shift(168)).astype(int)  # สัปดาห์
        
        # Price position in long-term ranges
        if 'Resistance_168' in df.columns:
            df['Weekly_Position'] = (close_prices - df['Support_168']) / (df['Resistance_168'] - df['Support_168'])
        
        # Volatility regime (long-term vs short-term)
        df['Vol_Regime'] = (df['Volatility_24'] > df['Volatility_24'].rolling(168).mean()).astype(int)
        
        # Fibonacci retracement levels (improved)
        df['Fib_Position_24'] = (close_prices - df['Support_24']) / (df['Resistance_24'] - df['Support_24'])
        if 'Support_168' in df.columns:
            df['Fib_Position_Weekly'] = (close_prices - df['Support_168']) / (df['Resistance_168'] - df['Support_168'])
        
        # Price acceleration (rate of change of rate of change)
        df['Price_Acceleration'] = df['Return_1'].diff()
        df['Volume_Acceleration'] = df['Volume_Ratio'].diff()
        
        # Long-term price patterns
        if len(close_prices) >= 100:
            # Higher highs, lower lows pattern
            df['Higher_High'] = (close_prices > close_prices.rolling(50).max().shift(1)).astype(int)
            df['Lower_Low'] = (close_prices < close_prices.rolling(50).min().shift(1)).astype(int)
            
            # Breakout patterns
            df['Breakout_Up'] = ((close_prices > df['Resistance_24'].shift(1)) & 
                                (close_prices.shift(1) <= df['Resistance_24'].shift(1))).astype(int)
            df['Breakout_Down'] = ((close_prices < df['Support_24'].shift(1)) & 
                                  (close_prices.shift(1) >= df['Support_24'].shift(1))).astype(int)
        
        print(f"✅ สร้าง features สำเร็จ: {len(df.columns)} total columns")
        
        # แสดงหมวดหมู่ features
        feature_cols = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
        technical_count = len([c for c in feature_cols if any(x in c for x in ['EMA', 'SMA', 'RSI', 'BB'])])
        price_count = len([c for c in feature_cols if any(x in c for x in ['Return', 'Ratio', 'Position', 'HL'])])
        volume_count = len([c for c in feature_cols if 'Volume' in c])
        time_count = len([c for c in feature_cols if any(x in c for x in ['Hour', 'Day', 'Is_'])])
        
        print(f"   📊 Technical: {technical_count}")
        print(f"   💰 Price: {price_count}")
        print(f"   📈 Volume: {volume_count}")
        print(f"   ⏰ Time: {time_count}")
        
        return df
    
    def prepare_ml_data(self, df):
        """เตรียมข้อมูลสำหรับ Machine Learning"""
        print("🤖 เตรียมข้อมูลสำหรับ ML...")
        
        # สร้าง target: ราคาขึ้นในอีก 6 ชั่วโมงข้างหน้า
        df['Target_6h'] = (df['close'].shift(-6) > df['close']).astype(int)
        
        # ลบข้อมูลที่ไม่ครบ
        df_clean = df.dropna()
        
        if len(df_clean) < 100:
            print("❌ ข้อมูลไม่เพียงพอสำหรับ ML")
            return None, None, None
        
        # เลือก features
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'Target_6h']
        feature_cols = [col for col in df_clean.columns if col not in exclude_cols]
        
        X = df_clean[feature_cols].fillna(0)
        y = df_clean['Target_6h']
        
        print(f"📊 ML Data:")
        print(f"   📋 Features: {len(feature_cols)}")
        print(f"   📅 Records: {len(X)}")
        print(f"   📈 Up: {(y == 1).sum()} ({(y == 1).mean():.1%})")
        print(f"   📉 Down: {(y == 0).sum()} ({(y == 0).mean():.1%})")
        
        return X, y, feature_cols
    
    def train_model(self, X, y, feature_cols):
        """เทรนโมเดล"""
        print("🚀 กำลังเทรนโมเดล...")
        
        # แบ่งข้อมูล (80/20)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # สร้างโมเดล ensemble
        self.model = VotingClassifier([
            ('rf', RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                min_samples_split=5,
                random_state=42
            )),
            ('xgb', XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss'
            ))
        ], voting='soft')
        
        # เทรน
        self.model.fit(X_train_scaled, y_train)
        
        # ทดสอบ
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ เทรนเสร็จ!")
        print(f"   🎯 Accuracy: {accuracy:.2%}")
        print(f"   📊 Training: {len(X_train)} records")
        print(f"   🧪 Testing: {len(X_test)} records")
        
        self.feature_columns = feature_cols
        
        # บันทึกโมเดล
        try:
            import joblib
            joblib.dump(self.model, self.model_file)
            joblib.dump(self.scaler, self.scaler_file)
            print(f"💾 บันทึกโมเดลลง {self.model_file}")
        except:
            print("⚠️ ไม่สามารถบันทึกโมเดลได้")
        
        return accuracy
    
    def predict_next_6h(self, df):
        """พยากรณ์ 6 ชั่วโมงข้างหน้า"""
        if self.model is None or self.scaler is None:
            print("❌ ไม่มีโมเดลสำหรับพยากรณ์")
            return None
        
        try:
            # เตรียมข้อมูลล่าสุด
            latest_data = df[self.feature_columns].iloc[-1:].fillna(0)
            latest_scaled = self.scaler.transform(latest_data)
            
            # พยากรณ์
            prediction_prob = self.model.predict_proba(latest_scaled)[0]
            prediction = self.model.predict(latest_scaled)[0]
            
            return {
                'direction': 'UP' if prediction == 1 else 'DOWN',
                'probability_up': prediction_prob[1],
                'probability_down': prediction_prob[0],
                'confidence': max(prediction_prob)
            }
            
        except Exception as e:
            print(f"❌ ไม่สามารถพยากรณ์ได้: {e}")
            return None
    
    def run_training(self, data_years=3):
        """รันการเทรนโมเดลครั้งเดียว"""
        print("🏋️ เริ่มการเทรนโมเดล...")
        
        # คำนวณจำนวน records ที่ต้องการ
        records_per_year = 365 * 24  # 1 ปี = 365 วัน * 24 ชั่วโมง
        total_records = records_per_year * data_years
        
        print(f"📊 เป้าหมาย: {total_records:,} records ({data_years} ปี)")
        
        # ดึงข้อมูลประวัติศาสตร์
        df = self.get_historical_klines(interval="1h", total_records=total_records)
        
        if df is None:
            print("❌ ไม่สามารถดึงข้อมูลได้")
            return False
        
        # บันทึกข้อมูล
        self.save_data(df)
        
        # สร้าง features
        df = self.create_features(df)
        
        # เตรียมข้อมูล ML
        X, y, feature_cols = self.prepare_ml_data(df)
        
        if X is None:
            return False
        
        # เทรนโมเดล
        accuracy = self.train_model(X, y, feature_cols)
        
        print(f"\n🎉 การเทรนเสร็จสิ้น!")
        print(f"   📊 ข้อมูลที่ใช้: {len(df):,} records")
        print(f"   🎯 Accuracy: {accuracy:.2%}")
        print(f"   📅 ช่วงข้อมูล: {df.index[0].strftime('%Y-%m-%d')} ถึง {df.index[-1].strftime('%Y-%m-%d')}")
        
        return True
    
    def run_live_prediction(self, update_interval=10):
        """รันการพยากรณ์แบบ real-time"""
        print("🔴 เริ่มการพยากรณ์แบบ Live...")
        
        # โหลดข้อมูลและโมเดล
        df = self.load_data()
        
        if df is None:
            print("❌ ไม่พบข้อมูลประวัติศาสตร์ - กรุณารันการเทรนก่อน")
            return
        
        # โหลดโมเดล
        try:
            import joblib
            self.model = joblib.load(self.model_file)
            self.scaler = joblib.load(self.scaler_file)
            print("✅ โหลดโมเดลสำเร็จ")
        except:
            print("❌ ไม่พบโมเดล - กรุณารันการเทรนก่อน")
            return
        
        # สร้าง features (ใช้ข้อมูลประวัติศาสตร์)
        df = self.create_features(df)
        
        # เลือก feature columns
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'Target_6h']
        self.feature_columns = [col for col in df.columns if col not in exclude_cols]
        
        print(f"🚀 เริ่มการพยากรณ์ live - อัพเดททุก {update_interval} วินาที")
        print("กด Ctrl+C เพื่อหยุด")
        
        try:
            while True:
                # ดึงราคาปัจจุบัน
                current_price = self.get_current_price()
                
                if current_price is None:
                    time.sleep(update_interval)
                    continue
                
                # พยากรณ์ (ใช้ข้อมูลล่าสุดที่มี)
                prediction = self.predict_next_6h(df)
                
                # แสดงผล
                current_time = datetime.now().strftime('%H:%M:%S')
                
                print(f"\n[{current_time}] 💰 ราคา PAXG: ${current_price:.2f}")
                
                if prediction:
                    direction_emoji = "📈" if prediction['direction'] == 'UP' else "📉"
                    confidence_level = "🔥" if prediction['confidence'] > 0.7 else "⚡" if prediction['confidence'] > 0.6 else "💫"
                    
                    print(f"🔮 พยากรณ์ 6 ชั่วโมงข้างหน้า: {direction_emoji} {prediction['direction']}")
                    print(f"📊 ความน่าจะเป็น UP: {prediction['probability_up']*100:.1f}%")
                    print(f"{confidence_level} ความมั่นใจ: {prediction['confidence']*100:.1f}%")
                    
                    # แสดงสัญญาณการซื้อขาย
                    if prediction['confidence'] > 0.65:
                        signal = "🟢 BUY" if prediction['direction'] == 'UP' else "🔴 SELL"
                        print(f"📡 สัญญาณ: {signal}")
                    else:
                        print(f"🟡 สัญญาณ: HOLD (ความมั่นใจต่ำ)")
                else:
                    print("⚠️ ไม่สามารถพยากรณ์ได้")
                
                time.sleep(update_interval)
                
        except KeyboardInterrupt:
            print("\n\n🛑 หยุดการพยากรณ์")
    
    def show_status(self):
        """แสดงสถานะระบบ"""
        print("📊 สถานะระบบ:")
        
        # ตรวจสอบการเชื่อมต่อ Binance
        current_price = self.get_current_price()
        if current_price:
            print(f"   ✅ เชื่อมต่อ Binance สำเร็จ - ราคาปัจจุบัน: ${current_price:.2f}")
        else:
            print(f"   ❌ ไม่สามารถเชื่อมต่อ Binance ได้")
        
        # ตรวจสอบข้อมูลประวัติศาสตร์
        if os.path.exists(self.data_file):
            df = self.load_data()
            print(f"   ✅ มีข้อมูลประวัติศาสตร์: {len(df)} records")
            print(f"   📅 ข้อมูลล่าสุด: {df.index[-1]}")
        else:
            print(f"   ❌ ไม่มีข้อมูลประวัติศาสตร์")
        
        # ตรวจสอบโมเดล
        if os.path.exists(self.model_file):
            print(f"   ✅ มีโมเดลที่เทรนแล้ว")
        else:
            print(f"   ❌ ไม่มีโมเดล")

def main():
    predictor = BinanceGoldPredictor()
    
    print("🌟 ระบบพยากรณ์ราคาทองแบบ Real-time")
    print("=" * 60)
    
    while True:
        print("\n📋 เลือกการทำงาน:")
        print("1. 🏋️ เทรนโมเดล (ครั้งแรก/อัพเดทโมเดล)")
        print("2. 🔴 เริ่มการพยากรณ์แบบ Live")
        print("3. 📊 ตรวจสอบสถานะระบบ")
        print("4. 💰 ดูราคาปัจจุบันเท่านั้น")
        print("5. 🚪 ออก")
        
        try:
            choice = input("\nเลือก (1-5): ").strip()
            
            if choice == "1":
                years = input("ต้องการข้อมูลกี่ปี? (1-5, default: 3): ").strip()
                years = int(years) if years.isdigit() and 1 <= int(years) <= 5 else 3
                print(f"📊 เลือกใช้ข้อมูล {years} ปี")
                predictor.run_training(data_years=years)
                
            elif choice == "2":
                interval = input("อัพเดททุกกี่วินาที? (default: 10): ").strip()
                interval = int(interval) if interval.isdigit() else 10
                predictor.run_live_prediction(interval)
                
            elif choice == "3":
                predictor.show_status()
                
            elif choice == "4":
                price = predictor.get_current_price()
                if price:
                    print(f"💰 ราคา PAXG ปัจจุบัน: ${price:.2f}")
                
            elif choice == "5":
                print("👋 ขอบคุณที่ใช้ระบบ!")
                break
                
            else:
                print("❌ กรุณาเลือก 1-5")
                
        except KeyboardInterrupt:
            print("\n👋 ขอบคุณที่ใช้ระบบ!")
            break
        except Exception as e:
            print(f"❌ เกิดข้อผิดพลาด: {e}")

if __name__ == "__main__":
    main()