import time
import requests

while True:
    res = requests.get("https://api.binance.com/api/v3/ticker/price?symbol=PAXGUSDT").json()
    print(f"[{time.strftime('%H:%M:%S')}] ราคาทอง: ${res['price']}")
    time.sleep(10)
