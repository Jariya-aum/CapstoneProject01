# test_api.py
import requests
import json
import numpy as np

url = "http://127.0.0.1:5001/invocations"
headers = {"Content-Type": "application/json"}

print("Testing MLflow Model API")
print("="*50)

# สร้างข้อมูลทดสอบ (40 features)
# S1: 4 bands × 10 months = 40 features
feature_names = []
for month in ['05', '06', '07', '08', '09', '10', '11', '12', '01', '02']:
    feature_names.extend([
        f"S1VV_{month}",
        f"S1VH_{month}",
        f"S1Diff_{month}",
        f"S1Ratio_{month}"
    ])

print(f"จำนวน features: {len(feature_names)}")
print(f"ตัวอย่าง features: {feature_names[:5]}")

# สร้างข้อมูลจำลอง (ค่าสุ่ม)
np.random.seed(42)
test_data = np.random.rand(1, 40).tolist()

payload = {
    "dataframe_split": {
        "columns": feature_names,
        "data": test_data
    }
}

print("\nกำลังเชื่อมต่อกับ MLflow Model Server...")

try:
    response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=10)
    
    if response.status_code == 200:
        print("\nPrediction successful!")
        print(f"HTTP Status: {response.status_code}")
        
        result = response.json()
        print(f"Raw Result: {result}")
        
        # แปลผลลัพธ์
        prediction = result['predictions'][0]
        label = "Corn (ข้าวโพด)" if prediction == 1 else "Non-Corn (ไม่ใช่ข้าวโพด)"
        
        print("\n" + "="*50)
        print(f"ผลการทำนาย: {label}")
        print("="*50)
    else:
        print(f"\nHTTP Error: Status {response.status_code}")
        print(f"Response: {response.text}")
        
except requests.exceptions.ConnectionError:
    print("\nConnection Error!")
    print("\nสาเหตุ: MLflow Model Server ยังไม่ได้เปิด")
    print("\nวิธีแก้ไข:")
    print("1. เปิด Terminal ใหม่")
    print("2. รันคำสั่ง:")
    print("   mlflow models serve -m models:/RandomForest/1 -p 5001")
    print("   หรือ")
    print("   mlflow models serve -m runs:/<run_id>/model -p 5001")
    
except requests.exceptions.Timeout:
    print("\nTimeout Error!")
    print("Server ใช้เวลานานเกินไป")
    
except Exception as e:
    print(f"\nUnexpected Error: {type(e).__name__}")
    print(f"Details: {e}")

print("\nTest completed!")