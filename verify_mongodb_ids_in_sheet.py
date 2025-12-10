#!/usr/bin/env python3
"""
สคริปต์ตรวจสอบว่า MongoDB IDs ใน Google Sheets มาจาก MongoDB จริงๆ
"""

import os
from dotenv import load_dotenv
from pymongo import MongoClient
import gspread
from google.oauth2.service_account import Credentials

load_dotenv()

print("="*60)
print("🔍 ตรวจสอบ MongoDB IDs ใน Google Sheets")
print("="*60)

# 1. เชื่อมต่อ MongoDB
# ตรวจสอบชื่อตัวแปรที่ใช้ในโปรเจกต์ (ใช้ MONGO_URL ตามที่ใช้ใน generate_ragas_dataset_from_mongo.py)
mongo_uri = os.getenv("MONGO_URL") or os.getenv("MONGODB_URI") or os.getenv("MONGO_URI") or os.getenv("MONGODB_CONNECTION_STRING")
if not mongo_uri or mongo_uri == "mongodb+srv://your-username:your-password@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority":
    print("❌ MONGO_URL ไม่ได้ตั้งค่าหรือยังเป็นค่า default")
    print("   ตรวจสอบ .env file ว่ามีตัวแปร MONGO_URL หรือไม่")
    exit(1)

try:
    from config import SUMMARY_DB_NAME
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=10000)
    db = client[SUMMARY_DB_NAME]
    print(f"✅ เชื่อมต่อ MongoDB สำเร็จ (Database: {SUMMARY_DB_NAME})")
except Exception as e:
    print(f"❌ Error connecting to MongoDB: {e}")
    exit(1)

# 2. เชื่อมต่อ Google Sheets
google_sheets_id = os.getenv("GOOGLE_SHEETS_ID")
if not google_sheets_id:
    print("❌ GOOGLE_SHEETS_ID ไม่ได้ตั้งค่า")
    exit(1)

# Extract ID from URL if needed
if "/" in google_sheets_id:
    google_sheets_id = google_sheets_id.split("/")[-1]

try:
    # Load credentials
    creds_path = os.getenv("GOOGLE_SHEETS_CREDENTIALS_PATH")
    if creds_path and os.path.exists(creds_path):
        creds = Credentials.from_service_account_file(creds_path)
    else:
        creds_json = os.getenv("GOOGLE_SHEETS_CREDENTIALS_JSON")
        if creds_json:
            import json
            creds = Credentials.from_service_account_info(json.loads(creds_json))
        else:
            print("❌ ไม่พบ Google Sheets credentials")
            exit(1)
    
    gc = gspread.authorize(creds)
    sheet = gc.open_by_key(google_sheets_id)
    
    # เปิด worksheet "Dataset"
    try:
        worksheet = sheet.worksheet("Dataset")
    except gspread.exceptions.WorksheetNotFound:
        print("❌ ไม่พบ worksheet 'Dataset'")
        exit(1)
    
    print("✅ เชื่อมต่อ Google Sheets สำเร็จ")
    
    # 3. อ่านข้อมูลจาก Google Sheets
    all_values = worksheet.get_all_values()
    if not all_values:
        print("❌ ไม่พบข้อมูลใน Google Sheets")
        exit(1)
    
    headers = all_values[0]
    data_rows = all_values[1:]
    
    # หา index ของคอลัมน์ MongoDB ID
    try:
        mongo_id_col_idx = headers.index("MongoDB ID")
    except ValueError:
        print("❌ ไม่พบคอลัมน์ 'MongoDB ID' ใน Google Sheets")
        exit(1)
    
    # 4. ตรวจสอบ MongoDB IDs
    print(f"\n📊 ตรวจสอบ {len(data_rows)} รายการ...")
    
    valid_ids = 0
    invalid_ids = 0
    not_found_ids = []
    found_ids = []
    
    for i, row in enumerate(data_rows, start=2):  # start=2 เพราะ row 1 เป็น header
        mongo_id = row[mongo_id_col_idx] if mongo_id_col_idx < len(row) else ""
        mongo_id = mongo_id.strip()
        
        if not mongo_id or mongo_id == "unknown":
            invalid_ids += 1
            continue
        
        # ตรวจสอบว่า ID นี้มีใน MongoDB หรือไม่
        try:
            from bson import ObjectId
            doc = db.chunks.find_one({"_id": ObjectId(mongo_id)})
            if doc:
                valid_ids += 1
                found_ids.append({
                    "row": i,
                    "mongo_id": mongo_id,
                    "page": doc.get("page", "unknown"),
                    "type": doc.get("type", "unknown"),
                    "has_embeddings": "embeddings" in doc and doc["embeddings"]
                })
            else:
                not_found_ids.append({
                    "row": i,
                    "mongo_id": mongo_id
                })
        except Exception as e:
            invalid_ids += 1
            print(f"⚠️  Row {i}: Invalid MongoDB ID format: {mongo_id} ({e})")
    
    # 5. แสดงผลลัพธ์
    print(f"\n📋 สรุปผลการตรวจสอบ:")
    print(f"   ✅ MongoDB IDs ที่พบในฐานข้อมูล: {valid_ids} รายการ")
    print(f"   ❌ MongoDB IDs ที่ไม่พบในฐานข้อมูล: {len(not_found_ids)} รายการ")
    print(f"   ⚠️  MongoDB IDs ที่ไม่ถูกต้องหรือเป็น 'unknown': {invalid_ids} รายการ")
    
    if found_ids:
        print(f"\n✅ ตัวอย่าง MongoDB IDs ที่พบในฐานข้อมูล:")
        for item in found_ids[:5]:
            print(f"   Row {item['row']}: {item['mongo_id'][:20]}... (page={item['page']}, type={item['type']}, has_embeddings={item['has_embeddings']})")
    
    if not_found_ids:
        print(f"\n❌ MongoDB IDs ที่ไม่พบในฐานข้อมูล:")
        for item in not_found_ids[:10]:
            print(f"   Row {item['row']}: {item['mongo_id']}")
        if len(not_found_ids) > 10:
            print(f"   ... และอีก {len(not_found_ids) - 10} รายการ")
    
    print(f"\n" + "="*60)
    if valid_ids == len(data_rows):
        print("✅ ทุก MongoDB ID มาจาก MongoDB จริงๆ")
    else:
        print(f"⚠️  พบ MongoDB IDs ที่ไม่พบในฐานข้อมูล {len(not_found_ids)} รายการ")
    print("="*60)
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
