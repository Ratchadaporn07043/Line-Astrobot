#!/usr/bin/env python3
"""
สคริปต์ตรวจสอบว่า MongoDB IDs ใน dataset JSON มาจาก MongoDB จริงๆ
"""

import os
import json
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

print("="*60)
print("🔍 ตรวจสอบ MongoDB IDs ใน Dataset JSON")
print("="*60)

# 1. โหลด dataset
dataset_file = "dataset_from_mongo.json"
if not os.path.exists(dataset_file):
    print(f"❌ ไม่พบไฟล์ {dataset_file}")
    exit(1)

with open(dataset_file, 'r', encoding='utf-8') as f:
    dataset = json.load(f)

print(f"\n📊 จำนวนคำถาม-คำตอบ: {len(dataset)}")

# 2. เชื่อมต่อ MongoDB
mongo_uri = os.getenv("MONGO_URL")
if not mongo_uri or mongo_uri == "mongodb+srv://your-username:your-password@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority":
    print("❌ MONGO_URL ไม่ได้ตั้งค่าหรือยังเป็นค่า default")
    exit(1)

try:
    from config import SUMMARY_DB_NAME
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=10000)
    db = client[SUMMARY_DB_NAME]
    print(f"✅ เชื่อมต่อ MongoDB สำเร็จ (Database: {SUMMARY_DB_NAME})")
except Exception as e:
    print(f"❌ Error connecting to MongoDB: {e}")
    exit(1)

# 3. ตรวจสอบ MongoDB IDs
print(f"\n📊 ตรวจสอบ {len(dataset)} รายการ...")

valid_ids = 0
invalid_ids = 0
not_found_ids = []
found_ids = []
items_without_id = []

for i, item in enumerate(dataset, 1):
    mongo_id = item.get("_mongodb_id", "unknown")
    
    if not mongo_id or mongo_id == "unknown":
        items_without_id.append({
            "index": i,
            "question": item.get("question", "")[:60]
        })
        invalid_ids += 1
        continue
    
    # ตรวจสอบว่า ID นี้มีใน MongoDB หรือไม่
    try:
        from bson import ObjectId
        doc = db.chunks.find_one({"_id": ObjectId(mongo_id)})
        if doc:
            valid_ids += 1
            found_ids.append({
                "index": i,
                "mongo_id": mongo_id,
                "page": doc.get("page", "unknown"),
                "type": doc.get("type", "unknown"),
                "has_embeddings": "embeddings" in doc and doc["embeddings"]
            })
        else:
            not_found_ids.append({
                "index": i,
                "mongo_id": mongo_id,
                "question": item.get("question", "")[:60]
            })
    except Exception as e:
        invalid_ids += 1
        print(f"⚠️  Item {i}: Invalid MongoDB ID format: {mongo_id[:20]}... ({e})")

# 4. แสดงผลลัพธ์
print(f"\n📋 สรุปผลการตรวจสอบ:")
print(f"   ✅ MongoDB IDs ที่พบในฐานข้อมูล: {valid_ids} รายการ")
print(f"   ❌ MongoDB IDs ที่ไม่พบในฐานข้อมูล: {len(not_found_ids)} รายการ")
print(f"   ⚠️  MongoDB IDs ที่ไม่ถูกต้องหรือเป็น 'unknown': {invalid_ids} รายการ")
print(f"      - ไม่มี MongoDB ID: {len(items_without_id)} รายการ")

if found_ids:
    print(f"\n✅ ตัวอย่าง MongoDB IDs ที่พบในฐานข้อมูล:")
    for item in found_ids[:5]:
        print(f"   Item {item['index']}: {item['mongo_id'][:20]}... (page={item['page']}, type={item['type']}, has_embeddings={item['has_embeddings']})")

if not_found_ids:
    print(f"\n❌ MongoDB IDs ที่ไม่พบในฐานข้อมูล:")
    for item in not_found_ids[:10]:
        print(f"   Item {item['index']}: {item['mongo_id']} - {item['question']}...")
    if len(not_found_ids) > 10:
        print(f"   ... และอีก {len(not_found_ids) - 10} รายการ")

if items_without_id:
    print(f"\n⚠️  รายการที่ไม่มี MongoDB ID:")
    for item in items_without_id[:10]:
        print(f"   Item {item['index']}: {item['question']}...")
    if len(items_without_id) > 10:
        print(f"   ... และอีก {len(items_without_id) - 10} รายการ")

# 5. ตรวจสอบคำตอบที่บอกว่าไม่มีข้อมูล
invalid_phrases = [
    "ไม่มีในเนื้อหา", "ไม่มีให้ในเนื้อหา", "ไม่มีข้อมูล", "ไม่พบข้อมูล",
    "ไม่ได้ถูกระบุไว้ในเนื้อหา", "ไม่ได้ระบุในเนื้อหา", "ไม่ได้ถูกระบุในเนื้อหา",
    "ไม่ปรากฏในเนื้อหา", "ยังไม่มีการระบุ", "ไม่มีการระบุ", "ไม่ได้ระบุไว้",
    "ไม่พบในเนื้อหา", "ไม่มีให้ในเนื้อหาที่ให้มา", "ไม่ได้ระบุไว้ในเนื้อหาที่ให้มา",
    "ไม่สามารถให้ได้จากเนื้อหาที่มีอยู่", "ยังไม่ได้มีการระบุ", "ไม่สามารถให้ได้",
    "ไม่สามารถระบุ", "ไม่สามารถระบุได้", "ไม่สามารถระบุจาก", "ไม่สามารถระบุจากข้อมูล",
    "ไม่สามารถระบุจากข้อมูลที่มีอยู่", "ไม่สามารถระบุการคำนวณ", "ไม่สามารถระบุราศี",
    "ไม่สามารถระบุลัคณา", "ไม่สามารถระบุการคำนวณลัคณา", "ไม่สามารถระบุการคำนวณราศี"
]

invalid_answers = []
for i, item in enumerate(dataset, 1):
    answer = str(item.get("ground_truth", "")).strip()
    if any(phrase in answer for phrase in invalid_phrases):
        invalid_answers.append({
            "index": i,
            "question": item.get("question", "")[:60],
            "answer": answer[:80],
            "mongo_id": item.get("_mongodb_id", "unknown")
        })

if invalid_answers:
    print(f"\n❌ พบคำตอบที่บอกว่าไม่มีข้อมูล: {len(invalid_answers)} รายการ")
    for item in invalid_answers[:5]:
        print(f"   Item {item['index']}: {item['question']}...")
        print(f"      คำตอบ: {item['answer']}...")
        print(f"      MongoDB ID: {item['mongo_id'][:20] if item['mongo_id'] != 'unknown' else 'unknown'}")
    if len(invalid_answers) > 5:
        print(f"   ... และอีก {len(invalid_answers) - 5} รายการ")
else:
    print(f"\n✅ ไม่พบคำตอบที่บอกว่าไม่มีข้อมูล")

print(f"\n" + "="*60)
if valid_ids == len(dataset) and len(invalid_answers) == 0:
    print("✅ ทุก MongoDB ID มาจาก MongoDB จริงๆ และไม่มีคำตอบที่บอกว่าไม่มีข้อมูล")
elif valid_ids > 0:
    print(f"✅ พบ MongoDB IDs ที่มาจาก MongoDB {valid_ids} รายการ")
    if len(not_found_ids) > 0:
        print(f"⚠️  แต่มี MongoDB IDs ที่ไม่พบในฐานข้อมูล {len(not_found_ids)} รายการ")
    if len(invalid_answers) > 0:
        print(f"⚠️  และมีคำตอบที่บอกว่าไม่มีข้อมูล {len(invalid_answers)} รายการ")
        print("   ควรรัน generate_ragas_dataset_from_mongo.py ใหม่เพื่อกรองออก")
else:
    print(f"⚠️  ไม่พบ MongoDB IDs ที่ถูกต้อง - dataset อาจสร้างก่อนการแก้ไข")
    print("   ควรรัน generate_ragas_dataset_from_mongo.py ใหม่")
print("="*60)
