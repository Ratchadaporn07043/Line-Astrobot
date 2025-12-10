#!/usr/bin/env python3
"""
สคริปต์ตรวจสอบว่าข้อมูลมาจาก MongoDB จริงๆ หรือไม่
"""

import os
import sys
from dotenv import load_dotenv
from pymongo import MongoClient
import json

# โหลด environment variables
load_dotenv()

print("="*60)
print("🔍 ตรวจสอบว่าข้อมูลมาจาก MongoDB")
print("="*60)

# 1. ตรวจสอบ MongoDB connection
mongo_uri = os.getenv("MONGO_URL")
if not mongo_uri or mongo_uri == "mongodb+srv://your-username:your-password@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority":
    print("❌ MONGO_URL ไม่ได้ตั้งค่าหรือยังเป็นค่า default")
    sys.exit(1)

try:
    from config import SUMMARY_DB_NAME
    
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=10000)
    db = client[SUMMARY_DB_NAME]
    
    print(f"\n✅ เชื่อมต่อ MongoDB สำเร็จ")
    print(f"   Database: {SUMMARY_DB_NAME}")
    
    # 2. ตรวจสอบ collections
    collections = [
        "processed_text_chunks",
        "processed_image_chunks",
        "processed_table_chunks",
    ]
    
    total_docs = 0
    total_with_embeddings = 0
    
    for collection_name in collections:
        collection = db[collection_name]
        total_count = collection.count_documents({})
        with_embeddings = collection.count_documents({"embeddings": {"$exists": True, "$ne": None}})
        
        total_docs += total_count
        total_with_embeddings += with_embeddings
        
        print(f"\n📊 Collection: {collection_name}")
        print(f"   จำนวนเอกสารทั้งหมด: {total_count}")
        print(f"   จำนวนเอกสารที่มี embeddings: {with_embeddings}")
        
        # ตัวอย่างข้อมูล
        sample = collection.find_one({"embeddings": {"$exists": True, "$ne": None}})
        if sample:
            print(f"   ตัวอย่างข้อมูล:")
            print(f"     - _id: {sample.get('_id', 'N/A')}")
            print(f"     - page: {sample.get('page', 'N/A')}")
            print(f"     - source: {sample.get('source', 'N/A')}")
            print(f"     - chunk_id: {sample.get('chunk_id', 'N/A')}")
            print(f"     - มี summary: {'Yes' if sample.get('summary') else 'No'}")
            print(f"     - มี text: {'Yes' if sample.get('text') else 'No'}")
            if sample.get('summary'):
                print(f"     - summary length: {len(sample.get('summary', ''))} ตัวอักษร")
            if sample.get('text'):
                print(f"     - text length: {len(sample.get('text', ''))} ตัวอักษร")
    
    print(f"\n📊 สรุป:")
    print(f"   จำนวนเอกสารทั้งหมด: {total_docs}")
    print(f"   จำนวนเอกสารที่มี embeddings: {total_with_embeddings}")
    
    # 3. ตรวจสอบ dataset ที่สร้างแล้ว
    dataset_file = "dataset_from_mongo.json"
    if os.path.exists(dataset_file):
        print(f"\n📁 ตรวจสอบ dataset: {dataset_file}")
        with open(dataset_file, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        print(f"   จำนวนคำถาม-คำตอบ: {len(dataset)}")
        
        # ตรวจสอบว่ามีคำตอบที่บอกว่าไม่มีข้อมูลหรือไม่
        invalid_phrases = [
            "ไม่ได้ถูกระบุไว้ในเนื้อหา", "ไม่ได้ระบุในเนื้อหา", "ไม่ปรากฏในเนื้อหา",
            "ยังไม่มีการระบุ", "ไม่มีการระบุ", "ไม่ได้ระบุไว้", "ไม่พบในเนื้อหา",
            "ไม่มีในเนื้อหา", "ไม่มีให้ในเนื้อหา", "ไม่มีข้อมูล", "ไม่พบข้อมูล"
        ]
        
        invalid_count = 0
        for item in dataset:
            answer = str(item.get("ground_truth", "")).strip()
            if any(phrase in answer for phrase in invalid_phrases):
                invalid_count += 1
        
        print(f"   พบคำตอบที่บอกว่าไม่มีข้อมูล: {invalid_count} รายการ")
        
        if invalid_count > 0:
            print(f"\n⚠️  ยังมีคำตอบที่บอกว่าไม่มีข้อมูล {invalid_count} รายการ")
            print(f"   ควรรัน generate_ragas_dataset_from_mongo.py ใหม่เพื่อกรองออก")
        else:
            print(f"\n✅ ไม่พบคำตอบที่บอกว่าไม่มีข้อมูล")
    
    client.close()
    
    print("\n" + "="*60)
    print("✅ การตรวจสอบเสร็จสิ้น")
    print("="*60)
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
