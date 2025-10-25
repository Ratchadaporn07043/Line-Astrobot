#!/usr/bin/env python3
"""
สคริปต์ตรวจสอบ collections ใน MongoDB
"""
import os
from pymongo import MongoClient
from dotenv import load_dotenv

def check_collections():
    """ตรวจสอบ collections ใน MongoDB"""
    print("🔍 กำลังตรวจสอบ collections ใน MongoDB...")
    
    # ลองโหลด .env ก่อน
    dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path)
        print("✅ โหลดไฟล์ .env สำเร็จ")
    else:
        print("⚠️ ไม่พบไฟล์ .env")
    
    # ตรวจสอบตัวแปรสภาพแวดล้อม
    mongo_url = os.getenv("MONGO_URL")
    
    if not mongo_url:
        print("❌ ไม่พบ MONGO_URL ในตัวแปรสภาพแวดล้อม")
        return False
    
    try:
        # ทดสอบการเชื่อมต่อ
        print(f"🔗 กำลังเชื่อมต่อ: {mongo_url[:50]}...")
        client = MongoClient(mongo_url, serverSelectionTimeoutMS=5000)
        
        # ทดสอบการ ping
        client.admin.command('ping')
        print("✅ เชื่อมต่อ MongoDB สำเร็จ!")
        
        # ตรวจสอบฐานข้อมูล
        db_name = os.getenv("DB_NAME", "astrobot")
        db = client[db_name]
        collections = db.list_collection_names()
        
        print(f"📊 ฐานข้อมูล: {db_name}")
        print(f"📁 Collections ที่มีอยู่: {len(collections)}")
        
        if collections:
            for collection in collections:
                count = db[collection].count_documents({})
                print(f"   - {collection}: {count} documents")
                
                # แสดงตัวอย่างข้อมูลสำหรับ collections ใหม่
                if collection in ["original_doc", "summary_doc"]:
                    sample = db[collection].find_one()
                    if sample:
                        print(f"     📄 ตัวอย่าง: {sample.get('type', 'unknown')}")
                        if collection == "original_doc":
                            source_files = sample.get("source_files", {})
                            print(f"     📝 Text length: {source_files.get('text', {}).get('length', 0)}")
                            print(f"     🖼️ Images count: {source_files.get('images', {}).get('count', 0)}")
                            print(f"     📊 Tables count: {source_files.get('tables', {}).get('count', 0)}")
                        elif collection == "summary_doc":
                            summary_data = sample.get("summary_data", {})
                            print(f"     📝 Text chunks: {summary_data.get('text_chunks', {}).get('count', 0)}")
                            print(f"     🖼️ Image chunks: {summary_data.get('image_chunks', {}).get('count', 0)}")
                            print(f"     📊 Table chunks: {summary_data.get('table_chunks', {}).get('count', 0)}")
        else:
            print("   - ไม่มี collections")
        
        client.close()
        return True
        
    except Exception as e:
        print(f"❌ ไม่สามารถเชื่อมต่อ MongoDB ได้: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🔍 ตรวจสอบ Collections ใน MongoDB")
    print("=" * 60)
    
    check_collections()
