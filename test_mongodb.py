#!/usr/bin/env python3
"""
สคริปต์ทดสอบการเชื่อมต่อ MongoDB และรัน pipeline
"""
import os
import sys
from pymongo import MongoClient
from dotenv import load_dotenv

# เพิ่ม path สำหรับ import modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

def test_mongodb_connection():
    """ทดสอบการเชื่อมต่อ MongoDB"""
    print("🔍 กำลังทดสอบการเชื่อมต่อ MongoDB...")
    
    # ลองโหลด .env ก่อน
    dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path)
        print("✅ โหลดไฟล์ .env สำเร็จ")
    else:
        print("⚠️ ไม่พบไฟล์ .env")
    
    # ตรวจสอบตัวแปรสภาพแวดล้อม
    mongo_url = os.getenv("MONGO_URL")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if not mongo_url:
        print("❌ ไม่พบ MONGO_URL ในตัวแปรสภาพแวดล้อม")
        print("กรุณาสร้างไฟล์ .env และใส่ MongoDB connection string")
        return False
    
    if not openai_key:
        print("❌ ไม่พบ OPENAI_API_KEY ในตัวแปรสภาพแวดล้อม")
        print("กรุณาสร้างไฟล์ .env และใส่ OpenAI API key")
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
        else:
            print("   - ไม่มี collections")
        
        client.close()
        return True
        
    except Exception as e:
        print(f"❌ ไม่สามารถเชื่อมต่อ MongoDB ได้: {e}")
        return False

def run_pipeline():
    """รัน pipeline เพื่อบันทึกข้อมูล"""
    print("\n🚀 กำลังรัน pipeline...")
    
    try:
        # Import และรัน multimodel_rag
        from multimodel_rag import main
        main()
        return True
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดในการรัน pipeline: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 ทดสอบระบบ AstroBot MongoDB Pipeline")
    print("=" * 60)
    
    # ทดสอบการเชื่อมต่อ
    if test_mongodb_connection():
        print("\n" + "=" * 60)
        
        # ถามผู้ใช้ว่าต้องการรัน pipeline หรือไม่
        response = input("ต้องการรัน pipeline เพื่อบันทึกข้อมูลหรือไม่? (y/n): ")
        if response.lower() in ['y', 'yes', 'ใช่']:
            run_pipeline()
        else:
            print("❌ ยกเลิกการรัน pipeline")
    else:
        print("\n❌ ไม่สามารถดำเนินการต่อได้เนื่องจากไม่สามารถเชื่อมต่อ MongoDB")
        print("\n📝 วิธีการแก้ไข:")
        print("1. สร้างไฟล์ .env ในโฟลเดอร์หลัก")
        print("2. ใส่ MongoDB connection string และ OpenAI API key")
        print("3. รันสคริปต์นี้อีกครั้ง")
