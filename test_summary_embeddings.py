#!/usr/bin/env python3
"""
ทดสอบระบบ Summary Embeddings ใหม่
- Summary มี embedding ✅
- Text ต้นฉบับไม่มี embedding ❌
- ใช้ summary embeddings ในการค้นหา
"""

import os
import sys
import json
from datetime import datetime

# เพิ่ม path สำหรับ import modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

def test_summary_embeddings():
    """ทดสอบการสร้าง summary embeddings"""
    print("🧪 ทดสอบระบบ Summary Embeddings ใหม่")
    print("=" * 50)
    
    try:
        from multimodel_rag import create_embeddings, summarize_with_openai
        
        # ทดสอบข้อความตัวอย่าง
        test_text = """
        ราศีเมษ (Aries) เป็นราศีแรกในจักรราศี เริ่มต้นจากวันที่ 21 มีนาคม ถึง 19 เมษายน 
        ราศีเมษเป็นราศีธาตุไฟ มีดาวอังคารเป็นดาวเจ้าเรือน ราศีเมษมีลักษณะเด่นคือ 
        ความกล้าหาญ ความเป็นผู้นำ และความกระตือรือร้น
        """
        
        print("📝 ข้อความต้นฉบับ:")
        print(f"   {test_text.strip()}")
        print()
        
        # สร้าง summary
        print("🤖 สร้าง Summary...")
        summary = summarize_with_openai(test_text, "text")
        print(f"   Summary: {summary}")
        print()
        
        # สร้าง embeddings จาก summary
        print("🔢 สร้าง Embeddings จาก Summary...")
        summary_embeddings = create_embeddings(summary)
        print(f"   Embedding size: {len(summary_embeddings)}")
        print(f"   First 5 values: {summary_embeddings[:5]}")
        print()
        
        # สร้าง embeddings จาก text ต้นฉบับ (เพื่อเปรียบเทียบ)
        print("🔢 สร้าง Embeddings จาก Text ต้นฉบับ (เพื่อเปรียบเทียบ)...")
        text_embeddings = create_embeddings(test_text)
        print(f"   Embedding size: {len(text_embeddings)}")
        print(f"   First 5 values: {text_embeddings[:5]}")
        print()
        
        # เปรียบเทียบ embeddings
        import numpy as np
        similarity = np.dot(summary_embeddings, text_embeddings) / (
            np.linalg.norm(summary_embeddings) * np.linalg.norm(text_embeddings)
        )
        print(f"📊 ความคล้ายคลึงระหว่าง Summary และ Text ต้นฉบับ: {similarity:.4f}")
        print()
        
        # ทดสอบการค้นหา
        print("🔍 ทดสอบการค้นหา...")
        from retrieval_utils import ask_question_to_rag
        
        test_questions = [
            "ราศีเมษมีลักษณะเด่นอย่างไร?",
            "ราศีเมษเป็นราศีธาตุอะไร?",
            "ดาวเจ้าเรือนของราศีเมษคืออะไร?"
        ]
        
        for i, question in enumerate(test_questions, 1):
            print(f"   คำถาม {i}: {question}")
            try:
                answer = ask_question_to_rag(question, "test_user")
                print(f"   คำตอบ: {answer[:200]}...")
            except Exception as e:
                print(f"   ❌ Error: {e}")
            print()
        
        print("✅ การทดสอบเสร็จสิ้น!")
        
    except Exception as e:
        print(f"❌ Error ในการทดสอบ: {e}")
        import traceback
        traceback.print_exc()

def test_database_structure():
    """ทดสอบโครงสร้างฐานข้อมูล"""
    print("\n🗄️ ทดสอบโครงสร้างฐานข้อมูล")
    print("=" * 50)
    
    try:
        from pymongo import MongoClient
        from config import MONGO_URL, SUMMARY_DB_NAME, ORIGINAL_DB_NAME
        
        client = MongoClient(MONGO_URL, serverSelectionTimeoutMS=5000)
        
        # ตรวจสอบ SUMMARY_DB_NAME
        print(f"📊 ตรวจสอบ {SUMMARY_DB_NAME}:")
        summary_db = client[SUMMARY_DB_NAME]
        collections = summary_db.list_collection_names()
        print(f"   Collections: {collections}")
        
        for collection_name in collections:
            collection = summary_db[collection_name]
            count = collection.count_documents({})
            print(f"   {collection_name}: {count} documents")
            
            # ตรวจสอบโครงสร้างข้อมูล
            if count > 0:
                sample_doc = collection.find_one()
                print(f"   Sample fields: {list(sample_doc.keys())}")
                
                # ตรวจสอบว่า embeddings ถูกสร้างจาก summary หรือไม่
                if 'embeddings' in sample_doc and 'summary' in sample_doc:
                    print(f"   ✅ มี summary และ embeddings")
                    print(f"   Summary length: {len(sample_doc['summary'])}")
                    print(f"   Embeddings size: {len(sample_doc['embeddings'])}")
                else:
                    print(f"   ❌ ไม่มี summary หรือ embeddings")
        
        # ตรวจสอบ ORIGINAL_DB_NAME
        print(f"\n📁 ตรวจสอบ {ORIGINAL_DB_NAME}:")
        original_db = client[ORIGINAL_DB_NAME]
        collections = original_db.list_collection_names()
        print(f"   Collections: {collections}")
        
        for collection_name in collections:
            collection = original_db[collection_name]
            count = collection.count_documents({})
            print(f"   {collection_name}: {count} documents")
            
            # ตรวจสอบว่าไม่มี embeddings
            if count > 0:
                sample_doc = collection.find_one()
                if 'embeddings' in sample_doc:
                    print(f"   ❌ มี embeddings (ไม่ควรมี)")
                else:
                    print(f"   ✅ ไม่มี embeddings (ถูกต้อง)")
        
        client.close()
        print("\n✅ การตรวจสอบฐานข้อมูลเสร็จสิ้น!")
        
    except Exception as e:
        print(f"❌ Error ในการตรวจสอบฐานข้อมูล: {e}")

if __name__ == "__main__":
    print("🚀 เริ่มทดสอบระบบ Summary Embeddings ใหม่")
    print(f"⏰ เวลา: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # ทดสอบการสร้าง embeddings
    test_summary_embeddings()
    
    # ทดสอบโครงสร้างฐานข้อมูล
    test_database_structure()
    
    print("\n🎉 การทดสอบทั้งหมดเสร็จสิ้น!")
