#!/usr/bin/env python3
"""
วิเคราะห์ขนาดการจัดเก็บข้อมูลระหว่าง original และ summary
"""

import os
import sys
from pymongo import MongoClient
from dotenv import load_dotenv

# โหลด .env
load_dotenv()

def analyze_storage():
    """วิเคราะห์ขนาดการจัดเก็บข้อมูล"""
    print("🔍 วิเคราะห์ขนาดการจัดเก็บข้อมูล")
    print("=" * 50)
    
    try:
        mongo_uri = os.getenv("MONGO_URL")
        if not mongo_uri or mongo_uri == "mongodb+srv://your-username:your-password@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority":
            print("❌ MONGO_URL not configured properly")
            return
        
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        
        # ตรวจสอบ astrobot_original
        print("\n📁 astrobot_original (ข้อมูลต้นฉบับ):")
        original_db = client["astrobot_original"]
        original_collections = original_db.list_collection_names()
        
        original_total_size = 0
        for collection_name in original_collections:
            collection = original_db[collection_name]
            count = collection.count_documents({})
            
            # คำนวณขนาดโดยประมาณ
            sample_doc = collection.find_one()
            if sample_doc:
                doc_size = len(str(sample_doc))
                total_size = doc_size * count
                original_total_size += total_size
                
                print(f"   {collection_name}: {count} docs, ~{total_size/1024:.1f}KB")
                
                # ตรวจสอบ fields
                print(f"      Fields: {list(sample_doc.keys())}")
                if 'text' in sample_doc:
                    print(f"      Text length: {len(sample_doc['text'])} chars")
        
        print(f"   📊 Total estimated size: ~{original_total_size/1024:.1f}KB")
        
        # ตรวจสอบ astrobot_summary
        print("\n📊 astrobot_summary (ข้อมูลที่ประมวลผลแล้ว):")
        summary_db = client["astrobot_summary"]
        summary_collections = summary_db.list_collection_names()
        
        summary_total_size = 0
        for collection_name in summary_collections:
            collection = summary_db[collection_name]
            count = collection.count_documents({})
            
            # คำนวณขนาดโดยประมาณ
            sample_doc = collection.find_one()
            if sample_doc:
                doc_size = len(str(sample_doc))
                total_size = doc_size * count
                summary_total_size += total_size
                
                print(f"   {collection_name}: {count} docs, ~{total_size/1024:.1f}KB")
                
                # ตรวจสอบ fields
                print(f"      Fields: {list(sample_doc.keys())}")
                if 'text' in sample_doc:
                    print(f"      Text length: {len(sample_doc['text'])} chars")
                if 'summary' in sample_doc:
                    print(f"      Summary length: {len(sample_doc['summary'])} chars")
                if 'embeddings' in sample_doc:
                    print(f"      Embeddings size: {len(sample_doc['embeddings'])} dimensions")
        
        print(f"   📊 Total estimated size: ~{summary_total_size/1024:.1f}KB")
        
        # เปรียบเทียบ
        print(f"\n📈 การเปรียบเทียบ:")
        print(f"   Original: ~{original_total_size/1024:.1f}KB")
        print(f"   Summary:  ~{summary_total_size/1024:.1f}KB")
        print(f"   Difference: {summary_total_size - original_total_size:.1f}KB")
        
        if summary_total_size < original_total_size:
            print("   ❌ Summary ควรใหญ่กว่า Original (เพราะมี embeddings)")
        else:
            print("   ✅ Summary ใหญ่กว่า Original (ถูกต้อง)")
        
        # วิเคราะห์รายละเอียด
        print(f"\n🔍 วิเคราะห์รายละเอียด:")
        if summary_collections:
            sample_doc = summary_db[summary_collections[0]].find_one()
            if sample_doc:
                text_size = len(sample_doc.get('text', ''))
                summary_size = len(sample_doc.get('summary', ''))
                embeddings_size = len(sample_doc.get('embeddings', [])) * 8  # 8 bytes per float64
                
                print(f"   Text size: {text_size} chars")
                print(f"   Summary size: {summary_size} chars")
                print(f"   Embeddings size: {embeddings_size} bytes")
                print(f"   Summary/Text ratio: {summary_size/text_size:.2f}")
                print(f"   Embeddings overhead: {embeddings_size/1024:.1f}KB per doc")
        
        client.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_storage()
