import os
import json
import random
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# Load environment variables
load_dotenv()

MONGO_URL = os.getenv("MONGO_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DB_NAME = "astrobot_original"  # Correct DB for content chunks
# Based on multimodel_rag.py
COLLECTIONS = ["original_text_chunks", "original_image_chunks", "original_table_chunks"]

KEYWORDS = ["วันเดือนปีเกิด", "เวลาเกิด", "การงาน", "การเงิน", "ความรัก", "สีมงคล"]
TARGET_COUNT = 10

def get_mongo_client():
    try:
        client = MongoClient(MONGO_URL)
        # Verify connection
        client.admin.command('ping')
        print(f"✅ เชื่อมต่อ MongoDB สำเร็จ: {MONGO_URL.split('@')[-1]}")  # Hide credentials
        return client
    except Exception as e:
        print(f"❌ เชื่อมต่อ MongoDB ล้มเหลว: {e}")
        return None

def fetch_candidate_chunks(client, db_name, collections, keywords, limit_per_keyword=50):
    db = client[db_name]
    candidates = []
    
    print(f"\n🔍 กำลังดึงข้อมูลจาก MongoDB Database: '{db_name}'...")
    
    for collection_name in collections:
        if collection_name not in db.list_collection_names():
            print(f"⚠️ ไม่พบ Collection: {collection_name} - ข้าม")
            continue
            
        collection = db[collection_name]
        doc_count = collection.count_documents({})
        print(f"   📂 Collection '{collection_name}' มีทั้งหมด {doc_count} เอกสาร")
        
        for keyword in keywords:
            # Simple text search regex
            query = {"text": {"$regex": keyword, "$options": "i"}}
            cursor = collection.find(query).limit(limit_per_keyword)
            found_docs = list(cursor)
            
            if found_docs:
                print(f"      - Keyword '{keyword}': พบ {len(found_docs)} เอกสาร")
                for doc in found_docs:
                    # Avoid duplicates if traversing multiple keywords
                    if not any(c['_id'] == doc['_id'] for c in candidates):
                        candidates.append({
                            "_id": str(doc['_id']),
                            "text": doc.get('text', ''),
                            "source": doc.get('source', 'Unknown'),
                            "page": doc.get('page', 'N/A'),
                            "collection": collection_name,
                            "matched_keyword": keyword
                        })
    
    print(f"✅ รวมเอกสารที่เกี่ยวข้องทั้งหมด: {len(candidates)} รายการ")
    
    # Verification: Print the first retrieved document to prove it comes from MongoDB
    if candidates:
        sample = candidates[0]
        print("\n🔎 [Verification] ตัวอย่างข้อมูลที่ดึงมาจาก MongoDB:")
        print(f"   🆔 ID: {sample['_id']}")
        print(f"   📂 Collection: {sample['collection']}")
        print(f"   📄 Source: {sample['source']}")
        print(f"   📝 Text (Snippet): {sample['text'][:200]}...")
        print("--------------------------------------------------\n")
        
    return candidates

def generate_qa_pair(client_openai, context):
    prompt = f"""
    ข้อมูลบริบทอยู่ด้านล่างนี้
    ---------------------
    {context}
    ---------------------
    จากข้อมูลบริบทที่กำหนดให้ (และห้ามใช้ความรู้อื่นนอกเหนือจากบริบท)
    ให้สร้าง "คำถาม" และ "คำตอบ" จำนวน 1 คู่
    
    ข้อกำหนดสำคัญ (Critical Requirement):
    1. **ต้อง** สมมติ "วันเดือนปีเกิด" ใส่ลงในคำถามในรูปแบบ **DD/MM/YYYY** (เช่น "คนเกิดวันที่ 15/04/1990...") เพื่อจำลองผู้ใช้งานจริง
    2. หากบริบทกล่าวถึง "ราศี" ใด ให้เลือกวันเกิดที่อยู่ในช่วงราศีนั้น (เช่น พฤษภ -> 15/05/xxxx) เพื่อความสมเหตุสมผล
    3. **คำตอบ (Answer)** ต้องสร้างจาก **ข้อมูลในบริบทเท่านั้น** (ห้ามนำความรู้นอกเหนือจากบริบทมาตอบ แม้ความรู้นั้นจะถูกตามหลักโหราศาสตร์ก็ตาม) เพื่อให้การวัดผล Faithfulness ได้คะแนนเต็ม
    4. หากบริบทเป็นเรื่องทั่วไป (เช่น ความหมายดาว) ให้ตั้งคำถามโดยสมมติวันเกิด ใส่ลงไปประกอบฉาก แต่คำตอบยังคงอิงตามเนื้อหา
    
    รูปแบบภาษา:
    - คำถาม: ภาษาไทย มีวันเกิดระบุชัดเจน
    - คำตอบ: ภาษาไทย ตอบตามเนื้อหาในบริบทอย่างเคร่งครัด
    
    Format the output as JSON:
    {{
        "question": "คำถามภาษาไทยที่ระบุวันเวลาเกิด",
        "answer": "คำตอบภาษาไทย"
    }}
    """
    
    try:
        response = client_openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates Q&A pairs from text."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error generating Q&A: {e}")
        return None

def main():
    if not MONGO_URL:
        print("❌ Error: MONGO_URL not found in environment variables.")
        return
    if not OPENAI_API_KEY:
        print("❌ Error: OPENAI_API_KEY not found in environment variables.")
        return

    mongo_client = get_mongo_client()
    if not mongo_client:
        return

    # 1. Fetch Data from MongoDB
    print("\n--- ขั้นตอนการดึงข้อมูล ---")
    candidates = fetch_candidate_chunks(mongo_client, DB_NAME, COLLECTIONS, KEYWORDS)
    
    if len(candidates) < TARGET_COUNT:
        print(f"⚠️ พบเอกสารเพียง {len(candidates)} รายการ ซึ่งน้อยกว่าเป้าหมาย {TARGET_COUNT} ข้อ")
        print("   ระบบจะใช้เอกสารที่มีทั้งหมด (อาจมีซ้ำหากจำเป็นต้องเพิ่มจำนวน)")
    
    # Select samples (ensure diversity if possible, or just take random allowed)
    # If we have enough, sample w/o replacement. If not, we might need to reuse or just clamp.
    if len(candidates) >= TARGET_COUNT:
        selected_chunks = random.sample(candidates, TARGET_COUNT)
    else:
        selected_chunks = candidates # Take all
    
    print(f"\n--- ขั้นตอนการสร้าง Q&A ด้วย LLM ({len(selected_chunks)} รายการ) ---")
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    
    dataset = []
    
    for i, chunk in enumerate(tqdm(selected_chunks, desc="Generating")):
        try:
            qa = generate_qa_pair(openai_client, chunk['text'])
            if qa:
                dataset.append({
                    "question": qa['question'],
                    "ground_truth": qa['answer'], # Ragas expects 'ground_truth' or 'ground_truths' usually, but for simple viewing 'answer' is fine. We will use 'answer' for CSV
                    "answer": qa['answer'],      # Keeping 'answer' for clear CSV reading
                    "context": chunk['text'],    # Ragas expects 'contexts' as list of strings
                    "source_page": chunk['page'],
                    "collection": chunk['collection'],
                    "keyword": chunk['matched_keyword']
                })
        except Exception as e:
            print(f"Skipping chunk {i}: {e}")

    # Output to files
    df = pd.DataFrame(dataset)
    
    # Create final JSON structure for Ragas (if we were using the HF dataset loader directly, but simple JSON/CSV is fine for our custom evaluation script)
    # We will save as simple records
    
    print("\n--- บันทึกผลลัพธ์ ---")
    print(f"📊 ได้ชุดข้อมูลจำนวน: {len(df)} ข้อ")
    
    csv_path = "generated_dataset.csv"
    json_path = "generated_dataset.json"
    
    df.to_csv(csv_path, index=False, encoding='utf-8-sig') # utf-8-sig for Excel Thai support
    df.to_csv(csv_path, index=False, encoding='utf-8-sig') # utf-8-sig for Excel Thai support
    
    # Use json.dump for cleaner formatting (no escaped /)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print(f"✅ บันทึกไฟล์ CSV ที่: {os.path.abspath(csv_path)}")
    print(f"✅ บันทึกไฟล์ JSON ที่: {os.path.abspath(json_path)}")

if __name__ == "__main__":
    main()
