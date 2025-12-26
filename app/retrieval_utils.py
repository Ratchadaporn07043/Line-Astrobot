import os
import re
import logging
from datetime import datetime, timedelta, time as dt_time
from typing import Tuple
from pymongo import MongoClient
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from .birth_date_parser import generate_astrology_reading, generate_detailed_astrology_reading, extract_birth_info_from_message

# แก้ไขปัญหา MPS device - ใช้ CPU แทน
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# โหลด environment variables

# โหลด environment variables
load_dotenv()

# ============================
# 🆕 Constants for Entity Filtering
# ============================
ASTRO_SYSTEM_ENTITIES = {
    # ดาวเคราะห์ (Planets)
    "sun": ["อาทิตย์", "sun", "apollon"],
    "moon": ["จันทร์", "moon", "luna"],
    "mercury": ["พุธ", "mercury", "hermes"],
    "venus": ["ศุกร์", "venus", "aphrodite"],
    "mars": ["อังคาร", "mars", "ares"],
    "jupiter": ["พฤหัส", "พฤหัสบดี", "jupiter", "zeus"],
    "saturn": ["เสาร์", "saturn", "kronos"],
    "uranus": ["มฤตยู", "ยูเรนัส", "uranus"],
    "neptune": ["เนปจูน", "neptune", "poseidon"],
    "pluto": ["พลูโต", "pluto", "hades"],
    "rahu": ["ราหู", "node", "north node"],
    "ketu": ["เกตุ", "south node"],
    
    # ราศี (Zodiacs)
    "aries": ["เมษ", "aries"],
    "taurus": ["พฤษภ", "taurus"],
    "gemini": ["มิถุน", "เมถุน", "gemini"],
    "cancer": ["กรกฎ", "cancer"],
    "leo": ["สิงห์", "leo"],
    "virgo": ["กันย์", "virgo"],
    "libra": ["ตุลย์", "libra"],
    "scorpio": ["พิจิก", "scorpio"],
    "sagittarius": ["ธนู", "sagittarius"],
    "capricorn": ["มังกร", "capricorn"],
    "aquarius": ["กุมภ์", "aquarius"],
    "pisces": ["มีน", "pisces"]
}

NOISE_KEYWORDS = ["pottery", "ceramic", "clay", "vessel", "sherd", "kiln", "excavation"]  # คำที่มักเจอในเอกสารขยะ

# Helper function to extract entities
def extract_astro_entities(text: str) -> dict:
    """
    แยกแยะชื่อดาวและราศีจากข้อความ
    Returns: {'planets': [list of keys], 'zodiacs': [list of keys]}
    """
    text_lower = text.lower()
    found = {'planets': [], 'zodiacs': []}
    
    for key, keywords in ASTRO_SYSTEM_ENTITIES.items():
        for kw in keywords:
            if kw in text_lower:
                # แยกประเภทว่าเป็นดาวหรือราศี (ง่ายๆ ด้วยการเช็ค key)
                if key in ["aries", "taurus", "gemini", "cancer", "leo", "virgo", "libra", "scorpio", "sagittarius", "capricorn", "aquarius", "pisces"]:
                    if key not in found['zodiacs']:
                        found['zodiacs'].append(key)
                else:
                    if key not in found['planets']:
                        found['planets'].append(key)
                break
    return found


# ตั้งค่า Logger
logger = logging.getLogger(__name__)

# Import database configuration
from config import ORIGINAL_DB_NAME

# ============================
# ⚠️ ระบบ RAG: ใช้ข้อมูลจาก MongoDB ต้นฉบับเท่านั้น
# ============================
# ระบบ RAG นี้ใช้ข้อมูลจาก ORIGINAL_DB_NAME (astrobot_original) เท่านั้น
# - Collections: original_text_chunks, original_image_chunks, original_table_chunks
# - ใช้ field 'text' จากเอกสารต้นฉบับ
# - ใช้ embeddings ที่สร้างจาก text ต้นฉบับ
# - ไม่ใช้ summary หรือข้อมูลที่ประมวลผลแล้ว
# ============================

# ============================
# MongoDB Connection Verification
# ============================
def verify_mongodb_connection_for_retrieval() -> Tuple[bool, str, dict]:
    """
    ตรวจสอบการเชื่อมต่อ MongoDB และเตรียมพร้อมสำหรับ retrieval
    
    Returns:
        tuple: (is_ready, message, connection_info)
            - is_ready: True ถ้า MongoDB พร้อมใช้งานสำหรับ retrieval
            - message: ข้อความสรุปผลการตรวจสอบ
            - connection_info: ข้อมูลการเชื่อมต่อ (client, db, collections)
    """
    connection_info = {
        'client': None,
        'db': None,
        'collections': {}
    }
    
    # ตรวจสอบ MONGO_URL
    mongo_uri = os.getenv("MONGO_URL")
    if not mongo_uri or mongo_uri == "mongodb+srv://your-username:your-password@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority":
        return False, "MONGO_URL ไม่ได้ตั้งค่าหรือยังเป็นค่า default", connection_info
    
    try:
        # เชื่อมต่อ MongoDB
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=10000, connectTimeoutMS=10000)
        
        # ทดสอบการเชื่อมต่อ
        try:
            client.server_info()  # จะ raise exception ถ้าเชื่อมต่อไม่ได้
        except Exception as conn_err:
            client.close()
            return False, f"ไม่สามารถเชื่อมต่อ MongoDB ได้: {conn_err}", connection_info
        
        # ตรวจสอบว่า database มีอยู่หรือไม่
        db_names = client.list_database_names()
        if ORIGINAL_DB_NAME not in db_names:
            client.close()
            return False, f"Database '{ORIGINAL_DB_NAME}' ไม่มีอยู่", connection_info
        
        # ตรวจสอบ collections ที่จำเป็น
        db = client[ORIGINAL_DB_NAME]
        collection_names = db.list_collection_names()
        
        required_collections = [
            "original_text_chunks",
            "original_image_chunks",
            "original_table_chunks"
        ]
        
        collections_status = {}
        total_docs = 0
        all_collections_exist = True
        all_collections_have_data = True
        
        for collection_name in required_collections:
            if collection_name not in collection_names:
                collections_status[collection_name] = {
                    'exists': False,
                    'doc_count': 0,
                    'has_embeddings': False
                }
                all_collections_exist = False
                all_collections_have_data = False
            else:
                collection = db[collection_name]
                doc_count = collection.count_documents({})
                total_docs += doc_count
                
                # ตรวจสอบว่ามี embeddings หรือไม่
                has_embeddings = False
                if doc_count > 0:
                    sample_doc = collection.find_one()
                    if sample_doc and 'embeddings' in sample_doc:
                        emb = sample_doc['embeddings']
                        if isinstance(emb, (list, tuple)) and len(emb) > 0:
                            has_embeddings = True
                
                collections_status[collection_name] = {
                    'exists': True,
                    'doc_count': doc_count,
                    'has_embeddings': has_embeddings
                }
                
                if doc_count == 0:
                    all_collections_have_data = False
        
        connection_info['client'] = client
        connection_info['db'] = db
        connection_info['collections'] = collections_status
        
        # สรุปผลการตรวจสอบ
        if not all_collections_exist:
            message = f"บาง collections ไม่มีอยู่ (ต้องมี: {', '.join(required_collections)})"
            return False, message, connection_info
        
        if total_docs == 0:
            message = f"Database '{ORIGINAL_DB_NAME}' มี collections แต่ไม่มีข้อมูล (0 เอกสาร)"
            return False, message, connection_info
        
        if not all_collections_have_data:
            empty_collections = [name for name, status in collections_status.items() 
                               if status['doc_count'] == 0]
            message = f"บาง collections ว่างเปล่า: {', '.join(empty_collections)}"
            # ยังคง return True เพราะมีบาง collections มีข้อมูล
        
        message = f"✅ MongoDB พร้อมใช้งาน: พบ {total_docs} เอกสารใน {len([s for s in collections_status.values() if s['doc_count'] > 0])} collections"
        return True, message, connection_info
        
    except Exception as e:
        logger.error(f"Error verifying MongoDB connection: {e}")
        if connection_info.get('client'):
            try:
                connection_info['client'].close()
            except:
                pass
        return False, f"เกิดข้อผิดพลาดในการตรวจสอบ MongoDB: {e}", connection_info

# ============================
# Answer Source Verification
# ============================
def verify_answer_source(answer: str, retrieved_docs: list, question: str) -> bool:
    """
    ตรวจสอบว่าคำตอบมาจาก MongoDB เท่านั้นหรือไม่
    
    Args:
        answer: คำตอบที่ได้จาก GPT
        retrieved_docs: เอกสารที่ retrieve จาก MongoDB
        question: คำถามที่ถาม
        
    Returns:
        bool: True ถ้าคำตอบน่าจะมาจาก MongoDB, False ถ้าไม่แน่ใจ
    """
    if not answer or not retrieved_docs:
        return False
    
    # ตรวจสอบว่าคำตอบมีวลีที่บอกว่าไม่มีข้อมูลในฐานข้อมูล
    no_data_phrases = [
        "ไม่พบข้อมูล",
        "ไม่มีข้อมูล",
        "ขออภัย",
        "ไม่สามารถ",
        "ไม่มีข้อมูลในฐานข้อมูล"
    ]
    
    # ถ้าคำตอบบอกว่าไม่มีข้อมูล แสดงว่าใช้ข้อมูลจาก MongoDB (แต่ไม่มีข้อมูล)
    if any(phrase in answer for phrase in no_data_phrases):
        return True
    
    # ตรวจสอบว่ามีข้อมูลจาก MongoDB ที่สามารถใช้ตอบคำถามได้
    if not retrieved_docs or len(retrieved_docs) == 0:
        return False
    
    # ตรวจสอบว่าคำตอบมีเนื้อหาที่เกี่ยวข้องกับข้อมูลที่ retrieve มา
    # โดยตรวจสอบว่ามีคำสำคัญจาก retrieved_docs ปรากฏในคำตอบ
    answer_lower = answer.lower()
    
    # สร้างชุดคำสำคัญจาก retrieved_docs
    key_phrases = set()
    for doc in retrieved_docs[:3]:  # ตรวจสอบเฉพาะ 3 เอกสารแรก
        if isinstance(doc, dict):
            content = doc.get('text', '')
            if content:
                # แยกคำสำคัญ (คำที่มีความยาวมากกว่า 3 ตัวอักษร)
                words = content.lower().split()
                key_phrases.update([w for w in words if len(w) > 3])
    
    # ตรวจสอบว่าคำตอบมีคำสำคัญจาก MongoDB หรือไม่
    if key_phrases:
        matches = sum(1 for phrase in key_phrases if phrase in answer_lower)
        # ถ้ามีคำสำคัญจาก MongoDB ปรากฏในคำตอบมากกว่า 10% ถือว่าใช้ข้อมูลจาก MongoDB
        match_ratio = matches / len(key_phrases) if key_phrases else 0
        return match_ratio > 0.1
    
    return True  # ถ้าไม่มีข้อมูลให้ตรวจสอบ ถือว่าใช้ข้อมูลจาก MongoDB

# ============================
# Pretty Terminal Reporting
# ============================
def _print_divider(title: str):
    print(f"\n== {title} ==")


def print_ragas_terminal_report(
    question: str,
    retrieved_docs: list,
    answer: str,
    user_id: str = "unknown",
):
    """
    แสดงผลสรุปบนเทอร์มินัลในรูปแบบอ่านง่าย เพื่อใช้ประกอบการประเมินด้วย RAGAS
    - สรุปผลการค้นหาและจำนวนเอกสาร
    - แหล่งที่มาพร้อม Similarity (ถ้ามี)
    - ความยาวคำตอบจาก GPT
    """
    try:
        # ตรวจสอบเอกสารที่มี similarity ต่ำเกินไปเพื่อแสดง warning
        low_similarity_docs = []
        valid_docs = []
        
        for doc in retrieved_docs:
            if isinstance(doc, dict) and doc.get('below_threshold', False):
                low_similarity_docs.append(doc)
            else:
                valid_docs.append(doc)
        
        # แสดง warning สำหรับเอกสารที่ต่ำกว่า threshold
        if low_similarity_docs:
            for idx, doc in enumerate(low_similarity_docs):
                sim = doc.get("similarity", 0)
                doc_num = len(valid_docs) + idx + 1
                print(f"! เอกสารที่ {doc_num} มี similarity ต่ำเกินไป: {sim:.4f}")
        
        # สรุปผลการค้นหา
        print("\n=== สรุปผลการค้นหา ===")
        total_found = len(valid_docs) if isinstance(valid_docs, list) else 0
        print(f"เอกสารที่พบทั้งหมด : {total_found} เอกสาร")
        if total_found > 0:
            print("✔ พบข้อมูลที่เกี่ยวข้อง สามารถใช้ RAG ได้")
        else:
            print("ไม่พบข้อมูลที่เกี่ยวข้อง -> ไม่สามารถใช้ข้อมูลจาก MongoDB ได้")
        print("==== เสร็จสิ้นการค้นหา ===\n")

        # แสดงข้อมูลที่ใช้จากฐานข้อมูล
        if total_found > 0:
            print(f"🗄️ ใช้ข้อมูลจากฐานข้อมูล: {total_found} เอกสาร")
            print("💬 กำลังส่งคำถามไปยัง GPT...")
        
        # GPT Response (แสดงแค่ความยาว ไม่แสดงคำตอบ)
        ans_len = len(answer) if isinstance(answer, str) else 0
        if ans_len > 0:
            print(f"✔ ได้รับค่าตอบจาก GPT (ความยาว: {ans_len} ตัวอักษร)\n")

        # สรุปแหล่งที่มาของข้อมูล - แสดงเฉพาะเอกสารที่มี similarity > 0.5
        if total_found:
            print("=== สรุปแหล่งที่มาของข้อมูล (แสดงเฉพาะเอกสารที่มี Similarity > 0.5) ===")
            high_similarity_docs = []
            for doc in valid_docs:
                if isinstance(doc, dict):
                    sim = doc.get("similarity", 0)
                    if sim > 0.5:
                        high_similarity_docs.append(doc)
            
            if high_similarity_docs:
                for i, doc in enumerate(high_similarity_docs, 1):
                    try:
                        source = doc.get("source", "Unknown source")
                        sim = doc.get("similarity", 0)
                        text_content = doc.get("text", "")
                        
                        # กำหนด emoji ตามประเภทของเอกสาร
                        collection = doc.get("collection", "")
                        if "image" in collection:
                            emoji = "🖼️"
                        elif "table" in collection:
                            emoji = "📊"
                        else:
                            emoji = "📄"
                        
                        # แสดงข้อมูลเอกสาร
                        print(f"{emoji} เอกสารที่ {i}: {source} (Similarity: {sim:.4f})")
                        
                        # แสดง context (ข้อความที่ใช้) - แสดงทั้งหมดไม่จำกัดความยาว
                        if text_content:
                            print(f"   📝 Context: {text_content}")
                        print()  # เว้นบรรทัดระหว่างเอกสาร
                    except Exception as e:
                        print(f"❓ เอกสารที่ {i}: ไม่สามารถแสดงรายละเอียดได้ - {e}")
                
                print(f"=== เสร็จสิ้นการสรุปแหล่งที่มา (แสดง {len(high_similarity_docs)} เอกสารจาก {total_found} เอกสารทั้งหมด) ===\n")
            else:
                print("⚠️ ไม่มีเอกสารที่มี Similarity > 0.5")
                print("=== เสร็จสิ้นการสรุปแหล่งที่มา ===\n")

    except Exception:
        # อย่าทำให้ flow ล้ม หากมีปัญหาในการพิมพ์ report
        pass



# ✔️ บันทึกคำตอบที่ใช้ตอบผู้ใช้ใน collection responses (ไม่เก็บคำถาม)
def store_user_response(
    question: str,
    answer: str,
    user_id: str = "unknown",
    response_type: str = "rag_response",
    context_data: dict = None
):
    """
    บันทึกคำตอบที่ใช้ตอบผู้ใช้ใน collection responses (ไม่เก็บคำถาม)
    และอัปเดตข้อมูลใน user_profiles สำหรับการถามคำถามต่อเนื่อง
    
    🆕 สร้าง embeddings สำหรับ question และ answer เพื่อใช้ในการทำ Semantic Similarity 
    สำหรับ follow-up detection
    
    Args:
        question (str): คำถามของผู้ใช้ (ใช้สำหรับอัปเดต user_profiles และสร้าง embedding)
        answer (str): คำตอบที่ส่งให้ผู้ใช้
        user_id (str): ID ของผู้ใช้
        response_type (str): ประเภทของคำตอบ (rag_response, birth_chart, etc.)
        context_data (dict): ข้อมูลบริบทเพิ่มเติม
    """
    try:
        mongo_uri = os.getenv("MONGO_URL")
        if not mongo_uri or mongo_uri == "mongodb+srv://your-username:your-password@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority":
            logger.warning("MONGO_URL not configured properly, skipping response storage")
            return
        
        logger.info(f"🔄 Attempting to store response for user {user_id}, type: {response_type}")
        
        # 🆕 สร้าง embeddings สำหรับ question และ answer เพื่อใช้ใน Semantic Similarity
        try:
            model = SentenceTransformer("minishlab/potion-multilingual-128M", device="cpu")
            question_embedding = model.encode(question, convert_to_numpy=True).tolist()
            answer_embedding = model.encode(answer, convert_to_numpy=True).tolist()
            logger.debug(f"✅ Created embeddings for question and answer (dim: {len(question_embedding)})")
        except Exception as e:
            logger.warning(f"⚠️ Failed to create embeddings: {e}")
            question_embedding = None
            answer_embedding = None
        
        mongo_client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000, connectTimeoutMS=5000)
        responses_collection = mongo_client["astrobot"]["responses"]
        profiles_collection = mongo_client["astrobot"]["user_profiles"]
        
        # สร้างข้อมูลสำหรับบันทึกใน responses (ไม่เก็บคำถาม แต่เก็บ embedding)
        response_data = {
            "user_id": user_id,
            "answer": answer,
            "response_type": response_type,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        # 🆕 เพิ่ม embeddings ถ้าสร้างสำเร็จ
        if question_embedding is not None:
            response_data["question_embedding"] = question_embedding
        if answer_embedding is not None:
            response_data["answer_embedding"] = answer_embedding
        
        # เพิ่มข้อมูลบริบทถ้ามี
        if context_data:
            response_data.update(context_data)
        
        # บันทึกลง collection responses
        result = responses_collection.insert_one(response_data)
        logger.info(f"✅ Successfully stored response in astrobot.responses: {result.inserted_id}")
        
        # อัปเดตข้อมูลใน user_profiles สำหรับการถามคำถามต่อเนื่อง
        profile_update_data = {
            "user_id": user_id,
            "last_question": question,
            "last_response": answer,
            "last_response_type": response_type,
            "updated_at": datetime.utcnow()
        }
        
        # เพิ่มข้อมูลบริบทในโปรไฟล์ถ้ามี
        if context_data:
            # เก็บข้อมูลสำคัญสำหรับการถามคำถามต่อเนื่อง
            if "zodiac_sign" in context_data:
                profile_update_data["zodiac_sign"] = context_data["zodiac_sign"]
            if "zodiac_element" in context_data:
                profile_update_data["zodiac_element"] = context_data["zodiac_element"]
            if "birth_date" in context_data:
                profile_update_data["birth_date"] = context_data["birth_date"]
            if "birth_time" in context_data:
                profile_update_data["birth_time"] = context_data["birth_time"]
        
        # อัปเดตหรือสร้างโปรไฟล์ใหม่
        profiles_collection.update_one(
            {"user_id": user_id},
            {"$set": profile_update_data},
            upsert=True
        )
        
        logger.info(f"📊 Response data: user_id={user_id}, type={response_type}, question_length={len(question)}, answer_length={len(answer)}")
        logger.info(f"🔄 Updated user profile for context management")
        
        mongo_client.close()
        
    except Exception as e:
        logger.error(f"❌ Failed to store response in astrobot.responses: {e}")
        logger.error(f"📝 Error details - user_id: {user_id}, response_type: {response_type}")
        import traceback
        logger.error(f"🔍 Full traceback: {traceback.format_exc()}")

# ✔️ บันทึกคำถามของผู้ใช้ใน user_profiles collection
def store_user_question(
    question: str,
    user_id: str = "unknown",
    context_data: dict = None
):
    # ปิดการบันทึกคำถามลง MongoDB (no-op) เพื่อไม่เก็บ user_profiles ใดๆ
    return

# ✔️ บันทึกหรืออัปเดต user_profiles พร้อมบริบทการสนทนา
def log_user_interaction(
    question: str,
    answer: str,
    embedding: list,
    user_id: str = "unknown",
    context_data: dict = None
):
    # ปิดการบันทึก/อัปเดตโปรไฟล์ (no-op)
    return

# ดึงข้อมูลวันเกิดของผู้ใช้
def get_user_birth_date(user_id: str):
    try:
        # print(f"กำลังค้นหาข้อมูลวันเกิดสำหรับ User ID: {user_id}")
        mongo_uri = os.getenv("MONGO_URL")
        if not mongo_uri or mongo_uri == "mongodb+srv://your-username:your-password@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority":
            # print("MONGO_URL not configured properly. Please set up your .env file with valid MongoDB connection string.")
            return None
        mongo_client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000, connectTimeoutMS=5000)
        collection = mongo_client[ORIGINAL_DB_NAME]["user_profiles"]
        
        user_profile = collection.find_one({"user_id": user_id})
        if user_profile and "birth_date" in user_profile:
            birth_date = user_profile["birth_date"]
            # print(f"พบข้อมูลวันเกิด: {birth_date}")
            return birth_date
        else:
            # print(f"ไม่พบข้อมูลวันเกิดสำหรับ User ID: {user_id}")
            return None
    except Exception as e:
        # print(f"ไม่สามารถดึงข้อมูลวันเกิดได้: {e}")
        return None

# ดึงข้อมูลบริบทการสนทนาของผู้ใช้
def get_user_context(user_id: str):
    """
    ดึงข้อมูลบริบทการสนทนาของผู้ใช้ รวมถึงราศีและข้อมูลอื่นๆ
    จากทั้ง user_profiles และ responses collections
    
    Args:
        user_id (str): ID ของผู้ใช้
        
    Returns:
        dict: ข้อมูลบริบทการสนทนา
    """
    try:
        mongo_uri = os.getenv("MONGO_URL")
        if not mongo_uri or mongo_uri == "mongodb+srv://your-username:your-password@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority":
            return None
            
        mongo_client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000, connectTimeoutMS=5000)
        profiles_collection = mongo_client["astrobot"]["user_profiles"]
        responses_collection = mongo_client["astrobot"]["responses"]
        
        # ดึงข้อมูลจาก user_profiles
        user_profile = profiles_collection.find_one({"user_id": user_id})
        
        # ดึงข้อมูลการสนทนาล่าสุดจาก responses
        latest_response = responses_collection.find_one(
            {"user_id": user_id},
            sort=[("created_at", -1)]
        )
        
        # ดึงข้อมูลการสนทนาทั้งหมดของผู้ใช้ (สำหรับการวิเคราะห์บริบท)
        all_responses = list(responses_collection.find(
            {"user_id": user_id},
            sort=[("created_at", -1)],
            limit=5  # เอาแค่ 5 การสนทนาล่าสุด
        ))
        
        context = {}
        
        # ข้อมูลจาก user_profiles
        if user_profile:
            context.update({
                "birth_date": user_profile.get("birth_date"),
                "zodiac_sign": user_profile.get("zodiac_sign"),
                "zodiac_element": user_profile.get("zodiac_element"),
                "zodiac_quality": user_profile.get("zodiac_quality"),
                "birth_time": user_profile.get("birth_time"),
                "daily_question_count": user_profile.get("daily_question_count", 0),
                "last_question_date": user_profile.get("last_question_date"),
                "updated_at": user_profile.get("updated_at"),
                "last_question": user_profile.get("last_question"),  # ดึงจาก user_profiles
                "last_response": user_profile.get("last_response"),  # ดึงจาก user_profiles
                "last_response_type": user_profile.get("last_response_type")
            })
        
        # ข้อมูลจาก responses collection (ใช้เป็น fallback หรือข้อมูลเพิ่มเติม)
        if latest_response:
            # ถ้ายังไม่มี last_question หรือ last_response ให้ใช้จาก responses
            if not context.get("last_question") and latest_response.get("question"):
                context["last_question"] = latest_response.get("question")
            if not context.get("last_response") and latest_response.get("answer"):
                context["last_response"] = latest_response.get("answer")
            if not context.get("last_response_type") and latest_response.get("response_type"):
                context["last_response_type"] = latest_response.get("response_type")
            context["last_response_time"] = latest_response.get("created_at")
            # 🆕 เก็บ response object ไว้เพื่อใช้ embeddings (ถ้ามี)
            context["_last_response_obj"] = latest_response
        
        # ข้อมูลการสนทนาหลายครั้งล่าสุด
        if all_responses:
            context["recent_conversations"] = []
            for response in all_responses:
                context["recent_conversations"].append({
                    "question": response.get("question"),
                    "answer": response.get("answer"),
                    "response_type": response.get("response_type"),
                    "created_at": response.get("created_at"),
                    "context_data": response.get("context_data", {})
                })
            
            # เพิ่มข้อมูลการสนทนาล่าสุดสำหรับการตอบคำถามต่อเนื่อง
            if len(all_responses) >= 1:
                context["last_conversation"] = {
                    "question": all_responses[0].get("question"),
                    "answer": all_responses[0].get("answer"),
                    "response_type": all_responses[0].get("response_type"),
                    "created_at": all_responses[0].get("created_at")
                }
            
            # เพิ่มข้อมูลการสนทนาก่อนหน้าสำหรับบริบทเพิ่มเติม
            if len(all_responses) >= 2:
                context["previous_conversation"] = {
                    "question": all_responses[1].get("question"),
                    "answer": all_responses[1].get("answer"),
                    "response_type": all_responses[1].get("response_type"),
                    "created_at": all_responses[1].get("created_at")
                }
        
        # วิเคราะห์ข้อมูลราศีจาก context_data ใน responses
        zodiac_info = None
        for response in all_responses:
            context_data = response.get("context_data", {})
            if context_data.get("zodiac_sign"):
                zodiac_info = {
                    "zodiac_sign": context_data.get("zodiac_sign"),
                    "zodiac_element": context_data.get("zodiac_element"),
                    "birth_date": context_data.get("birth_date"),
                    "birth_time": context_data.get("birth_time")
                }
                break
        
        if zodiac_info:
            context.update(zodiac_info)
        
        # print(f"ดึงข้อมูลบริบทสำเร็จ: {context}")
        return context if context else None
        
    except Exception as e:
        # print(f"ไม่สามารถดึงข้อมูลบริบทได้: {e}")
        return None

# ดึงข้อมูลการสนทนาจาก collection responses และ user_profiles
def get_user_conversation_history(user_id: str, limit: int = 10):
    """
    ดึงประวัติการสนทนาของผู้ใช้จาก collection responses และ user_profiles
    
    Args:
        user_id (str): ID ของผู้ใช้
        limit (int): จำนวนการสนทนาที่ต้องการดึง
        
    Returns:
        list: รายการการสนทนาล่าสุด
    """
    try:
        mongo_uri = os.getenv("MONGO_URL")
        if not mongo_uri or mongo_uri == "mongodb+srv://your-username:your-password@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority":
            return []
            
        mongo_client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000, connectTimeoutMS=5000)
        responses_collection = mongo_client["astrobot"]["responses"]
        profiles_collection = mongo_client["astrobot"]["user_profiles"]
        
        # ดึงข้อมูลคำตอบล่าสุดจาก responses
        responses = list(responses_collection.find(
            {"user_id": user_id},
            sort=[("created_at", -1)],
            limit=limit
        ))
        
        # ดึงข้อมูลคำถามล่าสุดจาก user_profiles
        user_profile = profiles_collection.find_one({"user_id": user_id})
        
        # จัดรูปแบบข้อมูล
        formatted_conversations = []
        for response in responses:
            # หาคำถามที่เกี่ยวข้องจาก user_profiles
            question = None
            if user_profile and "last_question" in user_profile:
                question = user_profile.get("last_question")
            
            formatted_conversations.append({
                "question": question,
                "answer": response.get("answer"),
                "response_type": response.get("response_type"),
                "created_at": response.get("created_at"),
                "context_data": response.get("context_data", {})
            })
        
        mongo_client.close()
        return formatted_conversations
        
    except Exception as e:
        logger.error(f"ไม่สามารถดึงประวัติการสนทนาได้: {e}")
        return []

# ตรวจสอบและอัปเดตจำนวนคำถามต่อวัน
def check_and_update_question_limit(user_id: str, max_questions: int = 999999):
    """
    ตรวจสอบและอัปเดตจำนวนคำถามต่อวันของผู้ใช้ (ไม่จำกัดจำนวนครั้ง)
    
    Args:
        user_id (str): ID ของผู้ใช้
        max_questions (int): จำนวนคำถามสูงสุดที่อนุญาตต่อวัน (ค่าเริ่มต้น: 999999 - ไม่จำกัด)
        
    Returns:
        tuple: (is_allowed, current_count, message)
    """
    # ปิดระบบนับ/อัปเดตจำนวนคำถาม (no-op) และอนุญาตเสมอ โดยไม่เขียน DB
    return True, 0, ""

# ฟังก์ชันวิเคราะห์เจตนาของคำถาม
def analyze_question_intent(question: str) -> dict:
    """
    วิเคราะห์เจตนาของคำถามเพื่อระบุว่าผู้ใช้ต้องการข้อมูลเฉพาะด้านใด
    
    Args:
        question (str): คำถามของผู้ใช้
        
    Returns:
        dict: ข้อมูลเจตนาของคำถาม
    """
    question_lower = question.lower()
    
    # ตรวจสอบคำถามเฉพาะด้าน
    intent = {
        "specific_topic": None,
        "is_general": False,
        "is_personality": False,
        "is_love": False,
        "is_career": False,
        "is_health": False,
        "is_finance": False,
        "is_lucky_colors": False
    }
    
    # ตรวจสอบคำถามเกี่ยวกับความรัก (ตรวจสอบก่อน personality เพื่อความแม่นยำ)
    love_keywords = ["ความรัก", "รัก", "แฟน", "คู่รัก", "ความสัมพันธ์", "คนรัก", "ความรัก", "ความสัมพันธ์"]
    if any(keyword in question_lower for keyword in love_keywords):
        intent["is_love"] = True
        intent["specific_topic"] = "love"
    
    # ตรวจสอบคำถามเกี่ยวกับลักษณะนิสัย
    personality_keywords = ["นิสัย", "ลักษณะ", "สัย", "เป็นคน", "บุคลิก", "ลักษณะนิสัย"]
    if any(keyword in question_lower for keyword in personality_keywords):
        intent["is_personality"] = True
        intent["specific_topic"] = "personality"
    
    # ตรวจสอบคำถามเกี่ยวกับอาชีพ/การงาน
    career_keywords = ["อาชีพ", "การงาน", "งาน", "อาชีพ", "การทำงาน", "งานที่เหมาะ", "อาชีพที่เหมาะ"]
    if any(keyword in question_lower for keyword in career_keywords):
        intent["is_career"] = True
        intent["specific_topic"] = "career"
    
    # ตรวจสอบคำถามเกี่ยวกับสุขภาพ
    health_keywords = ["สุขภาพ", "การดูแลสุขภาพ", "สุขภาพ", "การดูแลร่างกาย", "สุขภาพดี"]
    if any(keyword in question_lower for keyword in health_keywords):
        intent["is_health"] = True
        intent["specific_topic"] = "health"
    
    # ตรวจสอบคำถามเกี่ยวกับการเงิน
    finance_keywords = ["การเงิน", "เงิน", "การลงทุน", "การเงิน", "เงินทอง", "การเงิน"]
    if any(keyword in question_lower for keyword in finance_keywords):
        intent["is_finance"] = True
        intent["specific_topic"] = "finance"
    
    # ตรวจสอบคำถามเกี่ยวกับสีมงคล
    color_keywords = ["สีมงคล", "สีดี", "สีที่เหมาะ", "สีมงคล", "สีที่เหมาะ", "สีดี"]
    if any(keyword in question_lower for keyword in color_keywords):
        intent["is_lucky_colors"] = True
        intent["specific_topic"] = "lucky_colors"
    
    # ตรวจสอบคำถามทั่วไปเกี่ยวกับดวงชะตา
    general_horoscope_keywords = ["ทำนายดวง", "ดูดวง", "ดวงชะตา", "ดวงกำเนิด", "ทำนายดวงกำเนิด", "ดูดวงกำเนิด", "ราศีอะไร"]
    if any(keyword in question_lower for keyword in general_horoscope_keywords):
        # ถ้ายังไม่ได้กำหนด specific_topic ให้ถือว่าเป็นคำถามทั่วไป
        if not intent["specific_topic"]:
            intent["is_general"] = True
            intent["specific_topic"] = "general"
    
    # ตรวจสอบคำถามทั่วไปที่ใช้คำว่า "เป็นยังไง", "เป็นอย่างไร", "ยังไง", "อย่างไร", "เป็นไง"
    general_keywords = ["เป็นยังไง", "เป็นอย่างไร", "ยังไง", "อย่างไร", "เป็นไง"]
    if any(keyword in question_lower for keyword in general_keywords):
        # ถ้ายังไม่ได้กำหนด specific_topic ให้ถือว่าเป็นคำถามทั่วไป
        if not intent["specific_topic"]:
            intent["is_general"] = True
            intent["specific_topic"] = "general"
    
    # ถ้าไม่มีคำถามเฉพาะด้าน ให้ถือว่าเป็นคำถามทั่วไป
    if not any([intent["is_personality"], intent["is_love"], intent["is_career"], 
                intent["is_health"], intent["is_finance"], intent["is_lucky_colors"]]):
        intent["is_general"] = True
        intent["specific_topic"] = "general"
    
    return intent

# ฟังก์ชันปรับปรุงคำถามให้ชัดเจนขึ้นสำหรับคำถามต่อเนื่อง

# ฟังก์ชันสร้างข้อมูลบริบทการสนทนาก่อนหน้า
def get_conversation_context(user_context: dict = None) -> str:
    """
    สร้างข้อมูลบริบทการสนทนาก่อนหน้าสำหรับส่งให้ GPT
    
    Args:
        user_context (dict): ข้อมูลบริบทของผู้ใช้
        
    Returns:
        str: ข้อมูลบริบทการสนทนาก่อนหน้า
    """
    if not user_context:
        return "ไม่มีข้อมูลการสนทนาก่อนหน้า"
    
    context_parts = []
    
    # ข้อมูลการสนทนาล่าสุด
    if user_context.get("last_conversation"):
        last_conv = user_context["last_conversation"]
        context_parts.append(f"คำถามก่อนหน้า: {last_conv.get('question', 'ไม่มีข้อมูล')}")
        context_parts.append(f"คำตอบก่อนหน้า: {last_conv.get('answer', 'ไม่มีข้อมูล')[:200]}...")
    
    # ข้อมูลการสนทนาก่อนหน้า (ถ้ามี)
    if user_context.get("previous_conversation"):
        prev_conv = user_context["previous_conversation"]
        context_parts.append(f"คำถามก่อนหน้านั้น: {prev_conv.get('question', 'ไม่มีข้อมูล')}")
        context_parts.append(f"คำตอบก่อนหน้านั้น: {prev_conv.get('answer', 'ไม่มีข้อมูล')[:200]}...")
    
    # ข้อมูลการสนทนาหลายครั้งล่าสุด
    if user_context.get("recent_conversations") and len(user_context["recent_conversations"]) > 2:
        context_parts.append(f"จำนวนการสนทนาล่าสุด: {len(user_context['recent_conversations'])} ครั้ง")
    
    if context_parts:
        return "\n".join(context_parts)
    else:
        return "ไม่มีข้อมูลการสนทนาก่อนหน้า"

# ฟังก์ชันสร้างคำถามต่อเนื่องอัตโนมัติ

def calculate_zodiac_from_date(day: int, month: int) -> str:
    """
    คำนวณราศีจากวันและเดือน (Western Astrology)
    
    Args:
        day (int): วัน
        month (int): เดือน
        
    Returns:
        str: ชื่อราศี
    """
    # คำนวณราศีตามวันที่ (โหราศาสตร์ตะวันตก)
    if (month == 1 and day >= 20) or (month == 2 and day <= 18):
        return "กุมภ์"  # Aquarius: Jan 20 - Feb 18
    elif (month == 2 and day >= 19) or (month == 3 and day <= 20):
        return "มีน"   # Pisces: Feb 19 - Mar 20
    elif (month == 3 and day >= 21) or (month == 4 and day <= 19):
        return "เมษ"   # Aries: Mar 21 - Apr 19
    elif (month == 4 and day >= 20) or (month == 5 and day <= 20):
        return "พฤษภ"  # Taurus: Apr 20 - May 20
    elif (month == 5 and day >= 21) or (month == 6 and day <= 20):
        return "เมถุน" # Gemini: May 21 - Jun 20
    elif (month == 6 and day >= 21) or (month == 7 and day <= 22):
        return "กรกฎ"  # Cancer: Jun 21 - Jul 22
    elif (month == 7 and day >= 23) or (month == 8 and day <= 22):
        return "สิงห์"  # Leo: Jul 23 - Aug 22
    elif (month == 8 and day >= 23) or (month == 9 and day <= 22):
        return "กันย์"  # Virgo: Aug 23 - Sep 22
    elif (month == 9 and day >= 23) or (month == 10 and day <= 22):
        return "ตุล"   # Libra: Sep 23 - Oct 22
    elif (month == 10 and day >= 23) or (month == 11 and day <= 21):
        return "พิจิก" # Scorpio: Oct 23 - Nov 21
    elif (month == 11 and day >= 22) or (month == 12 and day <= 21):
        return "ธนู"   # Sagittarius: Nov 22 - Dec 21
    elif (month == 12 and day >= 22) or (month == 1 and day <= 19):
        return "มังกร" # Capricorn: Dec 22 - Jan 19
    else:
        return "มังกร"  # default



# ฟังก์ชัน format_astrology_response ถูกลบออกแล้วเนื่องจากไม่ใช้งาน

# ฟังก์ชัน add_supplementary_info ถูกลบออกแล้วเนื่องจากไม่ใช้งาน

# 🆕 ฟังก์ชันคำนวณ Semantic Similarity สำหรับ Follow-up Detection
def calculate_semantic_similarity(text1: str, text2: str, model=None) -> float:
    """
    คำนวณ semantic similarity ระหว่างสองข้อความโดยใช้ embedding model
    
    Args:
        text1 (str): ข้อความแรก
        text2 (str): ข้อความที่สอง
        model: SentenceTransformer model (ถ้า None จะสร้างใหม่)
        
    Returns:
        float: similarity score (0-1, ยิ่งสูงยิ่งคล้ายกัน)
    """
    try:
        import numpy as np
        
        # โหลด embedding model ถ้ายังไม่มี
        if model is None:
            model = SentenceTransformer("minishlab/potion-multilingual-128M", device="cpu")
        
        # สร้าง embeddings
        embedding1 = model.encode(text1, convert_to_numpy=True)
        embedding2 = model.encode(text2, convert_to_numpy=True)
        
        # คำนวณ cosine similarity
        similarity = np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )
        
        return float(similarity)
        
    except Exception as e:
        logger.warning(f"Error calculating semantic similarity: {e}")
        return 0.0

# ✔️ ตรวจสอบคำถามต่อเนื่องด้วย Semantic Similarity (แทน LLM)
def check_follow_up_question_with_semantic_similarity(
    question: str, 
    user_context: dict = None,
    similarity_threshold: float = 0.25
) -> Tuple[bool, float]:
    """
    ตรวจสอบว่าเป็นคำถามต่อเนื่องหรือไม่โดยใช้ Semantic Similarity
    
    ใช้ embedding model เพื่อคำนวณความคล้ายคลึงทางความหมายระหว่าง:
    - คำถามปัจจุบัน กับ คำถามก่อนหน้า
    - คำถามปัจจุบัน กับ คำตอบก่อนหน้า
    - คำถามปัจจุบัน กับ บริบทการสนทนา (คำถาม + คำตอบ)
    
    Args:
        question (str): คำถามปัจจุบัน
        user_context (dict): ข้อมูลบริบทของผู้ใช้
        similarity_threshold (float): threshold สำหรับตัดสินว่าเป็น follow-up (default: 0.25)
        
    Returns:
        tuple[bool, float]: (is_follow_up, max_similarity_score)
            - is_follow_up: True ถ้าเป็นคำถามต่อเนื่อง
            - max_similarity_score: similarity score สูงสุดที่คำนวณได้
    """
    try:
        # ถ้าไม่มีบริบทการสนทนา ให้ถือว่าไม่ใช่คำถามต่อเนื่อง
        if not user_context or not user_context.get("last_question"):
            return False, 0.0
        
        # ถ้ามีข้อมูลวันเกิดในคำถาม ให้ถือว่าไม่ใช่คำถามต่อเนื่อง
        has_birth_date_in_question = any(pattern in question for pattern in [
            "/", "-", ".", "เดือน", "ปี", "วันเกิด", "เกิด", "มกราคม", "กุมภาพันธ์", "มีนาคม", 
            "เมษายน", "พฤษภาคม", "มิถุนายน", "กรกฎาคม", "สิงหาคม", "กันยายน", 
            "ตุลาคม", "พฤศจิกายน", "ธันวาคม"
        ])
        
        if has_birth_date_in_question:
            return False, 0.0
        
        # โหลด embedding model
        model = SentenceTransformer("minishlab/potion-multilingual-128M", device="cpu")
        
        # สร้าง embedding สำหรับคำถามปัจจุบัน
        current_question_embedding = model.encode(question, convert_to_numpy=True)
        
        # ดึงข้อมูลบริบทก่อนหน้า
        last_question = user_context.get("last_question", "")
        last_response = user_context.get("last_response", "")
        
        # คำนวณ similarity scores หลายแบบ
        similarities = []
        
        # 1. Similarity ระหว่างคำถามปัจจุบันกับคำถามก่อนหน้า
        if last_question:
            # 🆕 ใช้ embedding ที่เก็บไว้ถ้ามี (จาก responses collection)
            last_response_obj = user_context.get("_last_response_obj")
            if last_response_obj and "question_embedding" in last_response_obj:
                # ใช้ embedding ที่เก็บไว้แล้ว
                import numpy as np
                last_question_embedding = np.array(last_response_obj["question_embedding"])
                sim_with_last_question = float(np.dot(current_question_embedding, last_question_embedding) / (
                    np.linalg.norm(current_question_embedding) * np.linalg.norm(last_question_embedding)
                ))
                logger.debug(f"✅ Used stored question embedding for similarity calculation")
            else:
                # สร้าง embedding ใหม่ถ้าไม่มี
                sim_with_last_question = calculate_semantic_similarity(
                    question, last_question, model
                )
            similarities.append(("last_question", sim_with_last_question))
            logger.debug(f"Similarity with last question: {sim_with_last_question:.4f}")
        
        # 2. Similarity ระหว่างคำถามปัจจุบันกับคำตอบก่อนหน้า
        if last_response:
            # 🆕 ใช้ embedding ที่เก็บไว้ถ้ามี (จาก responses collection)
            last_response_obj = user_context.get("_last_response_obj")
            if last_response_obj and "answer_embedding" in last_response_obj:
                # ใช้ embedding ที่เก็บไว้แล้ว
                import numpy as np
                last_answer_embedding = np.array(last_response_obj["answer_embedding"])
                sim_with_last_response = float(np.dot(current_question_embedding, last_answer_embedding) / (
                    np.linalg.norm(current_question_embedding) * np.linalg.norm(last_answer_embedding)
                ))
                logger.debug(f"✅ Used stored answer embedding for similarity calculation")
            else:
                # สร้าง embedding ใหม่ถ้าไม่มี (ใช้เฉพาะส่วนแรกเพื่อความเร็ว)
                sim_with_last_response = calculate_semantic_similarity(
                    question, last_response[:500], model
                )
            similarities.append(("last_response", sim_with_last_response))
            logger.debug(f"Similarity with last response: {sim_with_last_response:.4f}")
        
        # 3. Similarity ระหว่างคำถามปัจจุบันกับบริบทรวม (คำถาม + คำตอบ)
        if last_question and last_response:
            context_text = f"{last_question} {last_response[:300]}"
            sim_with_context = calculate_semantic_similarity(
                question, context_text, model
            )
            similarities.append(("context", sim_with_context))
            logger.debug(f"Similarity with context: {sim_with_context:.4f}")
        
        # 4. Similarity กับ recent conversations (ถ้ามี)
        if user_context.get("recent_conversations"):
            recent_convs = user_context["recent_conversations"][:3]  # เอาแค่ 3 อันล่าสุด
            for i, conv in enumerate(recent_convs):
                conv_text = f"{conv.get('question', '')} {conv.get('answer', '')[:200]}"
                if conv_text.strip():
                    sim_with_recent = calculate_semantic_similarity(
                        question, conv_text, model
                    )
                    similarities.append((f"recent_conv_{i}", sim_with_recent))
                    logger.debug(f"Similarity with recent conversation {i}: {sim_with_recent:.4f}")
        
        # หา similarity score สูงสุด
        if not similarities:
            return False, 0.0
        
        max_similarity = max(sim[1] for sim in similarities)
        max_source = max(similarities, key=lambda x: x[1])[0]
        
        # ตัดสินว่าเป็น follow-up หรือไม่
        is_follow_up = max_similarity >= similarity_threshold
        
        logger.info(
            f"Semantic Similarity Check: '{question[:50]}...' -> "
            f"Max similarity: {max_similarity:.4f} (source: {max_source}), "
            f"Threshold: {similarity_threshold}, "
            f"Is follow-up: {is_follow_up}"
        )
        
        return is_follow_up, max_similarity
        
    except Exception as e:
        logger.warning(f"Error in semantic similarity follow-up check: {e}")
        # ถ้าเกิด error ให้ return False (ไม่ใช่ follow-up)
        return False, 0.0

# ✔️ ตรวจสอบคำถามต่อเนื่องด้วย LLM (OpenAI GPT)
def refine_follow_up_question_with_llm(question: str, user_context: dict = None) -> str:
    """
    ปรับคำถามล่าสุดกับคำถามก่อนหน้าเข้าด้วยกันโดยใช้ LLM (OpenAI GPT)
    
    ใช้ LLM เพื่อปรับปรุงคำถามปัจจุบันให้ชัดเจนขึ้นโดยใช้บริบทจากคำถามก่อนหน้าและคำตอบก่อนหน้า
    โดยส่งข้อมูล [last_question, last_response, Current question] ไปให้ LLM ปรับคำถาม
    
    Args:
        question (str): คำถามปัจจุบัน
        user_context (dict): ข้อมูลบริบทของผู้ใช้
        
    Returns:
        str: คำถามที่ปรับปรุงแล้ว
    """
    try:
        # ถ้าไม่มีบริบทการสนทนา ให้ส่งคำถามเดิมกลับไป
        if not user_context or not user_context.get("last_question"):
            return question
        
        # ถ้ามีข้อมูลวันเกิดในคำถาม ให้ส่งคำถามเดิมกลับไป (ไม่ต้องปรับ)
        has_birth_date_in_question = any(pattern in question for pattern in [
            "/", "-", ".", "เดือน", "ปี", "วันเกิด", "เกิด", "มกราคม", "กุมภาพันธ์", "มีนาคม", 
            "เมษายน", "พฤษภาคม", "มิถุนายน", "กรกฎาคม", "สิงหาคม", "กันยายน", 
            "ตุลาคม", "พฤศจิกายน", "ธันวาคม"
        ])
        
        if has_birth_date_in_question:
            return question
        
        # ใช้ LLM เพื่อปรับคำถาม
        from openai import OpenAI
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key or openai_key == "your-openai-api-key-here":
            logger.warning("OpenAI API key not configured, returning original question")
            return question
        
        client = OpenAI(api_key=openai_key)
        
        # ดึงข้อมูลบริบทก่อนหน้า
        last_question = user_context.get("last_question", "")
        last_response = user_context.get("last_response", "")
        user_zodiac = user_context.get("zodiac_sign", "")
        
        # สร้าง prompt สำหรับปรับคำถาม
        prompt = f"""คุณเป็นผู้เชี่ยวชาญในการปรับปรุงคำถามให้ชัดเจนขึ้นโดยใช้บริบทการสนทนาก่อนหน้า

คำถามก่อนหน้า: "{last_question}"
คำตอบก่อนหน้า: "{last_response[:500]}..."
คำถามปัจจุบัน: "{question}"
ราศีของผู้ใช้: {user_zodiac if user_zodiac else "ไม่ระบุ"}

กรุณาปรับปรุงคำถามปัจจุบันให้ชัดเจนขึ้นโดย:
1. ถ้าคำถามปัจจุบันอ้างอิงถึง "ราศีนี้", "ราศีของฉัน", "คนราศีนี้" ให้ระบุชื่อราศีชัดเจน
2. ถ้าคำถามปัจจุบันเป็นคำถามสั้นๆ ที่อ้างอิงถึงข้อมูลก่อนหน้า ให้ทำให้ชัดเจนขึ้น
3. ถ้าคำถามปัจจุบันถามเกี่ยวกับข้อมูลที่เกี่ยวข้องกับคำตอบก่อนหน้า ให้เชื่อมโยงให้ชัดเจน
4. รักษาความหมายเดิมของคำถามไว้
5. ใช้ภาษาธรรมชาติและเข้าใจง่าย

ตอบแค่คำถามที่ปรับปรุงแล้วเท่านั้น ไม่ต้องอธิบายเพิ่มเติม:"""
        
        # ใช้ชื่อโมเดลจาก ENV ถ้าไม่ระบุจะใช้ gpt-4o-mini
        openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        response = client.chat.completions.create(
            model=openai_model,
            messages=[
                {"role": "system", "content": "คุณเป็นผู้เชี่ยวชาญในการปรับปรุงคำถามให้ชัดเจนขึ้นโดยใช้บริบทการสนทนา ตอบแค่คำถามที่ปรับปรุงแล้วเท่านั้น"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=200
        )
        
        refined_question = response.choices[0].message.content.strip()
        
        logger.info(
            f"LLM Question Refinement: '{question[:50]}...' -> '{refined_question[:50]}...'"
        )
        
        # ประเมินคุณภาพของคำถามที่ปรับปรุงแล้ว
        try:
            evaluation_prompt = f"""คุณเป็นผู้เชี่ยวชาญในการประเมินคุณภาพของคำถามที่ปรับปรุงแล้ว

คำถามเดิม: "{question}"
คำถามที่ปรับปรุงแล้ว: "{refined_question}"
คำถามก่อนหน้า: "{last_question}"
คำตอบก่อนหน้า: "{last_response[:500]}..."

กรุณาประเมินคุณภาพของคำถามที่ปรับปรุงแล้วตามเกณฑ์ต่อไปนี้ (ให้คะแนน 1-10):
1. ความชัดเจน: คำถามที่ปรับปรุงแล้วชัดเจนและเข้าใจง่ายหรือไม่
2. ความเกี่ยวข้อง: คำถามที่ปรับปรุงแล้วเกี่ยวข้องกับบริบทการสนทนาก่อนหน้าหรือไม่
3. ความสมบูรณ์: คำถามที่ปรับปรุงแล้วมีข้อมูลครบถ้วนหรือไม่
4. การปรับปรุง: คำถามที่ปรับปรุงแล้วดีกว่าคำถามเดิมหรือไม่

ตอบแค่ตัวเลขคะแนนรวม (1-10) เท่านั้น:"""
            
            evaluation_response = client.chat.completions.create(
                model=openai_model,
                messages=[
                    {"role": "system", "content": "คุณเป็นผู้เชี่ยวชาญในการประเมินคุณภาพของคำถาม ตอบแค่ตัวเลขคะแนน 1-10 เท่านั้น"},
                    {"role": "user", "content": evaluation_prompt}
                ],
                temperature=0.1,
                max_tokens=10
            )
            
            score_text = evaluation_response.choices[0].message.content.strip()
            # แยกตัวเลขจากข้อความ (กรณีที่โมเดลตอบมาพร้อมข้อความอื่น)
            score_match = re.search(r'\d+', score_text)
            if score_match:
                score = int(score_match.group())
                # จำกัดคะแนนให้อยู่ในช่วง 1-10
                score = max(1, min(10, score))
            else:
                score = 5  # ค่า default ถ้าไม่สามารถแยกตัวเลขได้
            
            # พิมพ์คะแนนลง terminal
            print(f"\n{'='*60}")
            print(f"📊 การประเมินคุณภาพของคำถามที่ปรับปรุงแล้ว")
            print(f"{'='*60}")
            print(f"คำถามเดิม: {question}")
            print(f"คำถามที่ปรับปรุงแล้ว: {refined_question}")
            print(f"คะแนนการประเมิน: {score}/10")
            print(f"{'='*60}\n")
            
            logger.info(f"Query refinement self-evaluation score: {score}/10")
            
        except Exception as eval_error:
            logger.warning(f"Error in query refinement self-evaluation: {eval_error}")
            print(f"\n⚠️  ไม่สามารถประเมินคุณภาพของคำถามที่ปรับปรุงแล้วได้: {eval_error}\n")
        
        return refined_question
        
    except Exception as e:
        logger.warning(f"Error in LLM question refinement: {e}, returning original question")
        return question

def check_follow_up_question_with_llm(question: str, user_context: dict = None) -> bool:
    """
    ตรวจสอบว่าเป็นคำถามต่อเนื่องหรือไม่โดยใช้ LLM (OpenAI GPT)
    
    ใช้ LLM เพื่อวิเคราะห์ความเกี่ยวข้องระหว่างคำถามปัจจุบันกับบริบทการสนทนาก่อนหน้า
    โดยส่งข้อมูล [last_question, last_response, Current question] ไปให้ LLM วิเคราะห์
    
    Args:
        question (str): คำถามปัจจุบัน
        user_context (dict): ข้อมูลบริบทของผู้ใช้
        
    Returns:
        bool: True ถ้าเป็นคำถามต่อเนื่อง, False ถ้าไม่ใช่
    """
    try:
        # ถ้าไม่มีบริบทการสนทนา ให้ถือว่าไม่ใช่คำถามต่อเนื่อง
        if not user_context or not user_context.get("last_question"):
            return False
        
        # ถ้ามีข้อมูลวันเกิดในคำถาม ให้ถือว่าไม่ใช่คำถามต่อเนื่อง
        has_birth_date_in_question = any(pattern in question for pattern in [
            "/", "-", ".", "เดือน", "ปี", "วันเกิด", "เกิด", "มกราคม", "กุมภาพันธ์", "มีนาคม", 
            "เมษายน", "พฤษภาคม", "มิถุนายน", "กรกฎาคม", "สิงหาคม", "กันยายน", 
            "ตุลาคม", "พฤศจิกายน", "ธันวาคม"
        ])
        
        if has_birth_date_in_question:
            return False
        
        # ใช้ LLM เพื่อตรวจสอบความเกี่ยวข้อง
        from openai import OpenAI
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key or openai_key == "your-openai-api-key-here":
            logger.warning("OpenAI API key not configured, falling back to semantic similarity")
            # Fallback to semantic similarity if no API key
            is_follow_up, _ = check_follow_up_question_with_semantic_similarity(
                question, user_context, similarity_threshold=0.25
            )
            return is_follow_up
        
        client = OpenAI(api_key=openai_key)
        
        # ดึงข้อมูลบริบทก่อนหน้า
        last_question = user_context.get("last_question", "")
        last_response = user_context.get("last_response", "")
        
        # สร้าง prompt สำหรับตรวจสอบความเกี่ยวข้อง
        prompt = f"""คุณเป็นผู้เชี่ยวชาญในการวิเคราะห์ความเกี่ยวข้องของคำถามในบริบทการสนทนา

คำถามก่อนหน้า: "{last_question}"
คำตอบก่อนหน้า: "{last_response[:300]}..."
คำถามปัจจุบัน: "{question}"

กรุณาตอบว่า "YES" ถ้าคำถามปัจจุบันเกี่ยวข้องกับคำถามก่อนหน้าหรือคำตอบก่อนหน้า หรือ "NO" ถ้าไม่เกี่ยวข้อง

เกณฑ์การตัดสิน:
- ถ้าคำถามปัจจุบันถามเกี่ยวกับข้อมูลที่เกี่ยวข้องกับคำตอบก่อนหน้า = YES
- ถ้าคำถามปัจจุบันถามต่อจากหัวข้อเดียวกัน (เช่น ถาม "ความรัก" แล้วถาม "งาน" ต่อ) = YES
- ถ้าคำถามปัจจุบันถามเรื่องใหม่ที่ไม่เกี่ยวข้อง = NO
- ถ้าคำถามปัจจุบันมีข้อมูลวันเกิดใหม่ = NO
- ถ้าคำถามปัจจุบันถามต่อจากข้อมูลราศีที่ได้ = YES
- ถ้าคำถามปัจจุบันเป็นคำถามทั่วไปที่อ้างอิงถึงข้อมูลก่อนหน้า = YES

ตอบแค่ "YES" หรือ "NO" เท่านั้น:"""
        
        # ใช้ชื่อโมเดลจาก ENV ถ้าไม่ระบุจะใช้ gpt-4o-mini
        openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        response = client.chat.completions.create(
            model=openai_model,
            messages=[
                {"role": "system", "content": "คุณเป็นผู้เชี่ยวชาญในการวิเคราะห์ความเกี่ยวข้องของคำถามในบริบทการสนทนา ตอบแค่ YES หรือ NO เท่านั้น"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # ใช้ temperature ต่ำเพื่อความสม่ำเสมอ
            max_tokens=10
        )
        
        result = response.choices[0].message.content.strip().upper()
        is_follow_up = result == "YES"
        
        logger.info(
            f"LLM Follow-up Detection: '{question[:50]}...' -> {result} "
            f"(is_follow_up: {is_follow_up})"
        )
        
        # ประเมินตัวเองเมื่อตรวจพบว่าเป็น follow-up
        if is_follow_up:
            try:
                evaluation_prompt = f"""คุณเป็นผู้เชี่ยวชาญในการประเมินความมั่นใจในการตัดสินใจว่าเป็นคำถามต่อเนื่องหรือไม่

คำถามก่อนหน้า: "{last_question}"
คำตอบก่อนหน้า: "{last_response[:300]}..."
คำถามปัจจุบัน: "{question}"
ผลการตัดสิน: YES (เป็นคำถามต่อเนื่อง)

กรุณาประเมินความมั่นใจในการตัดสินใจนี้ตามเกณฑ์ต่อไปนี้ (ให้คะแนน 1-10):
1. ความเกี่ยวข้อง: คำถามปัจจุบันเกี่ยวข้องกับคำถามก่อนหน้าหรือคำตอบก่อนหน้ามากแค่ไหน
2. ความต่อเนื่อง: คำถามปัจจุบันเป็นคำถามต่อเนื่องจากบริบทการสนทนาก่อนหน้าหรือไม่
3. ความชัดเจน: การตัดสินใจนี้ชัดเจนและแน่ใจแค่ไหน
4. ความเหมาะสม: การตัดสินใจว่าเป็น follow-up นี้เหมาะสมหรือไม่

ตอบแค่ตัวเลขคะแนนรวม (1-10) เท่านั้น:"""
                
                evaluation_response = client.chat.completions.create(
                    model=openai_model,
                    messages=[
                        {"role": "system", "content": "คุณเป็นผู้เชี่ยวชาญในการประเมินความมั่นใจในการตัดสินใจ ตอบแค่ตัวเลขคะแนน 1-10 เท่านั้น"},
                        {"role": "user", "content": evaluation_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=10
                )
                
                score_text = evaluation_response.choices[0].message.content.strip()
                # แยกตัวเลขจากข้อความ (กรณีที่โมเดลตอบมาพร้อมข้อความอื่น)
                score_match = re.search(r'\d+', score_text)
                if score_match:
                    score = int(score_match.group())
                    # จำกัดคะแนนให้อยู่ในช่วง 1-10
                    score = max(1, min(10, score))
                else:
                    score = 5  # ค่า default ถ้าไม่สามารถแยกตัวเลขได้
                
                # พิมพ์คะแนนลง terminal
                print(f"\n{'='*60}")
                print(f"📊 การประเมินตัวเอง: Follow-up Detection")
                print(f"{'='*60}")
                print(f"คำถามก่อนหน้า: {last_question}")
                print(f"คำถามปัจจุบัน: {question}")
                print(f"ผลการตัดสิน: YES (เป็นคำถามต่อเนื่อง)")
                print(f"คะแนนการประเมินความมั่นใจ: {score}/10")
                print(f"{'='*60}\n")
                
                logger.info(f"Follow-up detection self-evaluation score: {score}/10")
                
            except Exception as eval_error:
                logger.warning(f"Error in follow-up detection self-evaluation: {eval_error}")
                print(f"\n⚠️  ไม่สามารถประเมินความมั่นใจในการตัดสินใจ follow-up ได้: {eval_error}\n")
        
        return is_follow_up
        
    except Exception as e:
        logger.warning(f"Error in LLM follow-up check: {e}, falling back to semantic similarity")
        # ถ้าเกิด error ให้ fallback ไปใช้ semantic similarity
        try:
            is_follow_up, _ = check_follow_up_question_with_semantic_similarity(
                question, user_context, similarity_threshold=0.25
            )
            return is_follow_up
        except:
            # ถ้า semantic similarity ก็ error ให้ return False
            return False

def ask_question_to_rag(question: str, user_id: str = "unknown", provided_chart_info: dict = None) -> str:
    # print(f"\n=== เริ่มการค้นหาข้อมูลสำหรับคำถาม: {question} ===")
    
    # ตรวจสอบจำนวนคำถามต่อเนื่องก่อน (ไม่จำกัดจำนวนครั้ง)
    is_allowed, current_count, limit_message = check_and_update_question_limit(user_id)
    if not is_allowed:
        logger.info(f"🚫 Question limit exceeded for user {user_id}: {current_count}/3")
        return limit_message
    
    # ดึงข้อมูลบริบทการสนทนาของผู้ใช้ก่อน
    user_context = get_user_context(user_id)
    
    # ตรวจสอบว่าเป็นคำถามต่อเนื่องหรือไม่โดยใช้ LLM (ตาม diagram)
    print(f"\n{'='*60}")
    print(f"🔍 กำลังตรวจสอบว่าเป็น Follow-up Question...")
    print(f"{'='*60}")
    print(f"คำถามปัจจุบัน: {question}")
    is_follow_up_question = check_follow_up_question_with_llm(question, user_context)
    print(f"ผลการตรวจสอบ: {'YES (เป็น follow-up)' if is_follow_up_question else 'NO (ไม่ใช่ follow-up)'}")
    print(f"{'='*60}\n")
    logger.info(f"Follow-up detection (LLM): question='{question[:50]}...', is_follow_up={is_follow_up_question}")
    
    user_birth_date = user_context.get("birth_date") if user_context else None
    user_zodiac = user_context.get("zodiac_sign") if user_context else None
    
    # ตรวจสอบว่ามีข้อมูลวันเกิดและเวลาเกิดในคำถามหรือไม่ (เสมอ)
    birth_info_from_question = extract_birth_info_from_message(question)
    astrology_chart = None
    
    # ถ้ามี chart_info ที่ส่งมา ให้ใช้เลย (กรณีเรียกจาก generate_birth_chart_prediction)
    if provided_chart_info:
        astrology_chart = provided_chart_info
        is_follow_up_question = False  # ถ้ามี chart_info ที่ส่งมา ให้ถือว่าไม่ใช่คำถามต่อเนื่อง
        logger.info(f"ใช้ chart_info ที่ส่งมา: ราศี{astrology_chart.get('zodiac_sign', 'Unknown')}")
    
    # เดิม: หากเป็นคำถามต่อเนื่องแต่ไม่มีบริบทจะคืนข้อความแจ้งเตือน
    # ใหม่: ตอบแบบทั่วไปไปก่อน (ไม่บังคับให้ระบุวันเกิด)
    if is_follow_up_question and not user_context and not (birth_info_from_question and birth_info_from_question.get('date')):
        is_follow_up_question = False
    
    # ถ้ามีข้อมูลวันเกิดในคำถาม ให้ถือว่าไม่ใช่คำถามต่อเนื่อง
    if birth_info_from_question and birth_info_from_question.get('date'):
        is_follow_up_question = False
        logger.info(f"ไม่ใช่คำถามต่อเนื่อง เพราะมีข้อมูลวันเกิดในคำถาม: {birth_info_from_question['date']}")
    
    # เดิม: ถ้าเป็น follow-up แต่ไม่มีราศีในบริบทจะคืนข้อความแจ้งเตือน
    # ใหม่: ปลดสถานะเป็นคำถามทั่วไป แล้วดำเนินการตอบตามปกติ
    if is_follow_up_question and user_context and not user_zodiac and not birth_info_from_question:
        is_follow_up_question = False
    
    # Debug: แสดงข้อมูลการตัดสินใจ (ปิดการแสดงผล)
    # print(f"DEBUG - คำถาม: {question}")
    # print(f"DEBUG - is_follow_up_question: {is_follow_up_question}")
    # print(f"DEBUG - user_context: {user_context is not None}")
    # print(f"DEBUG - user_zodiac: {user_zodiac}")
    # print(f"DEBUG - birth_info_from_question: {birth_info_from_question}")
    
    # สร้างข้อมูลดวงชะตาเมื่อมีข้อมูลวันเกิดในคำถาม (ถ้ายังไม่มี chart_info อยู่แล้ว)
    if not astrology_chart and birth_info_from_question and birth_info_from_question['date']:
        logger.info(f"พบข้อมูลวันเกิดในคำถาม: {birth_info_from_question['date']}")
        if birth_info_from_question['time']:
            logger.info(f"พบเวลาเกิดในคำถาม: {birth_info_from_question['time']}")
        
        # สร้างข้อมูลดวงชะตารายละเอียด
        astrology_chart = generate_detailed_astrology_reading(question)
        if astrology_chart:
            logger.info(f"สร้างดวงชะตาสำเร็จ: ราศี{astrology_chart['zodiac_sign']} ({astrology_chart['zodiac_element']})")
    elif not astrology_chart and user_context and user_zodiac and is_follow_up_question:
        # สำหรับคำถามต่อเนื่อง ให้ใช้ข้อมูลจากบริบท
        # print(f"DEBUG - ใช้ข้อมูลดวงชะตาจากบริบท: ราศี{user_zodiac}")
        # สร้างข้อมูลดวงชะตาจากบริบท
        zodiac_english_map = {
            'เมษ': 'Aries', 'พฤษภ': 'Taurus', 'มิถุน': 'Gemini', 'กรกฎ': 'Cancer',
            'สิงห์': 'Leo', 'กันย์': 'Virgo', 'ตุล': 'Libra', 'พิจิก': 'Scorpio',
            'ธนู': 'Sagittarius', 'มังกร': 'Capricorn', 'กุมภ์': 'Aquarius', 'มีน': 'Pisces'
        }
        
        astrology_chart = {
            'zodiac_sign': user_zodiac,
            'zodiac_english': zodiac_english_map.get(user_zodiac, user_zodiac),
            'zodiac_element': user_context.get('zodiac_element', ''),
            'zodiac_quality': user_context.get('zodiac_quality', ''),
            'birth_date': user_birth_date,
            'birth_time': user_context.get('birth_time', ''),
            'age': user_context.get('age', ''),
            'detailed_reading': user_context.get('detailed_reading', {})
        }
        # print(f"DEBUG - astrology_chart: {astrology_chart}")
    
    # ตรวจสอบว่ามีข้อมูลดวงชะตาหรือไม่ ถ้าไม่มีให้ตอบข้อความแจ้งเตือน
    if not astrology_chart or not astrology_chart.get('zodiac_sign'):
        # ไม่มีดวงชะตาเพียงพอ ก็ยังตอบแบบทั่วไปได้
        pass
    
    # สร้างข้อมูลบริบทสำหรับการสนทนา
    context_info = ""
    if user_context:
        if user_birth_date:
            context_info += f"\nข้อมูลผู้ใช้: วันเกิด {user_birth_date}"
        if user_zodiac:
            context_info += f" ราศี {user_zodiac}"
        if user_context.get("zodiac_element"):
            context_info += f" ธาตุ {user_context.get('zodiac_element')}"
        if user_context.get("last_question"):
            context_info += f"\nคำถามก่อนหน้า: {user_context.get('last_question')}"
    
    birth_info = context_info
    # print(f"ข้อมูลผู้ใช้จากฐานข้อมูล: {context_info if context_info else 'ไม่มีข้อมูล'}")
    
    # วิเคราะห์เจตนาของคำถาม
    question_intent = analyze_question_intent(question)
    
    # 🆕 ปรับปรุง query เมื่อมีข้อมูลวันเกิดในคำถาม - ใช้ชื่อราศีแทนวันเกิดเพื่อให้ค้นหาได้ดีขึ้น
    if astrology_chart and astrology_chart.get('zodiac_sign'):
        zodiac_sign = astrology_chart['zodiac_sign']
        # ตรวจสอบว่าคำถามมีวันเกิดหรือไม่ (เช่น "07/09/2003" หรือ "ทำนายดวง")
        has_birth_date_in_question = bool(birth_info_from_question and birth_info_from_question.get('date'))
        
        # ถ้ามีวันเกิดในคำถาม ให้สร้าง query ที่ใช้ชื่อราศีแทน
        if has_birth_date_in_question:
            # สร้าง query ที่ใช้ชื่อราศีแทนวันเกิด และเพิ่มคำสำคัญที่ตรงกับข้อมูลในฐานข้อมูล
            # ใช้คำที่หลากหลายเพื่อเพิ่มโอกาสในการค้นหา
            if 'ราศีอะไร' in question or 'ราศี' in question:
                # 🆕 ใช้ query ที่ครอบคลุมมากขึ้น - รวมทั้งการงาน การเงิน ความรัก เพื่อให้ค้นหาข้อมูลได้ครบ
                question = f"ราศี{zodiac_sign} ลักษณะนิสัย บุคลิกภาพ การงาน การเงิน ความรัก โหราศาสตร์"
            elif 'ทำนายดวง' in question or 'ดวงชะตา' in question or 'ดวงกำเนิด' in question:
                question = f"ราศี{zodiac_sign} ลักษณะนิสัย การงาน การเงิน ความรัก โหราศาสตร์"
            else:
                # ถ้ามีวันเกิดแต่ไม่มี keyword ชัดเจน ให้เพิ่มชื่อราศีใน query
                question = f"ราศี{zodiac_sign} {question} โหราศาสตร์"
            
            logger.info(f"ปรับปรุง query สำหรับวันเกิด: ใช้ชื่อราศี '{zodiac_sign}' แทนวันเกิด -> '{question}'")
    
    # ปรับปรุงคำถามให้ชัดเจนขึ้นสำหรับคำถามต่อเนื่องโดยใช้ LLM
    if is_follow_up_question and user_context:
        print(f"\n{'='*60}")
        print(f"🔄 กำลังปรับปรุงคำถาม (Refine Query)...")
        print(f"{'='*60}")
        print(f"คำถามเดิม: {question}")
        refined_question = refine_follow_up_question_with_llm(question, user_context)
        if refined_question and refined_question != question:
            logger.info(f"Question refined: '{question[:50]}...' -> '{refined_question[:50]}...'")
            question = refined_question
        else:
            print(f"คำถามไม่มีการเปลี่ยนแปลง (ไม่จำเป็นต้องปรับปรุง)")
            print(f"{'='*60}\n")
    else:
        if not is_follow_up_question:
            print(f"\n{'='*60}")
            print(f"ℹ️  ไม่ใช่ Follow-up Question - ไม่มีการ Refine Query")
            print(f"{'='*60}\n")
    
    # ลองค้นหาจาก MongoDB แบบ Manual Search
    retrieved_docs = []
    try:
        print("🔍 กำลังค้นหาจาก MongoDB...")
        
        # 🆕 ตรวจสอบการเชื่อมต่อ MongoDB ก่อนทำ retrieval
        print(f"\n{'='*60}")
        print(f"🔍 กำลังตรวจสอบการเชื่อมต่อ MongoDB...")
        print(f"{'='*60}")
        is_ready, verify_message, conn_info = verify_mongodb_connection_for_retrieval()
        print(f"{verify_message}")
        print(f"{'='*60}\n")
        
        if not is_ready:
            print(f"⚠️ MongoDB ไม่พร้อมใช้งานสำหรับ retrieval: {verify_message}")
            print(f"   การค้นหาจาก MongoDB ถูกข้าม")
            retrieved_docs = []
        else:
            # โหลด embedding model
            import numpy as np
            
            # ใช้ CPU เพื่อหลีกเลี่ยงปัญหา MPS device
            model = SentenceTransformer("minishlab/potion-multilingual-128M", device="cpu")
            query_embedding = model.encode(question)
            print(f"✅ สร้าง query embedding สำเร็จ (ขนาด: {len(query_embedding)} dimensions)")
            
            # ============================================================
            # ✅ ระบบ RAG: ดึงข้อมูลจาก MongoDB ต้นฉบับเท่านั้น
            # ============================================================
            # ใช้ ORIGINAL_DB_NAME (astrobot_original) เท่านั้น
            # Collections: original_text_chunks, original_image_chunks, original_table_chunks
            # ใช้ field 'text' จากเอกสารต้นฉบับ (ไม่ใช้ summary)
            # ใช้ embeddings ที่สร้างจาก text ต้นฉบับ
            # ============================================================
            collections_to_search = [
                "original_text_chunks",      # ✅ มี text ต้นฉบับ และ embeddings
                "original_image_chunks",     # ✅ มี text ต้นฉบับ, embeddings (text), และ image_embeddings
                "original_table_chunks",     # ✅ มี text ต้นฉบับ และ embeddings
            ]
            
            # 🆕 ใช้ client ที่ได้จากการตรวจสอบแล้ว
            client = conn_info.get('client')
            db = conn_info.get('db')
            
            # แก้ไข: MongoDB database objects ไม่สามารถใช้ truth value testing ได้
            if client is None or db is None:
                print("⚠️ ไม่สามารถใช้ MongoDB connection ที่ตรวจสอบแล้วได้")
                retrieved_docs = []
            else:
                print(f"🔗 ใช้ MongoDB connection ที่ตรวจสอบแล้ว")
                print(f"   Database: {ORIGINAL_DB_NAME}")
                print(f"   Collections ที่จะค้นหา: {collections_to_search}")
                
                try:
                    # แสดงข้อมูล collections ที่มี
                    collections_status = conn_info.get('collections', {})
                    for collection_name in collections_to_search:
                        status = collections_status.get(collection_name, {})
                        if status.get('exists'):
                            print(f"   ✅ {collection_name}: {status.get('doc_count', 0)} เอกสาร, มี embeddings: {status.get('has_embeddings', False)}")
                        else:
                            print(f"   ❌ {collection_name}: ไม่มี collection นี้")
                    
                    print("✅ MongoDB พร้อมสำหรับ retrieval")
                    
                    # เริ่มทำ retrieval โดยใช้ client และ db ที่ตรวจสอบแล้ว
                    for collection_name in collections_to_search:
                        try:
                            print(f"📂 กำลังค้นหาใน collection: {collection_name}")
                            
                            # ตรวจสอบว่า collection มีอยู่จริงและมีข้อมูล
                            collection_status = collections_status.get(collection_name, {})
                            if not collection_status.get('exists'):
                                print(f"   ⚠️ Collection '{collection_name}' ไม่มีอยู่ใน database!")
                                continue
                            
                            if collection_status.get('doc_count', 0) == 0:
                                print(f"   ⚠️ Collection '{collection_name}' ว่างเปล่า (0 เอกสาร)")
                                continue
                            
                            # ใช้ collection จาก db ที่ตรวจสอบแล้ว
                            collection = db[collection_name]
                            
                            # ดึงข้อมูลทั้งหมด
                            docs = list(collection.find({}))
                            print(f"   พบเอกสารใน {collection_name}: {len(docs)} เอกสาร")
                            
                            # Debug: แสดงโครงสร้างของเอกสารแรก (ถ้ามี)
                            if docs:
                                first_doc = docs[0]
                                print(f"   📋 โครงสร้างเอกสารแรก (ตัวอย่าง):")
                                print(f"      - Fields: {list(first_doc.keys())}")
                                print(f"      - มี 'embeddings': {'embeddings' in first_doc}")
                                if 'embeddings' in first_doc:
                                    emb = first_doc['embeddings']
                                    print(f"      - Embedding type: {type(emb)}, length: {len(emb) if isinstance(emb, (list, np.ndarray)) else 'N/A'}")
                                print(f"      - มี 'text': {'text' in first_doc}")
                            
                            if docs:
                                # ✅ คำนวณ similarity scores (ใช้ embeddings ที่สร้างจาก text)
                                similarities = []
                                docs_without_embeddings = 0
                                docs_with_dimension_mismatch = 0
                                
                                for doc_idx, doc in enumerate(docs):
                                    if 'embeddings' not in doc:
                                        docs_without_embeddings += 1
                                        if doc_idx < 3:  # แสดงเฉพาะ 3 ตัวแรกเพื่อไม่ให้ output เยอะเกินไป
                                            print(f"   ⚠️ เอกสารที่ {doc_idx+1} ไม่มี field 'embeddings'")
                                        continue
                                    
                                    try:
                                        # ✅ embeddings ถูกสร้างจาก text
                                        doc_embedding = np.array(doc['embeddings'])
                                        
                                        # ตรวจสอบว่า dimensions ตรงกัน
                                        if len(doc_embedding) != len(query_embedding):
                                            docs_with_dimension_mismatch += 1
                                            if doc_idx < 3:
                                                print(f"   ⚠️ Warning: Embedding dimensions ไม่ตรงกัน (doc: {len(doc_embedding)}, query: {len(query_embedding)})")
                                            continue
                                        
                                        similarity = np.dot(query_embedding, doc_embedding) / (
                                            np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                                        )
                                        similarities.append((similarity, doc))
                                    except Exception as emb_error:
                                        if doc_idx < 3:
                                            print(f"   ❌ Error ในการคำนวณ similarity สำหรับเอกสารที่ {doc_idx+1}: {emb_error}")
                                        continue
                                
                                # แสดงสรุปปัญหา
                                if docs_without_embeddings > 0:
                                    print(f"   ⚠️ พบเอกสารที่ไม่มี embeddings: {docs_without_embeddings}/{len(docs)} เอกสาร")
                                if docs_with_dimension_mismatch > 0:
                                    print(f"   ⚠️ พบเอกสารที่มี embedding dimension ไม่ตรงกัน: {docs_with_dimension_mismatch}/{len(docs)} เอกสาร")
                                
                                if len(similarities) == 0:
                                    print(f"   ⚠️ ไม่สามารถคำนวณ similarity ได้เลย (ไม่มีเอกสารที่มี embeddings ที่ถูกต้อง)")
                                    print(f"   💡 ตรวจสอบว่า:")
                                    print(f"      - เอกสารมี field 'embeddings' หรือไม่")
                                    print(f"      - Embedding dimensions ตรงกับ query embedding หรือไม่ ({len(query_embedding)} dimensions)")
                                    # 🆕 ถ้าไม่มี similarities และมีวันเกิด ให้ลองใช้ query ที่ง่ายกว่า
                                    if birth_info_from_question and birth_info_from_question.get('date'):
                                        print(f"   🔄 ลองใช้ query ที่ง่ายกว่า: 'โหราศาสตร์'")
                                        simple_query_emb = model.encode("โหราศาสตร์")
                                        simple_similarities = []
                                        for doc in docs:
                                            if 'embeddings' in doc:
                                                try:
                                                    doc_emb = np.array(doc['embeddings'])
                                                    if len(doc_emb) == len(simple_query_emb):
                                                        sim = np.dot(simple_query_emb, doc_emb) / (
                                                            np.linalg.norm(simple_query_emb) * np.linalg.norm(doc_emb)
                                                        )
                                                        simple_similarities.append((sim, doc))
                                                except:
                                                    continue
                                        if simple_similarities:
                                            simple_similarities.sort(key=lambda x: x[0], reverse=True)
                                            top_simple = simple_similarities[:3]  # เอา 3 อันดับแรก
                                            print(f"   ✅ พบ {len(simple_similarities)} เอกสารด้วย query 'โหราศาสตร์'")
                                            for i, (sim, doc) in enumerate(top_simple):
                                                if sim > 0.10:  # threshold ต่ำสำหรับ fallback
                                                    source_info = f"[{collection_name}]"
                                                    if 'page' in doc:
                                                        source_info += f" หน้า {doc['page']}"
                                                    # ✅ ใช้ข้อมูลจาก ORIGINAL_DB_NAME เท่านั้น
                                                    doc_info = {
                                                        'text': doc.get('text', ''),  # ข้อมูลต้นฉบับจาก original database
                                                        'source': source_info,
                                                        'similarity': sim,
                                                        'collection': collection_name,
                                                        'doc_id': doc.get('_id'),
                                                        'fallback_query': True  # ระบุว่าเป็น fallback query
                                                    }
                                                    retrieved_docs.append(doc_info)
                                                    print(f"   ✅ เอกสาร fallback ที่ {i+1} (Similarity: {sim:.4f})")
                                    continue
                                
                                # 🆕 Apply Re-ranking / Boosting logic
                                # Extract entities from query for boosting
                                query_entities = extract_astro_entities(question)
                                query_planets = query_entities.get('planets', [])
                                
                                boosted_similarities = []
                                for score, doc in similarities:
                                    boost_score = score
                                    text = doc.get('text', '')
                                    
                                    # 1. Zodiac Boost (+0.2)
                                    if astrology_chart and astrology_chart.get('zodiac_sign'):
                                        zodiac_patterns = [
                                            f"ราศี{astrology_chart['zodiac_sign']}", 
                                            f"คนราศี{astrology_chart['zodiac_sign']}",
                                            f"ชาวราศี{astrology_chart['zodiac_sign']}"
                                        ]
                                        if any(p in text for p in zodiac_patterns):
                                            boost_score += 0.2
                                            
                                    # 2. Day of Week Boost (+0.15)
                                    if astrology_chart and astrology_chart.get('day_of_week'):
                                        if astrology_chart['day_of_week'] in text:
                                            boost_score += 0.15
                                            
                                    # 3. Planet Boost (+0.1)
                                    for planet in query_planets:
                                        # Check keywords for planet
                                        keywords = ASTRO_SYSTEM_ENTITIES.get(planet, [])
                                        if any(kw in text.lower() for kw in keywords):
                                            boost_score += 0.1
                                            break # Boost only once per planet
                                            
                                    boosted_similarities.append((boost_score, doc))
                                
                                similarities = boosted_similarities

                                # เรียงตาม similarity score
                                similarities.sort(key=lambda x: x[0], reverse=True)
                                
                                # 🆕 ดึงชื่อราศีจาก astrology_chart เพื่อใช้ในการกรองเอกสาร
                                target_zodiac_sign = None
                                if astrology_chart and astrology_chart.get('zodiac_sign'):
                                    target_zodiac_sign = astrology_chart['zodiac_sign']
                                    print(f"   🔍 จะกรองเอกสารให้มีเฉพาะข้อมูลเกี่ยวกับราศี: {target_zodiac_sign}")
                                
                                # 🆕 กรองเอกสารที่มี similarity > 0.5 ก่อน (ตามที่ผู้ใช้ต้องการ)
                                similarity_threshold = 0.5
                                high_similarity_docs = [(sim, doc) for sim, doc in similarities if sim > similarity_threshold]
                                print(f"   ✅ คำนวณ similarity สำเร็จ: {len(similarities)} เอกสาร (จาก {len(docs)} เอกสารทั้งหมด)")
                                print(f"   📊 เอกสารที่มี similarity > {similarity_threshold}: {len(high_similarity_docs)} เอกสาร")
                                
                                # แสดง similarity score ทั้งหมด (เฉพาะ 10 อันดับแรก)
                                if similarities:
                                    print(f"   📊 Similarity scores (10 อันดับแรก):")
                                    for i, (sim, _) in enumerate(similarities[:10], 1):
                                        print(f"      {i}. {sim:.4f}")
                                
                                # ถ้าไม่มีเอกสารที่มี similarity > 0.5 ให้ใช้ threshold ที่ต่ำกว่า
                                if not high_similarity_docs:
                                    print(f"   ⚠️ ไม่มีเอกสารที่มี similarity > {similarity_threshold} - ใช้ threshold ที่ต่ำกว่า (0.15)")
                                    similarity_threshold = 0.15
                                    high_similarity_docs = [(sim, doc) for sim, doc in similarities if sim > similarity_threshold]
                                    print(f"   📊 เอกสารที่มี similarity > {similarity_threshold}: {len(high_similarity_docs)} เอกสาร")
                                
                                # 🆕 กรองเอกสารตามราศี (ถ้ามีการกรอง) - จากเอกสารที่มี similarity > 0.5 (หรือ threshold ที่ต่ำกว่า)
                                filtered_docs = []
                                if target_zodiac_sign:
                                    # 🆕 เพิ่มการค้นหาให้ครอบคลุมมากขึ้น - ใช้ top 50 หรือทั้งหมดที่มี similarity > 0.5
                                    initial_top_n = min(50, len(high_similarity_docs))
                                    top_docs_for_zodiac_filter = high_similarity_docs[:initial_top_n]
                                    
                                    print(f"   🔍 กำลังกรองเอกสารที่เกี่ยวข้องกับราศี{target_zodiac_sign} จาก {len(top_docs_for_zodiac_filter)} เอกสาร...")
                                    
                                    for similarity, doc in top_docs_for_zodiac_filter:
                                        text_content = doc.get('text', '')
                                        if text_content:
                                            # 🆕 ปรับปรุงการตรวจสอบให้ครอบคลุมมากขึ้น - ตรวจสอบหลายรูปแบบ
                                            zodiac_patterns = [
                                                f"ราศี{target_zodiac_sign}",
                                                f"คนราศี{target_zodiac_sign}",
                                                f"ชาวราศี{target_zodiac_sign}",
                                                f"ราศี {target_zodiac_sign}",  # มีช่องว่าง
                                                f"คนราศี {target_zodiac_sign}",  # มีช่องว่าง
                                                f"ชาวราศี {target_zodiac_sign}",  # มีช่องว่าง
                                                target_zodiac_sign  # ชื่อราศีโดยตรง
                                            ]
                                            
                                            # 🆕 ตรวจสอบว่ามีชื่อราศีในเอกสารหรือไม่ (หลายรูปแบบ)
                                            contains_zodiac = any(pattern in text_content for pattern in zodiac_patterns)
                                            
                                            if contains_zodiac:
                                                filtered_docs.append((similarity, doc))
                                    
                                    print(f"   🔍 หลังกรองตามราศี{target_zodiac_sign}: พบ {len(filtered_docs)} เอกสาร (จาก {len(top_docs_for_zodiac_filter)} เอกสารที่มี similarity > {similarity_threshold})")
                                    
                                    # 🆕 Strict Filtering: ใช้เอกสารที่ผ่านเกณฑ์เท่านั้น ไม่พยายามดึงมาให้ครบจำนวน
                                    if filtered_docs:
                                        # เรียงตาม similarity จากสูงไปต่ำ
                                        filtered_docs.sort(key=lambda x: x[0], reverse=True)
                                        # ตัดเอกสารที่มีคะแนนต่ำเกินไปออก (Strict Cutoff)
                                        # เช่น ถ้าคะแนนต่ำกว่า 0.45 ให้ตัดทิ้งเลย แม้จะไม่ครบ 7 เอกสารก็ตาม
                                        strict_threshold_for_cutoff = 0.45
                                        top_docs = [doc for doc in filtered_docs if doc[0] >= strict_threshold_for_cutoff]
                                        
                                        # ถ้า after cutoff ยังเหลือเอกสาร ให้ใช้ top 7
                                        if top_docs:
                                            top_docs = top_docs[:7]
                                            print(f"   ✅ [Strict Filter] ใช้ {len(top_docs)} เอกสารที่ผ่านเกณฑ์ (Sim >= {strict_threshold_for_cutoff}) และเกี่ยวข้องกับราศี{target_zodiac_sign}")
                                        else:
                                            # ถ้าตัดแล้วไม่เหลือเลย ให้ใช้ตัวที่ดีที่สุดสัก 2-3 ตัวแทน (Fallback แบบ Minimal)
                                            # เพื่อกันไม่ให้ตอบว่า "ไม่รู้" เลยถ้ายังมีข้อมูลที่พอถูไถได้
                                            top_docs = filtered_docs[:3]
                                            print(f"   ⚠️ [Strict Filter] เอกสารคะแนนต่ำกว่าเกณฑ์ ({strict_threshold_for_cutoff}) แต่ขอยกเว้นให้ใช้ 3 อันดับแรก")
                                    else:
                                        # 🆕 ถ้าไม่มีเอกสารที่กรองแล้ว ให้ลองค้นหาใหม่ด้วย query ที่เฉพาะเจาะจงมากขึ้น
                                        print(f"   ⚠️ ไม่พบเอกสารที่เกี่ยวข้องกับราศี{target_zodiac_sign} จาก {len(top_docs_for_zodiac_filter)} เอกสาร")
                                        # Strict Fallback: ถ้าไม่เจอราศีที่ตรงเป๊ะ ให้ใช้ Top Docs ปกติ แต่ต้องคะแนนสูงจริง
                                        strict_general_threshold = 0.50
                                        top_docs = [doc for doc in high_similarity_docs if doc[0] >= strict_general_threshold]
                                        if top_docs:
                                            top_docs = top_docs[:5] # ลดจำนวนลงอีกถ้าไม่ใช่ราศีที่ตรง
                                            print(f"   🔄 [Fallback] ใช้ {len(top_docs)} เอกสารทั่วไปที่มีความมั่นใจสูง (Sim >= {strict_general_threshold})")
                                        else:
                                             top_docs = high_similarity_docs[:3] # Minimal Fallback
                                             print(f"   🔄 [Fallback] ใช้ 3 เอกสารที่ดีที่สุด (Best Effort)")
                                else:
                                    # ถ้าไม่มีการกรองตามราศี ให้ใช้ top 7 จากเอกสารที่มี similarity > 0.5
                                    # Strict Filtering เช่นกัน
                                    strict_general_threshold = 0.50
                                    top_docs = [doc for doc in high_similarity_docs if doc[0] >= strict_general_threshold]
                                    if top_docs:
                                        top_docs = top_docs[:7]
                                    else:
                                        top_docs = high_similarity_docs[:3]
                                
                                # 🆕 ใช้ threshold เดียวกันสำหรับการแสดงผล
                                threshold = similarity_threshold
                                
                                # แสดง similarity score สูงสุด
                                if top_docs:
                                    max_similarity = top_docs[0][0]
                                    min_similarity = top_docs[-1][0]
                                    print(f"   📊 Similarity score: สูงสุด = {max_similarity:.4f}, ต่ำสุด (top 5) = {min_similarity:.4f}, Threshold = {threshold}")
                                
                                for i, (similarity, doc) in enumerate(top_docs):
                                    # เพิ่มข้อมูล source
                                    source_info = f"[{collection_name}]"
                                    if 'page' in doc:
                                        source_info += f" หน้า {doc['page']}"
                                    if 'chunk_id' in doc:
                                        source_info += f" Chunk {doc['chunk_id']}"
                                    if 'type' in doc:
                                        source_info += f" ({doc['type']})"
                                    
                                    # ✅ ใช้ข้อมูลจาก ORIGINAL_DB_NAME เท่านั้น
                                    # ใช้ field 'text' จากเอกสารต้นฉบับ (ไม่ใช้ summary)
                                    text_content = doc.get('text', '')  # ข้อมูลต้นฉบับจาก original database
                                    
                                    # 🆕 Debug: แสดงความยาวของ text ที่ดึงมาจาก MongoDB
                                    text_length = len(text_content) if text_content else 0
                                    if i < 3:  # แสดงเฉพาะ 3 อันดับแรก
                                        print(f"   🔍 Debug (Retrieval): เอกสารที่ {i+1} - Similarity: {similarity:.4f}, ความยาว text ใน MongoDB: {text_length} ตัวอักษร")
                                        if text_length > 0:
                                            print(f"      📝 ตัวอย่าง text (200 ตัวอักษรแรก): {text_content[:200]}...")
                                            if text_length > 200:
                                                print(f"      📝 ตัวอย่าง text (200 ตัวอักษรสุดท้าย): ...{text_content[-200:]}")
                                    
                                    doc_info = {
                                        'text': text_content,  # ข้อมูลต้นฉบับจาก original database
                                        'source': source_info,
                                        'similarity': similarity,
                                        'collection': collection_name,
                                        'doc_id': doc.get('_id'),
                                        'page': doc.get('page'),  # 🆕 เก็บ page number จาก doc โดยตรง
                                        'chunk_id': doc.get('chunk_id')  # 🆕 เก็บ chunk_id จาก doc โดยตรง
                                    }
                                    
                                    if similarity > threshold:
                                        print(f"   ✅ เอกสารที่ {i+1} จาก {collection_name} (Similarity: {similarity:.4f}) - ผ่าน threshold ({threshold})")
                                        retrieved_docs.append(doc_info)
                                    else:
                                        # เพิ่มเอกสารที่ต่ำกว่า threshold เพื่อแสดงใน terminal
                                        doc_info['below_threshold'] = True
                                        retrieved_docs.append(doc_info)
                                        print(f"   ⚠️ เอกสารที่ {i+1} มี similarity ต่ำเกินไป: {similarity:.4f} < {threshold} (threshold)")
                        except Exception as e:
                            print(f"   ❌ ไม่สามารถค้นหาใน {collection_name} ได้: {e}")
                            import traceback
                            traceback.print_exc()
                            continue
                    
                    # สรุปผลการค้นหา
                    # 🆕 นับจำนวนเอกสารที่ผ่าน threshold จริงๆ
                    valid_count = sum(1 for doc in retrieved_docs if not doc.get('below_threshold', False))
                    print(f"✅ ดึงข้อมูลจาก MongoDB เสร็จสิ้น: พบ {len(retrieved_docs)} เอกสารทั้งหมด, {valid_count} เอกสารที่ผ่าน threshold")
                    
                    # ไม่ต้องปิด client ที่นี่ เพราะใช้ client จาก verify function
                    # จะปิดภายหลังเมื่อเสร็จสิ้นการใช้งาน
                    
                except Exception as retrieval_error:
                    print(f"   ❌ เกิดข้อผิดพลาดในการทำ retrieval: {retrieval_error}")
                    import traceback
                    traceback.print_exc()
                    retrieved_docs = []
                finally:
                    # ปิด connection หลังจากการใช้งานเสร็จสิ้น
                    if client:
                        try:
                            client.close()
                            logger.debug("Closed MongoDB connection after retrieval")
                        except:
                            pass
                
    except Exception as e:
        print(f"❌ ไม่สามารถค้นหาจาก MongoDB ได้: {e}")
        import traceback
        traceback.print_exc()
        pass
    
    # หมายเหตุ: รายงานสรุปจะพิมพ์หลังจากได้คำตอบแล้ว เพื่อรวมความยาวคำตอบด้วย
    
    # กรองเฉพาะเอกสารที่ผ่าน threshold (ไม่มี below_threshold flag)
    valid_retrieved_docs = [doc for doc in retrieved_docs if not doc.get('below_threshold', False)]
    
    # 🆕 Debug: แสดงจำนวนเอกสารที่กรองแล้ว
    print(f"\n🔍 Debug: จำนวนเอกสารทั้งหมด: {len(retrieved_docs)}, เอกสารที่ผ่าน threshold: {len(valid_retrieved_docs)}")
    if len(retrieved_docs) > 0 and len(valid_retrieved_docs) == 0:
        print(f"⚠️ Warning: มีเอกสาร {len(retrieved_docs)} เอกสาร แต่ไม่มีเอกสารที่ผ่าน threshold")
        print(f"   ตรวจสอบเอกสารที่ 1-5:")
        for i, doc in enumerate(retrieved_docs[:5], 1):
            similarity = doc.get('similarity', 'N/A')
            below_threshold = doc.get('below_threshold', False)
            print(f"   {i}. Similarity: {similarity}, below_threshold: {below_threshold}")
        
        # 🆕 ถ้ามีเอกสารแต่ไม่มีเอกสารที่ผ่าน threshold ให้ใช้เอกสารที่มี similarity สูงสุดแทน
        if len(retrieved_docs) > 0:
            # เรียงตาม similarity จากสูงไปต่ำ
            sorted_docs = sorted(retrieved_docs, key=lambda x: x.get('similarity', 0), reverse=True)
            # ใช้เอกสารที่มี similarity สูงสุด 5 อันดับแรก (แม้จะต่ำกว่า threshold)
            top_docs_fallback = sorted_docs[:5]
            print(f"   🔄 ใช้เอกสารที่มี similarity สูงสุด {len(top_docs_fallback)} เอกสารแทน (fallback mode)")
            valid_retrieved_docs = top_docs_fallback
            # ลบ flag below_threshold เพื่อให้ระบบใช้เอกสารเหล่านี้
            for doc in valid_retrieved_docs:
                doc.pop('below_threshold', None)
    
    # ตรวจสอบว่ามีเอกสารจาก MongoDB หรือไม่
    # 🆕 ระบบ RAG ต้องใช้ข้อมูลจาก MongoDB ในการตอบคำถาม (ใช้ cosine similarity)
    # ถ้าไม่พบข้อมูลจาก MongoDB ให้ return error message
    if not valid_retrieved_docs or len(valid_retrieved_docs) == 0:
        print("\n⚠️ ไม่พบข้อมูลจาก MongoDB - ระบบ RAG ต้องใช้ข้อมูลจาก MongoDB ในการตอบคำถาม")
        
        # แสดงรายงานบนเทอร์มินัลสำหรับ RAGAS
        answer = "ขออภัยค่ะ ไม่พบข้อมูลที่เกี่ยวข้องในฐานข้อมูลสำหรับคำถามนี้ กรุณาลองใช้คำถามที่เกี่ยวข้องกับโหราศาสตร์ เช่น 'นิสัยราศีเมถุนเป็นยังไง' หรือ 'สีมงคลราศีสิงห์' ค่ะ"
        
        try:
            print_ragas_terminal_report(
                question=question,
                retrieved_docs=retrieved_docs,  # ส่งทั้งเอกสารทั้งหมดรวมถึงที่ต่ำกว่า threshold เพื่อแสดงในรายงาน
                answer=answer,
                user_id=user_id,
            )
        except Exception:
            pass
        
        # บันทึก interaction
        try:
            context_data = {}
            if astrology_chart:
                context_data.update({
                    "zodiac_sign": astrology_chart.get('zodiac_sign'),
                    "zodiac_element": astrology_chart.get('zodiac_element'),
                    "zodiac_quality": astrology_chart.get('zodiac_quality'),
                    "birth_date": astrology_chart.get('birth_date'),
                    "birth_time": astrology_chart.get('birth_time'),
                })
            
            store_user_response(
                question=question,
                answer=answer,
                user_id=user_id,
                response_type="no_data_found",
                context_data=context_data
            )
        except Exception:
            pass
        
        return answer

    # ✅ ใช้ RAG system - ใช้ข้อมูลจาก MongoDB ที่ค้นหาด้วย cosine similarity
    query_vector = []
    # กำหนดธงสำหรับสร้างคำถามต่อเนื่องอัตโนมัติเมื่อมีข้อมูลวันเกิดในคำถาม
    should_create_chart = bool(birth_info_from_question and birth_info_from_question.get('date'))

    # ✅ ใช้ GPT กับข้อมูลจาก MongoDB (RAG system - ใช้ cosine similarity)
    try:
        from openai import OpenAI
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key or openai_key == "your-openai-api-key-here":
            # ถ้าไม่ตั้งค่า API key ให้ตอบแบบ fallback ทั่วไปแทนการเรียก LLM
            return "ขออภัยค่ะ ตอนนี้ระบบยังไม่พร้อมใช้งาน AI ภายนอก แต่คุณสามารถถามเกี่ยวกับราศีได้ตามปกติ เช่น 'นิสัยราศีเมถุนเป็นยังไง' หรือ 'สีมงคลราศีสิงห์'"
        client = OpenAI(api_key=openai_key)
        
        # ============================================================
        # ✅ ระบบ RAG: สร้าง context จากข้อมูลต้นฉบับเท่านั้น
        # ============================================================
        # ใช้ข้อมูลจาก ORIGINAL_DB_NAME (astrobot_original) เท่านั้น
        # - ใช้ cosine similarity กับ embeddings ที่สร้างจาก text ต้นฉบับ
        # - ใช้ field 'text' จากเอกสารต้นฉบับ (ไม่ใช้ summary)
        # - ใช้เฉพาะเอกสารที่ผ่าน threshold
        # ============================================================
        context_info = ""
        if valid_retrieved_docs:
            # 🆕 กรองเฉพาะเอกสารที่มี similarity > 0.5
            high_similarity_docs = [doc for doc in valid_retrieved_docs 
                                   if isinstance(doc, dict) and doc.get('similarity', 0) > 0.5]
            
            # 🆕 ถ้ามีการกรองตามราศี ให้เพิ่มเอกสารที่เกี่ยวข้องกับราศีนั้นๆ (แม้จะ similarity ต่ำกว่า 0.5 แต่เกี่ยวข้องกับราศี)
            if astrology_chart and astrology_chart.get('zodiac_sign'):
                target_zodiac = astrology_chart['zodiac_sign']
                # ค้นหาเอกสารที่เกี่ยวข้องกับราศีจากเอกสารทั้งหมด (แม้จะ similarity ต่ำกว่า 0.5)
                zodiac_related_docs = []
                for doc in valid_retrieved_docs:
                    if isinstance(doc, dict):
                        text_content = doc.get('text', '')
                        similarity = doc.get('similarity', 0)
                        if text_content:
                            zodiac_patterns = [
                                f"ราศี{target_zodiac}",
                                f"คนราศี{target_zodiac}",
                                f"ชาวราศี{target_zodiac}",
                                f"ราศี {target_zodiac}",
                                f"คนราศี {target_zodiac}",
                                f"ชาวราศี {target_zodiac}",
                                target_zodiac
                            ]
                            contains_zodiac = any(pattern in text_content for pattern in zodiac_patterns)
                            if contains_zodiac and similarity > 0.3:  # ใช้ threshold ที่ต่ำกว่า (0.3) สำหรับเอกสารที่เกี่ยวข้องกับราศี
                                # ตรวจสอบว่าเอกสารนี้ยังไม่อยู่ใน high_similarity_docs
                                if doc not in high_similarity_docs:
                                    zodiac_related_docs.append(doc)
                
                if zodiac_related_docs:
                    print(f"   🔍 พบเอกสารเพิ่มเติมที่เกี่ยวข้องกับราศี{target_zodiac}: {len(zodiac_related_docs)} เอกสาร (similarity > 0.3)")
                    # เพิ่มเอกสารที่เกี่ยวข้องกับราศีเข้าไปใน high_similarity_docs
                    high_similarity_docs.extend(zodiac_related_docs)
                    # เรียงตาม similarity จากสูงไปต่ำ
                    high_similarity_docs.sort(key=lambda x: x.get('similarity', 0) if isinstance(x, dict) else 0, reverse=True)
            
            if high_similarity_docs:
                # 🆕 รวม chunks ที่อยู่หน้าเดียวกันและ chunk_id ใกล้กัน (เพื่อแก้ปัญหาที่ chunks สั้นเกินไป)
                # เก็บ doc_id และ page เพื่อดึง chunks เพิ่มเติมจากหน้าเดียวกัน
                merged_context_docs = []
                processed_doc_ids = set()
                
                for doc in high_similarity_docs:
                    if isinstance(doc, dict):
                        doc_id = doc.get('doc_id')
                        source_info = doc.get('source', 'Unknown')
                        
                        # 🆕 ถ้า chunk สั้นมาก (< 100 ตัวอักษร) และมี doc_id ให้ลองดึง chunks เพิ่มเติมจากหน้าเดียวกัน
                        text_content = doc.get('text', '')
                        text_length = len(text_content) if text_content else 0
                        
                        # 🆕 ดึง page number จาก doc โดยตรง (ไม่ใช่จาก source_info)
                        page_num = doc.get('page')
                        
                        if text_length < 100 and doc_id and doc_id not in processed_doc_ids and page_num:
                            # พยายามดึง chunks เพิ่มเติมจากหน้าเดียวกัน
                            try:
                                # ถ้ามี page number ให้ดึง chunks เพิ่มเติมจากหน้าเดียวกัน
                                from pymongo import MongoClient
                                mongo_url = os.getenv("MONGO_URL")
                                if mongo_url:
                                    temp_client = MongoClient(mongo_url, serverSelectionTimeoutMS=5000)
                                    db = temp_client[ORIGINAL_DB_NAME]
                                    collection_name = doc.get('collection', 'original_text_chunks')
                                    collection = db[collection_name]
                                    
                                    # 🆕 Debug: แสดงข้อมูลก่อนดึง chunks
                                    print(f"   🔍 พยายามรวม chunks จากหน้า {page_num} (collection: {collection_name}, doc_id: {doc_id})")
                                    
                                    # ดึง chunks ทั้งหมดจากหน้าเดียวกัน
                                    page_docs = list(collection.find({'page': page_num}, {'text': 1, 'chunk_id': 1, 'page': 1, 'type': 1}).sort('chunk_id', 1))
                                    
                                    print(f"   🔍 พบ {len(page_docs)} chunks ในหน้า {page_num}")
                                    
                                    if len(page_docs) > 1:
                                        # รวม text จาก chunks ทั้งหมดในหน้าเดียวกัน
                                        merged_texts = []
                                        for page_doc in page_docs:
                                            page_text = page_doc.get('text', '')
                                            if page_text and page_text.strip():
                                                merged_texts.append(page_text.strip())
                                        
                                        if merged_texts:
                                            merged_text = " ".join(merged_texts)
                                            if len(merged_text) > text_length:
                                                # ใช้ merged text แทน
                                                doc['text'] = merged_text
                                                doc['merged_from_page'] = True
                                                print(f"   🔄 รวม chunks จากหน้า {page_num}: {len(page_docs)} chunks → {len(merged_text)} ตัวอักษร (เพิ่มขึ้น {len(merged_text) - text_length} ตัวอักษร)")
                                            else:
                                                print(f"   ⚠️ รวม chunks แล้วแต่ความยาวไม่เพิ่มขึ้น (เดิม: {text_length}, ใหม่: {len(merged_text)})")
                                        else:
                                            print(f"   ⚠️ ไม่มี text ใน chunks จากหน้า {page_num}")
                                    else:
                                        print(f"   ⚠️ มี chunks เพียง 1 chunk ในหน้า {page_num} (ไม่ต้องรวม)")
                                    
                                    temp_client.close()
                                else:
                                    print(f"   ⚠️ ไม่พบ MONGO_URL ใน environment variables")
                            except Exception as merge_error:
                                # ถ้าไม่สามารถรวมได้ ให้ใช้ text เดิม
                                print(f"   ⚠️ ไม่สามารถรวม chunks จากหน้าเดียวกันได้: {merge_error}")
                                pass
                            
                            processed_doc_ids.add(doc_id)
                        
                        merged_context_docs.append(doc)
                
                # 🆕 ตรวจสอบว่ามีเอกสารที่เกี่ยวข้องกับราศีหรือไม่
                zodiac_related_count = 0
                if astrology_chart and astrology_chart.get('zodiac_sign'):
                    target_zodiac = astrology_chart['zodiac_sign']
                    for doc in merged_context_docs:
                        if isinstance(doc, dict):
                            text_content = doc.get('text', '')
                            if text_content:
                                zodiac_patterns = [
                                    f"ราศี{target_zodiac}",
                                    f"คนราศี{target_zodiac}",
                                    f"ชาวราศี{target_zodiac}",
                                    f"ราศี {target_zodiac}",
                                    f"คนราศี {target_zodiac}",
                                    f"ชาวราศี {target_zodiac}",
                                    target_zodiac
                                ]
                                if any(pattern in text_content for pattern in zodiac_patterns):
                                    zodiac_related_count += 1
                    
                    print(f"   📊 เอกสารที่เกี่ยวข้องกับราศี{target_zodiac}: {zodiac_related_count}/{len(merged_context_docs)} เอกสาร")
                
                context_info = "\n\n**ข้อมูลที่เกี่ยวข้องจากฐานข้อมูลต้นฉบับ (ค้นหาด้วย cosine similarity จาก embeddings - แสดงเฉพาะเอกสารที่มี Similarity > 0.5):**\n"
                for i, doc in enumerate(merged_context_docs):
                    if isinstance(doc, dict):
                        # ✅ ใช้ text ต้นฉบับจาก ORIGINAL_DB_NAME เท่านั้น
                        similarity_score = doc.get('similarity', 0)
                        content_to_use = doc.get('text', '')  # ใช้ text ต้นฉบับเท่านั้น (ไม่ตัด)
                        source_info = doc.get('source', 'Unknown')
                        
                        # 🆕 Debug: แสดงความยาวของ text ที่ดึงมาจาก MongoDB
                        text_length = len(content_to_use) if content_to_use else 0
                        is_merged = doc.get('merged_from_page', False)
                        merge_indicator = " (รวมจากหลาย chunks)" if is_merged else ""
                        
                        # 🆕 ตรวจสอบว่ามีข้อมูลเกี่ยวกับราศีหรือไม่
                        zodiac_indicator = ""
                        if astrology_chart and astrology_chart.get('zodiac_sign'):
                            target_zodiac = astrology_chart['zodiac_sign']
                            zodiac_patterns = [
                                f"ราศี{target_zodiac}",
                                f"คนราศี{target_zodiac}",
                                f"ชาวราศี{target_zodiac}",
                                f"ราศี {target_zodiac}",
                                f"คนราศี {target_zodiac}",
                                f"ชาวราศี {target_zodiac}",
                                target_zodiac
                            ]
                            if any(pattern in content_to_use for pattern in zodiac_patterns):
                                zodiac_indicator = " ✅ เกี่ยวข้องกับราศี"
                        
                        print(f"   🔍 Debug: เอกสารที่ {i+1} - Similarity: {similarity_score:.4f}, ความยาว text: {text_length} ตัวอักษร{merge_indicator}{zodiac_indicator}")
                        if text_length > 0:
                            print(f"      📝 ตัวอย่าง text (100 ตัวอักษรแรก): {content_to_use[:100]}...")
                            print(f"      📝 ตัวอย่าง text (100 ตัวอักษรสุดท้าย): ...{content_to_use[-100:]}")
                        
                        context_info += f"{i+1}. [Similarity: {similarity_score:.4f}] {source_info}{merge_indicator}{zodiac_indicator}\n"
                        context_info += f"   Context: {content_to_use}\n\n"  # 🆕 แสดงทั้งหมดไม่ตัด
                    else:
                        context_info += f"{i+1}. {doc}\n\n"  # 🆕 แสดงทั้งหมดไม่ตัด
                print(f"✅ ใช้ข้อมูลจาก MongoDB (RAG): {len(merged_context_docs)} เอกสาร (จาก {len(valid_retrieved_docs)} เอกสารทั้งหมด, กรองเฉพาะที่มี Similarity > 0.5)")
                
                # 🆕 แสดง context_info ที่ส่งให้ GPT ใน terminal
                print(f"\n{'='*60}")
                print(f"📋 Context ที่ส่งให้ GPT (แสดงทั้งหมด):")
                print(f"{'='*60}")
                print(context_info)
                print(f"{'='*60}\n")
                
                # 🆕 แสดง chart_info ที่ส่งให้ GPT ใน terminal (เพื่อตรวจสอบว่ามีข้อมูลอะไรบ้าง)
                if chart_info:
                    print(f"\n{'='*60}")
                    print(f"⚠️ Chart Info ที่ส่งให้ GPT (ห้ามใช้ในการตอบคำถาม - ใช้เฉพาะ context_info เท่านั้น):")
                    print(f"{'='*60}")
                    print(chart_info[:500] + "..." if len(chart_info) > 500 else chart_info)
                    print(f"{'='*60}\n")
            else:
                # ถ้าไม่มีเอกสารที่มี similarity > 0.5 แต่มีเอกสารอื่น ให้ใช้เอกสารที่มี similarity สูงสุด
                if valid_retrieved_docs:
                    sorted_docs = sorted(valid_retrieved_docs, 
                                       key=lambda x: x.get('similarity', 0) if isinstance(x, dict) else 0, 
                                       reverse=True)
                    top_docs = sorted_docs[:3]  # ใช้ 3 อันดับแรก
                    context_info = "\n\n**ข้อมูลที่เกี่ยวข้องจากฐานข้อมูลต้นฉบับ (ค้นหาด้วย cosine similarity จาก embeddings - ใช้เอกสารที่มี Similarity สูงสุด 3 อันดับแรก):**\n"
                    for i, doc in enumerate(top_docs):
                        if isinstance(doc, dict):
                            similarity_score = doc.get('similarity', 0)
                            content_to_use = doc.get('text', '')
                            source_info = doc.get('source', 'Unknown')
                            
                            # 🆕 Debug: แสดงความยาวของ text ที่ดึงมาจาก MongoDB
                            text_length = len(content_to_use) if content_to_use else 0
                            print(f"   🔍 Debug: เอกสารที่ {i+1} (fallback) - Similarity: {similarity_score:.4f}, ความยาว text: {text_length} ตัวอักษร")
                            if text_length > 0:
                                print(f"      📝 ตัวอย่าง text (100 ตัวอักษรแรก): {content_to_use[:100]}...")
                                print(f"      📝 ตัวอย่าง text (100 ตัวอักษรสุดท้าย): ...{content_to_use[-100:]}")
                            
                            context_info += f"{i+1}. [Similarity: {similarity_score:.4f}] {source_info}\n"
                            context_info += f"   Context: {content_to_use}\n\n"  # 🆕 แสดงทั้งหมดไม่ตัด
                    print(f"⚠️ ไม่มีเอกสารที่มี Similarity > 0.5 - ใช้ข้อมูลจาก MongoDB (RAG): {len(top_docs)} เอกสาร (เอกสารที่มี Similarity สูงสุด)")
                    
                    # 🆕 แสดง context_info ที่ส่งให้ GPT ใน terminal
                    print(f"\n{'='*60}")
                    print(f"📋 Context ที่ส่งให้ GPT (แสดงทั้งหมด):")
                    print(f"{'='*60}")
                    print(context_info)
                    print(f"{'='*60}\n")
        else:
            # ถ้าไม่มีข้อมูลจาก MongoDB แต่มี chart_info ให้ใช้ข้อมูลจาก chart_info เป็น fallback
            if astrology_chart and astrology_chart.get('zodiac_sign'):
                print(f"⚠️ ไม่พบข้อมูลจาก MongoDB แต่มีข้อมูลวันเกิด - จะใช้ข้อมูลจาก chart_info เป็น fallback")
                context_info = "\n\n**ข้อมูลจากวันเกิด (fallback - ไม่พบข้อมูลจาก MongoDB):**\n"
                context_info += f"ราศี: {astrology_chart.get('zodiac_sign')}\n"
                if astrology_chart.get('detailed_reading'):
                    detailed = astrology_chart['detailed_reading']
                    context_info += f"ลักษณะนิสัย: {detailed.get('ลักษณะนิสัย', 'ไม่มีข้อมูล')[:200]}...\n"
                    context_info += f"การงาน: {detailed.get('การงาน', 'ไม่มีข้อมูล')[:200]}...\n"
                    context_info += f"การเงิน: {detailed.get('การเงิน', 'ไม่มีข้อมูล')[:200]}...\n"
                    context_info += f"ความรัก: {str(detailed.get('ความรัก', 'ไม่มีข้อมูล'))[:200]}...\n"
        
        # สร้างข้อมูลดวงชะตาเพิ่มเติม
        chart_info = ""
        if astrology_chart:
            # ข้อมูลสถานที่เกิด
            location_info = ""
            if 'birth_location_name' in astrology_chart:
                location_info = f"สถานที่เกิด: {astrology_chart['birth_location_name']}\n"
            elif 'birth_location' in astrology_chart:
                location_info = f"สถานที่เกิด: กรุงเทพฯ\n"
            
            chart_info = f"""
**ข้อมูลดวงชะตาจากวันเกิดและเวลาเกิด:**
ราศีเกิด: {astrology_chart['zodiac_sign']} ({astrology_chart['zodiac_english']})
**คำสั่งสำคัญ: ต้องใช้ชื่อ "ราศี{astrology_chart['zodiac_sign']}" เท่านั้น ห้ามใช้คำว่า "ราศีปลา" หรือชื่อสัตว์อื่นๆ**

**ตัวอย่างการใช้งานที่ถูกต้อง:**
- ราศี{astrology_chart['zodiac_sign']} มีลักษณะอ่อนโยน
- คนราศี{astrology_chart['zodiac_sign']} มักจะ...
- ราศี{astrology_chart['zodiac_sign']} เป็นราศีธาตุ{astrology_chart['zodiac_element']}

**คำสั่งเด็ดขาด: ห้ามใช้คำว่า "ราศีปลา" ในคำตอบเด็ดขาด ต้องใช้ "ราศี{astrology_chart['zodiac_sign']}" เท่านั้น**

**ข้อมูลเพิ่มเติม:**
- ราศี{astrology_chart['zodiac_sign']} เป็นราศีสุดท้ายของจักรราศี
- ราศี{astrology_chart['zodiac_sign']} มีธาตุ{astrology_chart['zodiac_element']}

**คำสั่งสำคัญ:**
- ต้องใช้คำว่า "ลัคณา" แทน "Ascendant" ในทุกกรณี
- ต้องใช้ชื่อ "ราศี{astrology_chart['zodiac_sign']}" เท่านั้น ห้ามใช้ "ราศีปลา"
- ต้องใช้ชื่อ "ราศี{astrology_chart['zodiac_sign']}" ในทุกกรณี ห้ามใช้ชื่ออื่น

**ตัวอย่างการใช้งานที่ถูกต้อง:**
- ราศี{astrology_chart['zodiac_sign']} มีลักษณะอ่อนโยน
- ลัคณาคือราศี{astrology_chart['zodiac_sign']}
- คนราศี{astrology_chart['zodiac_sign']} มักจะ...
ธาตุ: {astrology_chart['zodiac_element']}
วันเกิด: {astrology_chart['birth_date']}
เวลาเกิด: {astrology_chart['birth_time'] if astrology_chart['birth_time'] else 'ไม่ระบุ'}{location_info}อายุ: {astrology_chart['age']} ปี

การตีความดวงชะตา:
- ราศี{astrology_chart['zodiac_sign']} เป็นราศีธาตุ{astrology_chart['zodiac_element']}
- ลักษณะเด่นของราศี{astrology_chart['zodiac_sign']} คือ{astrology_chart.get('detailed_reading', {}).get('ลักษณะนิสัย', 'มีเอกลักษณ์เฉพาะตัว')[:50]}...
"""

            # เพิ่มข้อมูลลัคณาถ้ามี
            if 'ascendant' in astrology_chart:
                ascendant = astrology_chart['ascendant']
                
                chart_info += f"""

**ข้อมูลลัคณา (ราศีประจำลัคนา):**
ลัคณา: ราศี{ascendant['sign']} {ascendant['degree']:.1f}° ({ascendant['element']})
การตีความลัคณา: {astrology_chart.get('ascendant_interpretation', 'ไม่มีข้อมูล')}

หมายเหตุ: ลัคณาเป็นราศีประจำลัคนาที่แสดงบุคลิกภาพภายนอกและวิธีการที่ผู้อื่นมองเห็นคุณ
"""

            # 🆕 ลบ detailed_reading ออกจาก chart_info เพื่อบังคับให้ใช้ข้อมูลจาก RAG (context_info) เท่านั้น
            # ไม่เพิ่ม detailed_reading ใน chart_info เพราะต้องการให้ใช้ข้อมูลจาก RAG retrieval เท่านั้น
            
            # เพิ่มข้อมูลสีมงคลถ้ามี
            if 'lucky_colors' in astrology_chart and astrology_chart['lucky_colors']:
                lucky_colors = astrology_chart['lucky_colors']
                bad_colors = astrology_chart.get('bad_colors', [])
                chart_info += f"""

**สีมงคลสำหรับราศี{astrology_chart['zodiac_sign']}:**
สีมงคล: {', '.join(lucky_colors) if isinstance(lucky_colors, list) else lucky_colors}
"""
                if bad_colors:
                    chart_info += f"สีที่ควรหลีกเลี่ยง: {', '.join(bad_colors) if isinstance(bad_colors, list) else bad_colors}\n"


        # สร้าง prompt สำหรับแชทบอทโหราศาสตร์ตะวันตก
        # กำหนดการตอบตามเจตนาของคำถาม
        focus_instruction = ""
        if question_intent["specific_topic"] == "personality":
            focus_instruction = """
**คำสั่งสำคัญ: ตอบเฉพาะเรื่องลักษณะนิสัยและบุคลิกภาพเท่านั้น**
- ห้ามตอบเรื่องความรัก การงาน การเงิน สุขภาพ หรือสีมงคล
- เน้นที่ลักษณะนิสัย จุดแข็ง จุดอ่อน และบุคลิกภาพเฉพาะตัว
- อธิบายว่าทำไมราศีนี้จึงมีลักษณะนิสัยแบบนี้
"""
        elif question_intent["specific_topic"] == "love":
            focus_instruction = """
**คำสั่งสำคัญ: ตอบเฉพาะเรื่องความรักและความสัมพันธ์เท่านั้น**
- ห้ามตอบเรื่องลักษณะนิสัย การงาน การเงิน สุขภาพ หรือสีมงคล
- เน้นที่ความรัก ความสัมพันธ์ และการเข้ากันได้กับคนอื่น
- ให้คำแนะนำเรื่องความรักสำหรับคนโสดและคนมีคู่
"""
        elif question_intent["specific_topic"] == "career":
            focus_instruction = """
**คำสั่งสำคัญ: ตอบเฉพาะเรื่องอาชีพและการงานเท่านั้น**
- ห้ามตอบเรื่องลักษณะนิสัย ความรัก การเงิน สุขภาพ หรือสีมงคล
- เน้นที่อาชีพที่เหมาะ การทำงาน และความสำเร็จในหน้าที่การงาน
- ให้คำแนะนำเรื่องการเลือกอาชีพและการพัฒนาตนเอง
"""
        elif question_intent["specific_topic"] == "health":
            focus_instruction = """
**คำสั่งสำคัญ: ตอบเฉพาะเรื่องสุขภาพและการดูแลร่างกายเท่านั้น**
- ห้ามตอบเรื่องลักษณะนิสัย ความรัก การงาน การเงิน หรือสีมงคล
- เน้นที่การดูแลสุขภาพ จุดอ่อนด้านสุขภาพ และการป้องกันโรค
- ให้คำแนะนำเรื่องการออกกำลังกายและการดูแลร่างกาย
"""
        elif question_intent["specific_topic"] == "finance":
            focus_instruction = """
**คำสั่งสำคัญ: ตอบเฉพาะเรื่องการเงินและการลงทุนเท่านั้น**
- ห้ามตอบเรื่องลักษณะนิสัย ความรัก การงาน สุขภาพ หรือสีมงคล
- เน้นที่การจัดการเงิน การลงทุน และการสร้างความมั่งคั่ง
- ให้คำแนะนำเรื่องการออมและการลงทุน
"""
        elif question_intent["specific_topic"] == "lucky_colors":
            focus_instruction = """
**คำสั่งสำคัญ: ตอบเฉพาะเรื่องสีมงคลเท่านั้น**
- ห้ามตอบเรื่องลักษณะนิสัย ความรัก การงาน การเงิน หรือสุขภาพ
- เน้นที่สีที่เหมาะ สีที่ควรหลีกเลี่ยง และความหมายของสี
- อธิบายว่าทำไมสีเหล่านี้จึงเหมาะกับราศีนี้
"""
        else:
            # 🆕 เมื่อมีวันเกิดในคำถาม ให้ตอบครบทั้ง 4 ด้าน: การงาน การเงิน ความรัก สีมงคล
            if birth_info_from_question and birth_info_from_question.get('date'):
                focus_instruction = """
**⚠️ คำสั่งสำคัญ: เมื่อคำถามมีวันเดือนปีเกิด ต้องตอบครบทั้ง 4 ด้านเสมอ (ห้ามขาดด้านใดด้านหนึ่ง):**

1. **ด้านการงาน (บังคับ):** 
   - ให้ข้อมูลเกี่ยวกับอาชีพที่เหมาะกับราศีนี้
   - การทำงานและความสำเร็จในหน้าที่การงาน
   - ทักษะที่โดดเด่นและจุดแข็งในการทำงาน
   - อาชีพที่ควรพิจารณา

2. **ด้านการเงิน (บังคับ):**
   - ให้ข้อมูลเกี่ยวกับการจัดการเงิน
   - การลงทุนและการออมที่เหมาะ
   - การสร้างความมั่งคั่ง
   - แนวทางการบริหารการเงิน

3. **ด้านความรัก (บังคับ):**
   - ให้ข้อมูลเกี่ยวกับความสัมพันธ์
   - การเข้ากันได้กับคนอื่น
   - คำแนะนำสำหรับคนโสด
   - คำแนะนำสำหรับคนมีคู่
   - ราศีที่เข้ากันได้ดี

4. **สีมงคล (บังคับ):**
   - ให้ข้อมูลเกี่ยวกับสีที่เหมาะกับราศีนี้
   - สีที่ควรหลีกเลี่ยง
   - ความหมายของสีแต่ละสี
   - สีที่ควรใช้ในชีวิตประจำวัน

**ข้อกำหนดเพิ่มเติม:**
- เริ่มต้นด้วยการระบุวันเกิดและราศีเกิดอย่างชัดเจน
- **ต้องตอบครบทั้ง 4 ด้านเสมอ** (การงาน, การเงิน, ความรัก, สีมงคล) ห้ามขาดด้านใดด้านหนึ่ง
- ใช้ข้อมูลจากฐานข้อมูล (MongoDB) ในการตอบคำถาม
- ห้ามตอบเรื่องสุขภาพ
- ตอบเป็นข้อความต่อเนื่องแบบธรรมชาติ ไม่ใช้หัวข้อหรือหมวดหมู่
"""
            else:
                focus_instruction = """
**คำสั่งสำคัญ: สำหรับคำถามเกี่ยวกับดวงชะตาโดยรวม ต้องตอบครบทั้ง 4 ด้านเสมอ**
- **ด้านการงาน:** ให้ข้อมูลเกี่ยวกับอาชีพที่เหมาะ การทำงาน ความสำเร็จในหน้าที่การงาน และทักษะที่โดดเด่น
- **ด้านการเงิน:** ให้ข้อมูลเกี่ยวกับการจัดการเงิน การลงทุน การออม และการสร้างความมั่งคั่ง
- **ด้านความรัก:** ให้ข้อมูลเกี่ยวกับความสัมพันธ์ การเข้ากันได้กับคนอื่น สำหรับคนโสดและคนมีคู่
- เริ่มต้นด้วยการระบุวันเกิดและราศีเกิดอย่างชัดเจน
- ห้ามตอบเรื่องสุขภาพหรือสีมงคล
- ต้องตอบครบทั้ง 4 ด้านเพื่อให้คำทำนายที่สมบูรณ์
"""

        # สร้าง astrology_prompt ที่เหมาะสม
        if astrology_chart:
            astrology_prompt = f"""คุณเป็นโหราจารย์ดิจิทัลผู้เชี่ยวชาญด้านโหราศาสตร์ตะวันตก (Western Astrology) ที่มีความรู้ลึกซึ้งเกี่ยวกับดาวเคราะห์ ราศี และการตีความดวงกำเนิด

**บทบาทและความเชี่ยวชาญ:**
- คุณเป็นระบบ RAG (Retrieval-Augmented Generation) ที่ใช้ข้อมูลจากฐานข้อมูล MongoDB ในการตอบคำถาม
- ข้อมูลที่ใช้ตอบคำถามถูกค้นหาด้วย cosine similarity จาก embeddings ที่สร้างไว้แล้ว
- คุณมีความเข้าใจในพลังของราศีเกิด และลัคณา (ราศีประจำลัคนา)
- คุณสามารถผสานข้อมูลจากฐานความรู้ (MongoDB) เพื่อสร้างคำทำนายที่เฉพาะตัวและแม่นยำ
- คุณให้คำแนะนำที่อบอุ่น เป็นมิตร และให้กำลังใจ
- คุณสามารถรักษาบริบทการสนทนาและตอบคำถามต่อเนื่องได้อย่างเป็นธรรมชาติ

**⚠️ ข้อกำหนดสำคัญสำหรับ RAG System (บังคับปฏิบัติตามอย่างเคร่งครัด):**
- **🚨 ห้ามใช้ความรู้จาก training data หรือความรู้ภายนอกใดๆ ทั้งสิ้น**
- **🚨 ต้องใช้ข้อมูลจากฐานข้อมูล (MongoDB) เท่านั้น** ในการตอบคำถาม
- **🚨 ห้ามสร้างข้อมูลหรือความรู้ใหม่ขึ้นมาเอง** ต้องใช้เฉพาะข้อมูลที่แสดงใน "ข้อมูลที่เกี่ยวข้องจากฐานข้อมูล"
- ข้อมูลที่แสดงใน "ข้อมูลที่เกี่ยวข้องจากฐานข้อมูล" ถูกค้นหาด้วย cosine similarity จาก embeddings
- **🚨 ถ้ามีข้อมูลใน "ข้อมูลที่เกี่ยวข้องจากฐานข้อมูล" ต้องใช้ข้อมูลนั้นในการตอบคำถามทันที ห้ามบอกว่า "ไม่พบข้อมูล"**
- **🚨 ถ้าไม่มีข้อมูลในฐานข้อมูลที่เกี่ยวข้องกับคำถามจริงๆ (ไม่มีข้อมูลใน "ข้อมูลที่เกี่ยวข้องจากฐานข้อมูล" เลย) ให้บอกว่า "ขออภัยค่ะ ไม่พบข้อมูลที่เกี่ยวข้องในฐานข้อมูลสำหรับคำถามนี้"**
- **🚨 ห้ามใช้ความรู้ทั่วไปเกี่ยวกับโหราศาสตร์ที่ไม่ได้มาจากฐานข้อมูล**
- **🚨 ต้องอ้างอิงและใช้ข้อมูลจากฐานข้อมูลเท่านั้น** ในการสร้างคำตอบ
- **🚨 ถ้ามีข้อมูลจากฐานข้อมูล ต้องใช้ข้อมูลนั้นในการตอบคำถามเท่านั้น ไม่ใช่สร้างคำตอบขึ้นมาเอง**

**ข้อกำหนดสำคัญ:**
- ใช้ชื่อราศีแบบไทยเท่านั้น: เมษ, พฤษภ, เมถุน, กรกฎ, สิงห์, กันย์, ตุล, พิจิก, ธนู, มังกร, กุมภ์, มีน
- ห้ามใช้ชื่อราศีแบบอังกฤษ เช่น Aries, Taurus, Gemini, Cancer, Leo, Virgo, Libra, Scorpio, Sagittarius, Capricorn, Aquarius, Pisces
- ห้ามใช้ชื่อสัตว์ เช่น ราศีปลา, ราศีแกะ, ราศีวัว, ราศีปู, ราศีสิงโต, ราศีแมงป่อง
- สำหรับราศีที่ 12 ต้องใช้ "ราศีมีน" เท่านั้น ห้ามใช้ "ราศีปลา" หรือ "Pisces"
- ใช้คำว่า "ลัคณา" แทน "Ascendant" ในทุกกรณี

{focus_instruction}

**ข้อมูลสำหรับการวิเคราะห์:**
{birth_info}
{chart_info}
{context_info}

**บริบทการสนทนาก่อนหน้า:**
{get_conversation_context(user_context)}

**คำถามของผู้ใช้:** {question}

**🚨 ข้อกำหนดสำคัญในการตอบคำถาม (อ่านให้ละเอียด):**
- **กฎสำคัญที่สุด: เมื่อคำถามมีวันเดือนปีเกิด (เช่น "07/09/2003", "ทำนายดวง", "ราศีอะไร" พร้อมวันเกิด) → ต้องตอบครบทั้ง 4 ด้านเสมอ (การงาน, การเงิน, ความรัก, สีมงคล) ห้ามขาดด้านใดด้านหนึ่ง**
- **วิเคราะห์คำถามให้ดีก่อนตอบ:**
  * **ถ้าถาม "ทำนายดวง" หรือมีวันเดือนปีเกิด → ต้องตอบครบทั้ง 4 ด้าน (การงาน, การเงิน, ความรัก, สีมงคล)**
  * **ถ้าถาม "ราศีอะไร" พร้อมวันเกิด → ต้องตอบครบทั้ง 4 ด้าน (การงาน, การเงิน, ความรัก, สีมงคล) ไม่ใช่แค่บอกชื่อราศี**
  * ถ้าถามว่า "เข้ากับราศีอะไร" หรือ "เข้ากันได้กับราศีอะไร" → ต้องตอบว่าควรเข้ากับราศีอะไร (เช่น ราศีเมษเข้ากับราศีสิงห์ได้ดี)
  * ถ้าถามว่า "อาชีพที่เหมาะ" หรือ "งานที่เหมาะ" → ต้องตอบว่าอาชีพอะไรที่เหมาะกับราศี
  * ถ้าถามว่า "นิสัยเป็นยังไง" → ต้องตอบว่าลักษณะนิสัยของราศีนั้น
  * ถ้าถามว่า "สีมงคล" → ต้องตอบว่าสีอะไรที่เป็นมงคล
- **ห้ามสับสนระหว่างคำถาม** เช่น ถ้าถาม "เข้ากับราศีอะไร" ห้ามตอบว่า "อาชีพที่เหมาะ" หรือ "ลักษณะนิสัย"
- **ตอบให้ตรงประเด็น** แต่ถ้ามีวันเดือนปีเกิด ต้องตอบครบทั้ง 4 ด้านเสมอ

**วิธีการตอบคำถาม (RAG System - บังคับปฏิบัติตามอย่างเคร่งครัด):**
1. **🚨 ต้องใช้ข้อมูลจากฐานข้อมูล (MongoDB) เท่านั้น** - ข้อมูลที่แสดงใน "ข้อมูลที่เกี่ยวข้องจากฐานข้อมูล" ถูกค้นหาด้วย cosine similarity จาก embeddings
2. **🚨 ห้ามใช้ความรู้จาก training data หรือความรู้ภายนอกใดๆ** - ต้องใช้เฉพาะข้อมูลที่แสดงใน "ข้อมูลที่เกี่ยวข้องจากฐานข้อมูล" เท่านั้น
3. **🚨 ถ้ามีข้อมูลใน "ข้อมูลที่เกี่ยวข้องจากฐานข้อมูล" ต้องใช้ข้อมูลนั้นในการตอบคำถามทันที ห้ามบอกว่า "ไม่พบข้อมูล"**
4. **🚨 ถ้าไม่มีข้อมูลในฐานข้อมูลที่เกี่ยวข้องกับคำถามจริงๆ (ไม่มีข้อมูลใน "ข้อมูลที่เกี่ยวข้องจากฐานข้อมูล" เลย) ให้บอกว่า "ขออภัยค่ะ ไม่พบข้อมูลที่เกี่ยวข้องในฐานข้อมูลสำหรับคำถามนี้"**
5. **⚠️ สำหรับคำถามที่มีวันเดือนปีเกิด (บังคับ):** 
   - **ต้องตอบครบทั้ง 4 ด้านเสมอ** (การงาน, การเงิน, ความรัก, สีมงคล) ห้ามขาดด้านใดด้านหนึ่ง
   - เริ่มต้นด้วยการระบุวันเกิดและราศีเกิดอย่างชัดเจน
   - **ใช้ข้อมูลจากฐานข้อมูลเท่านั้น** ในการตอบคำถาม
   - ตอบเป็นข้อความต่อเนื่องแบบธรรมชาติ ไม่ใช้หัวข้อหรือหมวดหมู่
   - ต้องครอบคลุมทั้ง 4 ด้าน: การงาน, การเงิน, ความรัก, สีมงคล
   - **ถ้าไม่มีข้อมูลในฐานข้อมูลสำหรับด้านใดด้านหนึ่ง ให้บอกว่า "ไม่พบข้อมูลในฐานข้อมูลสำหรับด้านนี้"**
5. **สำหรับคำถามทั่วไปเกี่ยวกับดวงชะตา (ไม่มีวันเกิด):** ต้องตอบครบทั้ง 4 ด้าน (ลักษณะนิสัยและบุคลิกภาพ, การงาน, การเงิน, ความรัก) โดยใช้ข้อมูลจากฐานข้อมูลเท่านั้น
6. **สำหรับคำถามเฉพาะด้าน:** ตอบเฉพาะด้านที่ถามเท่านั้น โดยใช้ข้อมูลจากฐานข้อมูลเท่านั้น (ถ้าถามเกี่ยวกับการงาน ก็ตอบเฉพาะการงาน เท่านั้น)
7. **สำหรับคำถามเกี่ยวกับความเข้ากันได้ของราศี:** ต้องตอบว่าควรเข้ากับราศีอะไร โดยใช้ข้อมูลจากฐานข้อมูลเท่านั้น
8. **สำหรับคำถามต่อเนื่อง:** ใช้ข้อมูลราศีที่มีอยู่แล้วและตอบคำถามเฉพาะเจาะจง โดยใช้ข้อมูลจากฐานข้อมูลเท่านั้น
9. **🚨 อ้างอิงข้อมูลจากฐานข้อมูลเท่านั้น** - ต้องอ้างอิงและใช้ข้อมูลจาก "ข้อมูลที่เกี่ยวข้องจากฐานข้อมูล" ในการสร้างคำตอบ ไม่ใช่สร้างคำตอบขึ้นมาเอง
10. อธิบายลักษณะนิสัยตามราศีและธาตุ โดยอ้างอิงจากข้อมูลในฐานความรู้ (MongoDB) เท่านั้น
11. **หากมีข้อมูล Ascendant:** ใช้ข้อมูล Ascendant เพื่อเพิ่มความแม่นยำในการทำนายบุคลิกภาพ (แต่ต้องใช้ข้อมูลจากฐานข้อมูลเท่านั้น)
12. ใช้ภาษาที่เป็นธรรมชาติ อ่อนโยน และเข้าใจง่าย
13. หลีกเลี่ยงคำทำนายเชิงโชคชะตาเด็ดขาด ใช้คำว่า "มีแนวโน้ม", "สะท้อนว่า", "บ่งบอกถึงพลังของ..."
14. ตอบเป็นข้อความต่อเนื่องแบบธรรมชาติ ไม่ใช้หัวข้อหรือหมวดหมู่
15. ห้ามใช้ emoji หรือสัญลักษณ์พิเศษใดๆ
16. **สำหรับคำถามต่อเนื่อง:** อย่าเปลี่ยนราศีหรือข้อมูลวันเกิด ให้ใช้ข้อมูลเดิมที่ผู้ใช้ให้มา

**การจัดการคำถามต่อเนื่อง:**
- ถ้าผู้ใช้ถามเกี่ยวกับ "ราศีนี้", "นิสัย", "ลักษณะ", "คนราศีนี้" โดยไม่ระบุราศี ให้ใช้ราศีจากข้อมูลบริบท
- ถ้าผู้ใช้ถามคำถามทั่วไปเกี่ยวกับโหราศาสตร์ ให้เชื่อมโยงกับราศีของเขา
- **ห้ามสร้างข้อมูลวันเกิดหรือราศีใหม่** สำหรับคำถามต่อเนื่อง
- **ห้ามเปลี่ยนราศี** จากที่ผู้ใช้ถามมาแล้ว
- รักษาบริบทการสนทนาให้ต่อเนื่องและเป็นธรรมชาติ
- **ใช้ข้อมูลการสนทนาก่อนหน้า** เพื่อให้คำตอบที่สอดคล้องและต่อเนื่อง
- **อย่าทำซ้ำข้อมูล** ที่ได้ให้ไปแล้วในคำตอบก่อนหน้า
- **ตอบคำถามเฉพาะเจาะจง** ตามที่ผู้ใช้ถาม โดยไม่ต้องอธิบายข้อมูลพื้นฐานซ้ำ

**น้ำเสียงและสไตล์:**
- ใช้โทนอบอุ่น ให้ผู้อ่านรู้สึกได้รับคำแนะนำจากผู้รู้ใจ
- ไม่ใช้ศัพท์โหราศาสตร์มากเกินไป แต่รักษาโทนเชิงจิตวิญญาณ
- ให้ความรู้สึกเหมือนโหราจารย์ผู้เข้าใจใจผู้อ่านจริงๆ
- สำหรับคำถามต่อเนื่อง ให้รู้สึกเหมือนการสนทนาต่อเนื่อง ไม่ใช่การเริ่มต้นใหม่
- **คำลงท้ายต้องใช้ "ค่ะ" เท่านั้น ห้ามใช้ "ครับ/ค่ะ" หรือ "ครับ"**

**การจัดการข้อมูลที่ไม่ครบ:**
- **หากไม่มีข้อมูลวันเกิดหรือราศีในคำถาม:**
  - ห้ามสร้างข้อมูลราศีหรือวันเกิดใหม่
  - ห้ามแจ้งเตือนผู้ใช้ในเนื้อหาของคำตอบ
  - ให้ส่งคำตอบแบบปกติโดยใช้ข้อมูลที่มีอยู่เท่านั้น
- หากมีข้อมูลบางส่วนไม่ครบ ให้ใช้ความรู้โหราศาสตร์ทั่วไปในการให้คำแนะนำ
- ห้ามใช้ข้อความเช่น "ไม่มีข้อมูลเพิ่มเติม", "ไม่สามารถให้คำแนะนำเฉพาะได้", "ข้อมูลไม่เพียงพอ" ในคำตอบ
- **หากมีข้อมูลดวงชะตาแล้ว ให้ใช้ข้อมูลนั้นในการตอบคำถามทันที ไม่ต้องแจ้งเตือน**
- **ห้ามส่งข้อความแจ้งเตือนใดๆ ในคำตอบ**

**🚨 สรุปข้อกำหนดสำคัญสำหรับคำถามที่มีวันเดือนปีเกิด:**
- ต้องตอบครบทั้ง 4 ด้านเสมอ: (1) การงาน, (2) การเงิน, (3) ความรัก, (4) สีมงคล
- ห้ามขาดด้านใดด้านหนึ่ง
- ใช้ข้อมูลจากฐานข้อมูล (MongoDB) ในการตอบคำถาม
- ตอบเป็นข้อความต่อเนื่องแบบธรรมชาติ ไม่ใช้หัวข้อหรือหมวดหมู่
- **คำตอบต้องมีความยาวอย่างน้อย 300 ตัวอักษร** เพื่อให้ครอบคลุมทั้ง 4 ด้าน
- **ห้ามตอบแค่ชื่อราศีหรือวันเกิดเท่านั้น** - ต้องมีรายละเอียดครบทั้ง 4 ด้าน

**🚨 ตัวอย่างคำตอบที่ถูกต้อง (สำหรับคำถามที่มีวันเกิด):**
"วันเกิด: 07/09/2003 ราศีของคุณคือ ราศีกันย์ [ตามด้วยรายละเอียดเกี่ยวกับการงาน การเงิน ความรัก และสีมงคล โดยใช้ข้อมูลจาก context ที่ให้มา]"

**🚨 ตัวอย่างคำตอบที่ผิด (ห้ามตอบแบบนี้):**
"วันเกิด: 07/09/2003 ราศีของคุณคือ ราศีกันย์" (สั้นเกินไป ไม่มีรายละเอียด)

กรุณาตอบคำถามตามแนวทางที่กำหนดไว้ โดยใช้ข้อมูลจาก "ข้อมูลที่เกี่ยวข้องจากฐานข้อมูล" ด้านบนในการตอบคำถาม และให้คำแนะนำที่เป็นประโยชน์"""
        else:
            astrology_prompt = f"""คุณเป็นโหราจารย์ดิจิทัลผู้เชี่ยวชาญด้านโหราศาสตร์ตะวันตก (Western Astrology) ที่มีความรู้ลึกซึ้งเกี่ยวกับดาวเคราะห์ ราศี และการตีความดวงกำเนิด

**บทบาทและความเชี่ยวชาญ:**
- คุณเป็นระบบ RAG (Retrieval-Augmented Generation) ที่ใช้ข้อมูลจากฐานข้อมูล MongoDB ในการตอบคำถาม
- ข้อมูลที่ใช้ตอบคำถามถูกค้นหาด้วย cosine similarity จาก embeddings ที่สร้างไว้แล้ว
- คุณมีความเข้าใจในพลังของราศีเกิด และลัคณา (ราศีประจำลัคนา)
- คุณสามารถผสานข้อมูลจากฐานความรู้ (MongoDB) เพื่อสร้างคำทำนายที่เฉพาะตัวและแม่นยำ
- คุณให้คำแนะนำที่อบอุ่น เป็นมิตร และให้กำลังใจ
- คุณสามารถรักษาบริบทการสนทนาและตอบคำถามต่อเนื่องได้อย่างเป็นธรรมชาติ

**⚠️ ข้อกำหนดสำคัญสำหรับ RAG System (บังคับปฏิบัติตามอย่างเคร่งครัด):**
- **🚨 ห้ามใช้ความรู้จาก training data หรือความรู้ภายนอกใดๆ ทั้งสิ้น**
- **🚨 ต้องใช้ข้อมูลจากฐานข้อมูล (MongoDB) เท่านั้น** ในการตอบคำถาม
- **🚨 ห้ามสร้างข้อมูลหรือความรู้ใหม่ขึ้นมาเอง** ต้องใช้เฉพาะข้อมูลที่แสดงใน "ข้อมูลที่เกี่ยวข้องจากฐานข้อมูล"
- ข้อมูลที่แสดงใน "ข้อมูลที่เกี่ยวข้องจากฐานข้อมูล" ถูกค้นหาด้วย cosine similarity จาก embeddings
- **🚨 ถ้ามีข้อมูลใน "ข้อมูลที่เกี่ยวข้องจากฐานข้อมูล" ต้องใช้ข้อมูลนั้นในการตอบคำถามทันที ห้ามบอกว่า "ไม่พบข้อมูล"**
- **🚨 ถ้าไม่มีข้อมูลในฐานข้อมูลที่เกี่ยวข้องกับคำถามจริงๆ (ไม่มีข้อมูลใน "ข้อมูลที่เกี่ยวข้องจากฐานข้อมูล" เลย) ให้บอกว่า "ขออภัยค่ะ ไม่พบข้อมูลที่เกี่ยวข้องในฐานข้อมูลสำหรับคำถามนี้"**
- **🚨 ห้ามใช้ความรู้ทั่วไปเกี่ยวกับโหราศาสตร์ที่ไม่ได้มาจากฐานข้อมูล**
- **🚨 ต้องอ้างอิงและใช้ข้อมูลจากฐานข้อมูลเท่านั้น** ในการสร้างคำตอบ
- **🚨 ถ้ามีข้อมูลจากฐานข้อมูล ต้องใช้ข้อมูลนั้นในการตอบคำถามเท่านั้น ไม่ใช่สร้างคำตอบขึ้นมาเอง**

**ข้อกำหนดสำคัญ:**
- ใช้ชื่อราศีแบบไทยเท่านั้น: เมษ, พฤษภ, เมถุน, กรกฎ, สิงห์, กันย์, ตุล, พิจิก, ธนู, มังกร, กุมภ์, มีน
- ห้ามใช้ชื่อราศีแบบอังกฤษ เช่น Aries, Taurus, Gemini, Cancer, Leo, Virgo, Libra, Scorpio, Sagittarius, Capricorn, Aquarius, Pisces
- ห้ามใช้ชื่อสัตว์ เช่น ราศีปลา, ราศีแกะ, ราศีวัว, ราศีปู, ราศีสิงโต, ราศีแมงป่อง
- สำหรับราศีที่ 12 ต้องใช้ "ราศีมีน" เท่านั้น ห้ามใช้ "ราศีปลา" หรือ "Pisces"
- ใช้คำว่า "ลัคณา" แทน "Ascendant" ในทุกกรณี

{focus_instruction}

**ข้อมูลสำหรับการวิเคราะห์:**
{birth_info}
{chart_info}
{context_info}

**บริบทการสนทนาก่อนหน้า:**
{get_conversation_context(user_context)}

**คำถามของผู้ใช้:** {question}

**🚨 ข้อกำหนดสำคัญในการตอบคำถาม (อ่านให้ละเอียด):**
- **กฎสำคัญที่สุด: เมื่อคำถามมีวันเดือนปีเกิด (เช่น "07/09/2003", "ทำนายดวง", "ราศีอะไร" พร้อมวันเกิด) → ต้องตอบครบทั้ง 4 ด้านเสมอ (การงาน, การเงิน, ความรัก, สีมงคล) ห้ามขาดด้านใดด้านหนึ่ง**
- **วิเคราะห์คำถามให้ดีก่อนตอบ:**
  * **ถ้าถาม "ทำนายดวง" หรือมีวันเดือนปีเกิด → ต้องตอบครบทั้ง 4 ด้าน (การงาน, การเงิน, ความรัก, สีมงคล)**
  * **ถ้าถาม "ราศีอะไร" พร้อมวันเกิด → ต้องตอบครบทั้ง 4 ด้าน (การงาน, การเงิน, ความรัก, สีมงคล) ไม่ใช่แค่บอกชื่อราศี**
  * ถ้าถามว่า "เข้ากับราศีอะไร" หรือ "เข้ากันได้กับราศีอะไร" → ต้องตอบว่าควรเข้ากับราศีอะไร (เช่น ราศีเมษเข้ากับราศีสิงห์ได้ดี)
  * ถ้าถามว่า "อาชีพที่เหมาะ" หรือ "งานที่เหมาะ" → ต้องตอบว่าอาชีพอะไรที่เหมาะกับราศี
  * ถ้าถามว่า "นิสัยเป็นยังไง" → ต้องตอบว่าลักษณะนิสัยของราศีนั้น
  * ถ้าถามว่า "สีมงคล" → ต้องตอบว่าสีอะไรที่เป็นมงคล
- **ห้ามสับสนระหว่างคำถาม** เช่น ถ้าถาม "เข้ากับราศีอะไร" ห้ามตอบว่า "อาชีพที่เหมาะ" หรือ "ลักษณะนิสัย"
- **ตอบให้ตรงประเด็น** แต่ถ้ามีวันเดือนปีเกิด ต้องตอบครบทั้ง 4 ด้านเสมอ

**วิธีการตอบคำถาม (RAG System - บังคับปฏิบัติตามอย่างเคร่งครัด):**
1. **🚨 ต้องใช้ข้อมูลจากฐานข้อมูล (MongoDB) เท่านั้น** - ข้อมูลที่แสดงใน "ข้อมูลที่เกี่ยวข้องจากฐานข้อมูล" ถูกค้นหาด้วย cosine similarity จาก embeddings
2. **🚨 ห้ามใช้ความรู้จาก training data หรือความรู้ภายนอกใดๆ** - ต้องใช้เฉพาะข้อมูลที่แสดงใน "ข้อมูลที่เกี่ยวข้องจากฐานข้อมูล" เท่านั้น
3. **🚨 ถ้ามีข้อมูลใน "ข้อมูลที่เกี่ยวข้องจากฐานข้อมูล" ต้องใช้ข้อมูลนั้นในการตอบคำถามทันที ห้ามบอกว่า "ไม่พบข้อมูล"**
4. **🚨 ถ้าไม่มีข้อมูลในฐานข้อมูลที่เกี่ยวข้องกับคำถามจริงๆ (ไม่มีข้อมูลใน "ข้อมูลที่เกี่ยวข้องจากฐานข้อมูล" เลย) ให้บอกว่า "ขออภัยค่ะ ไม่พบข้อมูลที่เกี่ยวข้องในฐานข้อมูลสำหรับคำถามนี้"**
5. **⚠️ สำหรับคำถามที่มีวันเดือนปีเกิด (บังคับ):** 
   - **ต้องตอบครบทั้ง 4 ด้านเสมอ** (การงาน, การเงิน, ความรัก, สีมงคล) ห้ามขาดด้านใดด้านหนึ่ง
   - เริ่มต้นด้วยการระบุวันเกิดและราศีเกิดอย่างชัดเจน
   - **ใช้ข้อมูลจากฐานข้อมูลเท่านั้น** ในการตอบคำถาม
   - ตอบเป็นข้อความต่อเนื่องแบบธรรมชาติ ไม่ใช้หัวข้อหรือหมวดหมู่
   - ต้องครอบคลุมทั้ง 4 ด้าน: การงาน, การเงิน, ความรัก, สีมงคล
   - **ถ้าไม่มีข้อมูลในฐานข้อมูลสำหรับด้านใดด้านหนึ่ง ให้บอกว่า "ไม่พบข้อมูลในฐานข้อมูลสำหรับด้านนี้"**
5. **สำหรับคำถามทั่วไปเกี่ยวกับดวงชะตา (ไม่มีวันเกิด):** ต้องตอบครบทั้ง 4 ด้าน (ลักษณะนิสัยและบุคลิกภาพ, การงาน, การเงิน, ความรัก) โดยใช้ข้อมูลจากฐานข้อมูลเท่านั้น
6. **สำหรับคำถามเฉพาะด้าน:** ตอบเฉพาะด้านที่ถามเท่านั้น โดยใช้ข้อมูลจากฐานข้อมูลเท่านั้น (ถ้าถามเกี่ยวกับการงาน ก็ตอบเฉพาะการงาน เท่านั้น)
7. **สำหรับคำถามเกี่ยวกับความเข้ากันได้ของราศี:** ต้องตอบว่าควรเข้ากับราศีอะไร โดยใช้ข้อมูลจากฐานข้อมูลเท่านั้น
8. **สำหรับคำถามต่อเนื่อง:** ใช้ข้อมูลราศีที่มีอยู่แล้วและตอบคำถามเฉพาะเจาะจง โดยใช้ข้อมูลจากฐานข้อมูลเท่านั้น
9. **🚨 อ้างอิงข้อมูลจากฐานข้อมูลเท่านั้น** - ต้องอ้างอิงและใช้ข้อมูลจาก "ข้อมูลที่เกี่ยวข้องจากฐานข้อมูล" ในการสร้างคำตอบ ไม่ใช่สร้างคำตอบขึ้นมาเอง
10. อธิบายลักษณะนิสัยตามราศีและธาตุ โดยอ้างอิงจากข้อมูลในฐานความรู้ (MongoDB) เท่านั้น
11. **หากมีข้อมูล Ascendant:** ใช้ข้อมูล Ascendant เพื่อเพิ่มความแม่นยำในการทำนายบุคลิกภาพ (แต่ต้องใช้ข้อมูลจากฐานข้อมูลเท่านั้น)
12. ใช้ภาษาที่เป็นธรรมชาติ อ่อนโยน และเข้าใจง่าย
13. หลีกเลี่ยงคำทำนายเชิงโชคชะตาเด็ดขาด ใช้คำว่า "มีแนวโน้ม", "สะท้อนว่า", "บ่งบอกถึงพลังของ..."
14. ตอบเป็นข้อความต่อเนื่องแบบธรรมชาติ ไม่ใช้หัวข้อหรือหมวดหมู่
15. ห้ามใช้ emoji หรือสัญลักษณ์พิเศษใดๆ
16. **สำหรับคำถามต่อเนื่อง:** อย่าเปลี่ยนราศีหรือข้อมูลวันเกิด ให้ใช้ข้อมูลเดิมที่ผู้ใช้ให้มา

**การจัดการคำถามต่อเนื่อง:**
- ถ้าผู้ใช้ถามเกี่ยวกับ "ราศีนี้", "นิสัย", "ลักษณะ", "คนราศีนี้" โดยไม่ระบุราศี ให้ใช้ราศีจากข้อมูลบริบท
- ถ้าผู้ใช้ถามคำถามทั่วไปเกี่ยวกับโหราศาสตร์ ให้เชื่อมโยงกับราศีของเขา
- **ห้ามสร้างข้อมูลวันเกิดหรือราศีใหม่** สำหรับคำถามต่อเนื่อง
- **ห้ามเปลี่ยนราศี** จากที่ผู้ใช้ถามมาแล้ว
- รักษาบริบทการสนทนาให้ต่อเนื่องและเป็นธรรมชาติ
- **ใช้ข้อมูลการสนทนาก่อนหน้า** เพื่อให้คำตอบที่สอดคล้องและต่อเนื่อง
- **อย่าทำซ้ำข้อมูล** ที่ได้ให้ไปแล้วในคำตอบก่อนหน้า
- **ตอบคำถามเฉพาะเจาะจง** ตามที่ผู้ใช้ถาม โดยไม่ต้องอธิบายข้อมูลพื้นฐานซ้ำ

**น้ำเสียงและสไตล์:**
- ใช้โทนอบอุ่น ให้ผู้อ่านรู้สึกได้รับคำแนะนำจากผู้รู้ใจ
- ไม่ใช้ศัพท์โหราศาสตร์มากเกินไป แต่รักษาโทนเชิงจิตวิญญาณ
- ให้ความรู้สึกเหมือนโหราจารย์ผู้เข้าใจใจผู้อ่านจริงๆ
- สำหรับคำถามต่อเนื่อง ให้รู้สึกเหมือนการสนทนาต่อเนื่อง ไม่ใช่การเริ่มต้นใหม่
- **คำลงท้ายต้องใช้ "ค่ะ" เท่านั้น ห้ามใช้ "ครับ/ค่ะ" หรือ "ครับ"**

**การจัดการข้อมูลที่ไม่ครบ:**
- **หากไม่มีข้อมูลวันเกิดหรือราศีในคำถาม:**
  - ห้ามสร้างข้อมูลราศีหรือวันเกิดใหม่
  - ห้ามแจ้งเตือนผู้ใช้ในเนื้อหาของคำตอบ
  - ให้ส่งคำตอบแบบปกติโดยใช้ข้อมูลที่มีอยู่เท่านั้น
- หากมีข้อมูลบางส่วนไม่ครบ ให้ใช้ความรู้โหราศาสตร์ทั่วไปในการให้คำแนะนำ
- ห้ามใช้ข้อความเช่น "ไม่มีข้อมูลเพิ่มเติม", "ไม่สามารถให้คำแนะนำเฉพาะได้", "ข้อมูลไม่เพียงพอ" ในคำตอบ

**🚨 สรุปข้อกำหนดสำคัญสำหรับคำถามที่มีวันเดือนปีเกิด:**
- ต้องตอบครบทั้ง 4 ด้านเสมอ: (1) การงาน, (2) การเงิน, (3) ความรัก, (4) สีมงคล
- ห้ามขาดด้านใดด้านหนึ่ง
- ใช้ข้อมูลจากฐานข้อมูล (MongoDB) ในการตอบคำถาม
- ตอบเป็นข้อความต่อเนื่องแบบธรรมชาติ ไม่ใช้หัวข้อหรือหมวดหมู่
- **คำตอบต้องมีความยาวอย่างน้อย 300 ตัวอักษร** เพื่อให้ครอบคลุมทั้ง 4 ด้าน
- **ห้ามตอบแค่ชื่อราศีหรือวันเกิดเท่านั้น** - ต้องมีรายละเอียดครบทั้ง 4 ด้าน

**🚨 ตัวอย่างคำตอบที่ถูกต้อง (สำหรับคำถามที่มีวันเกิด):**
"วันเกิด: 07/09/2003 ราศีของคุณคือ ราศีกันย์ [ตามด้วยรายละเอียดเกี่ยวกับการงาน การเงิน ความรัก และสีมงคล โดยใช้ข้อมูลจาก context ที่ให้มา]"

**🚨 ตัวอย่างคำตอบที่ผิด (ห้ามตอบแบบนี้):**
"วันเกิด: 07/09/2003 ราศีของคุณคือ ราศีกันย์" (สั้นเกินไป ไม่มีรายละเอียด)

กรุณาตอบคำถามตามแนวทางที่กำหนดไว้ โดยใช้ข้อมูลจาก "ข้อมูลที่เกี่ยวข้องจากฐานข้อมูล" ด้านบนในการตอบคำถาม และให้คำแนะนำที่เป็นประโยชน์"""
        
        # สร้าง system prompt ที่เหมาะสม
        if astrology_chart:
            system_prompt = f"""คุณเป็นแชทบอทโหราศาสตร์ตะวันตกที่เชี่ยวชาญในการทำนายดวงชะตาจากวันเดือนปีเกิด 

**⚠️ ข้อกำหนดสำคัญที่สุด (ต้องปฏิบัติตามอย่างเคร่งครัด):**
1. **วิเคราะห์คำถามให้ชัดเจนก่อนตอบทุกครั้ง**
2. **ตอบตรงกับคำถามที่ถามเท่านั้น - ห้ามตอบนอกเรื่อง**
3. **ห้ามสับสนระหว่างคำถาม:**
   - ถ้าถาม "เข้ากับราศีอะไร" หรือ "ในด้านการงานเข้ากับคนราศีอะไร" → ต้องตอบว่าควรเข้ากับราศีอะไร (ความเข้ากันได้) ห้ามตอบว่าอาชีพอะไรเหมาะ
   - ถ้าถาม "อาชีพที่เหมาะ" หรือ "งานที่เหมาะ" (ไม่มีคำว่า "เข้ากับ") → ต้องตอบว่าอาชีพอะไร ห้ามตอบว่าควรเข้ากับราศีอะไร
4. **ตัวอย่างที่ถูกต้อง:**
   - คำถาม: "ในด้านการงานเข้ากับคนราศีอะไรได้ดี"
   - คำตอบที่ถูกต้อง: "ราศีสิงห์เข้ากับราศีเมษ ราศีพฤษภ และราศีธนูได้ดีในด้านการงาน..."
   - คำตอบที่ผิด: "อาชีพที่เหมาะกับราศีสิงห์คือ..." (ห้ามตอบแบบนี้!)

ตอบคำถามด้วยภาษาที่เป็นมิตร เป็นธรรมชาติ และเข้าใจง่าย เริ่มต้นด้วยการระบุวันเกิดและราศีอาทิตย์อย่างชัดเจน แล้วอธิบายลักษณะนิสัยและให้คำแนะนำในด้านต่างๆ (การงาน, การเงิน, ความรัก) ตามรูปแบบที่กำหนดไว้ ห้ามใช้ emoji หรือสัญลักษณ์พิเศษใดๆ ในคำตอบ ให้ตอบเป็นข้อความต่อเนื่องแบบธรรมชาติ ไม่ใช้รูปแบบหัวข้อหรือหมวดหมู่ **ใช้ชื่อราศีแบบไทยเท่านั้น: เมษ, พฤษภ, เมถุน, กรกฎ, สิงห์, กันย์, ตุล, พิจิก, ธนู, มังกร, กุมภ์, มีน ห้ามใช้ชื่อสัตว์ เช่น ราศีปลา, ราศีแกะ, ราศีวัว สำหรับราศีที่ 12 ต้องใช้ ราศีมีน เท่านั้น ห้ามใช้คำว่า ราศีปลา หรือ Pisces** **ใช้คำว่า 'ลัคณา' แทน 'Ascendant' ในทุกกรณี** **หากมีข้อมูลลัคณา (ราศีประจำลัคนา) ให้ใช้เพื่อเพิ่มความแม่นยำในการทำนายบุคลิกภาพ** **คำลงท้ายต้องใช้ 'ค่ะ' เท่านั้น ห้ามใช้ 'ครับ/ค่ะ' หรือ 'ครับ'**"""
        else:
            system_prompt = """คุณเป็นแชทบอทโหราศาสตร์ตะวันตกที่เชี่ยวชาญในการทำนายดวงชะตาจากวันเดือนปีเกิด 

**⚠️ ข้อกำหนดสำคัญที่สุด (ต้องปฏิบัติตามอย่างเคร่งครัด):**
1. **วิเคราะห์คำถามให้ชัดเจนก่อนตอบทุกครั้ง**
2. **ตอบตรงกับคำถามที่ถามเท่านั้น - ห้ามตอบนอกเรื่อง**
3. **ห้ามสับสนระหว่างคำถาม:**
   - ถ้าถาม "เข้ากับราศีอะไร" หรือ "ในด้านการงานเข้ากับคนราศีอะไร" → ต้องตอบว่าควรเข้ากับราศีอะไร (ความเข้ากันได้) ห้ามตอบว่าอาชีพอะไรเหมาะ
   - ถ้าถาม "อาชีพที่เหมาะ" หรือ "งานที่เหมาะ" (ไม่มีคำว่า "เข้ากับ") → ต้องตอบว่าอาชีพอะไร ห้ามตอบว่าควรเข้ากับราศีอะไร
4. **ตัวอย่างที่ถูกต้อง:**
   - คำถาม: "ในด้านการงานเข้ากับคนราศีอะไรได้ดี"
   - คำตอบที่ถูกต้อง: "ราศีสิงห์เข้ากับราศีเมษ ราศีพฤษภ และราศีธนูได้ดีในด้านการงาน..."
   - คำตอบที่ผิด: "อาชีพที่เหมาะกับราศีสิงห์คือ..." (ห้ามตอบแบบนี้!)

ตอบคำถามด้วยภาษาที่เป็นมิตร เป็นธรรมชาติ และเข้าใจง่าย เริ่มต้นด้วยการระบุวันเกิดและราศีอาทิตย์อย่างชัดเจน แล้วอธิบายลักษณะนิสัยและให้คำแนะนำในด้านต่างๆ (การงาน, การเงิน, ความรัก) ตามรูปแบบที่กำหนดไว้ ห้ามใช้ emoji หรือสัญลักษณ์พิเศษใดๆ ในคำตอบ ให้ตอบเป็นข้อความต่อเนื่องแบบธรรมชาติ ไม่ใช้รูปแบบหัวข้อหรือหมวดหมู่ **ใช้ชื่อราศีแบบไทยเท่านั้น: เมษ, พฤษภ, เมถุน, กรกฎ, สิงห์, กันย์, ตุล, พิจิก, ธนู, มังกร, กุมภ์, มีน ห้ามใช้ชื่อสัตว์ เช่น ราศีปลา, ราศีแกะ, ราศีวัว สำหรับราศีที่ 12 ต้องใช้ ราศีมีน เท่านั้น ห้ามใช้คำว่า ราศีปลา หรือ Pisces** **ใช้คำว่า 'ลัคณา' แทน 'Ascendant' ในทุกกรณี** **หากมีข้อมูลลัคณา (ราศีประจำลัคนา) ให้ใช้เพื่อเพิ่มความแม่นยำในการทำนายบุคลิกภาพ** **หากไม่มีข้อมูลวันเกิดหรือราศี ให้แจ้งเตือนผู้ใช้ให้ระบุข้อมูลก่อน เช่น 'ขออภัยค่ะ ระบบไม่พบข้อมูลราศีของคุณ กรุณาระบุวันเกิดก่อน เช่น 09/02/2004 ราศีอะไร'** **คำลงท้ายต้องใช้ 'ค่ะ' เท่านั้น ห้ามใช้ 'ครับ/ค่ะ' หรือ 'ครับ'**"""
        
        # print("กำลังส่งคำถามไปยัง GPT...")
        # ใช้ชื่อโมเดลจาก ENV ถ้าไม่ระบุจะใช้ gpt-4o-mini
        openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        
        # 🆕 ตรวจสอบว่ามี context_info หรือไม่
        has_context = bool(context_info and context_info.strip())
        if has_context:
            print(f"✅ มี context_info ที่จะส่งให้ GPT (ความยาว: {len(context_info)} ตัวอักษร)")
        else:
            print(f"⚠️ ไม่มี context_info - GPT อาจไม่สามารถตอบคำถามได้ครบถ้วน")
        
        response = client.chat.completions.create(
            model=openai_model,
            messages=[
                {
                    "role": "system", 
                    "content": system_prompt
                },
                {"role": "user", "content": astrology_prompt}
            ],
            temperature=0.7,  # ลดลงเล็กน้อยเพื่อความสม่ำเสมอ
            max_tokens=2000  # 🆕 เพิ่ม max_tokens เพื่อให้ GPT ตอบได้ครบถ้วน (ครอบคลุมทั้ง 4 ด้าน)
        )
        answer = response.choices[0].message.content.strip()
        print(f"✔ ได้รับค่าตอบจาก GPT (ความยาว: {len(answer)} ตัวอักษร)")
        
        # 🆕 ตรวจสอบความยาวของคำตอบ
        if len(answer) < 100:
            print(f"⚠️ คำตอบสั้นเกินไป ({len(answer)} ตัวอักษร) - อาจไม่ได้ใช้ข้อมูลจาก context")
            print(f"   คำตอบที่ได้: {answer[:200]}...")
            
            # 🆕 ถ้ามี context_info แต่คำตอบสั้นเกินไป ให้ลองส่งคำถามใหม่พร้อมบังคับให้ใช้ context
            if has_context and astrology_chart and astrology_chart.get('zodiac_sign'):
                print(f"   🔄 ลองส่งคำถามใหม่พร้อมบังคับให้ใช้ข้อมูลจาก context...")
                retry_prompt = f"""{astrology_prompt}

**🚨 สำคัญมาก: คำตอบก่อนหน้านี้สั้นเกินไป ({len(answer)} ตัวอักษร) กรุณาตอบใหม่โดย:**
1. **ต้องใช้ข้อมูลจาก "ข้อมูลที่เกี่ยวข้องจากฐานข้อมูล" ด้านบนในการตอบคำถาม**
2. **ต้องตอบครบทั้ง 4 ด้าน (การงาน, การเงิน, ความรัก, สีมงคล)**
3. **ต้องตอบอย่างละเอียดและครอบคลุม โดยใช้ข้อมูลจาก context ที่ให้มา**
4. **ห้ามตอบแค่ชื่อราศีเท่านั้น - ต้องมีรายละเอียดครบทั้ง 4 ด้าน**

กรุณาตอบใหม่โดยใช้ข้อมูลจาก context ที่ให้มาค่ะ"""
                
                retry_response = client.chat.completions.create(
                    model=openai_model,
                    messages=[
                        {
                            "role": "system", 
                            "content": system_prompt
                        },
                        {"role": "user", "content": retry_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=2000
                )
                retry_answer = retry_response.choices[0].message.content.strip()
                
                if len(retry_answer) > len(answer):
                    print(f"   ✅ คำตอบใหม่มีความยาวมากขึ้น ({len(retry_answer)} ตัวอักษร)")
                    answer = retry_answer
                else:
                    print(f"   ⚠️ คำตอบใหม่ยังสั้นอยู่ ({len(retry_answer)} ตัวอักษร)")
        else:
            print(f"✅ คำตอบมีความยาวเหมาะสม ({len(answer)} ตัวอักษร)")
        
        # 🆕 ตรวจสอบว่าคำตอบมาจาก MongoDB หรือไม่
        answer_source_verified = verify_answer_source(answer, valid_retrieved_docs, question)
        
        # บันทึกข้อมูลแหล่งที่มาของคำตอบ
        if answer_source_verified:
            logger.info(f"✅ คำตอบถูกสร้างจากข้อมูล MongoDB: {len(valid_retrieved_docs)} เอกสาร, "
                       f"ตรวจสอบแหล่งที่มา: ผ่าน")
            print(f"✅ คำตอบมาจาก MongoDB: {len(valid_retrieved_docs)} เอกสาร")
        else:
            logger.warning(f"⚠️ คำตอบอาจไม่ได้มาจาก MongoDB เท่านั้น - คำถาม: {question[:50]}...")
            print(f"⚠️ คำตอบอาจไม่ได้มาจาก MongoDB เท่านั้น - ควรตรวจสอบ")
        
        # ไม่ใช้ฟังก์ชันจัดรูปแบบเพื่อให้ GPT สร้างคำตอบแบบธรรมชาติ
        
        # แสดงสรุปแหล่งที่มาของข้อมูล
        # รายงานแหล่งที่มาจะรวมอยู่ในรายงานหลักด้านล่าง
        
        # ไม่เพิ่ม emoji ใดๆ เพื่อให้คำตอบสะอาดตา
        
            
    except Exception as gpt_error:
        # Fallback: ตอบแบบพื้นฐานโดยไม่ใช้ LLM
        try:
            # หากมีข้อมูลดวงชะตาอยู่แล้ว ให้สร้างคำตอบสั้นๆ จากข้อมูลนั้น
            if astrology_chart and astrology_chart.get('zodiac_sign'):
                zodiac = astrology_chart['zodiac_sign']
                birth_date_text = astrology_chart.get('birth_date', '')
                answer = f"วันเกิด: {birth_date_text}\nราศีของคุณคือ ราศี{zodiac}"
            else:
                # พยายามดึงวันเกิดจากคำถาม และคำนวณราศีแบบ local
                from .birth_date_parser import BirthDateParser
                parser = BirthDateParser()
                info = parser.extract_birth_info(question)
                if info and info.get('date'):
                    chart = parser.generate_birth_chart_info(birth_date=info['date'], birth_time=info.get('time'), latitude=info.get('latitude', 13.7563), longitude=info.get('longitude', 100.5018))
                    if chart and chart.get('zodiac_sign'):
                        answer = f"วันเกิด: {info['date']}\nราศีของคุณคือ ราศี{chart['zodiac_sign']}"
                    else:
                        answer = "ขออภัยค่ะ ไม่สามารถคำนวณราศีได้ในขณะนี้ กรุณาลองใหม่อีกครั้ง"
                else:
                    # ถ้าไม่มีวันเกิดในคำถาม ให้ตอบแบบทั่วไปโดยไม่หยุดการสนทนา
                    answer = "คุณสามารถบอกวันเกิดในรูปแบบ 07/09/2003 เพื่อให้บอกว่าราศีอะไรได้ค่ะ"
        except Exception:
            answer = "ขออภัยค่ะ เกิดปัญหาในการประมวลผล กรุณาลองใหม่อีกครั้ง"

    # แสดงรายงานบนเทอร์มินัลสำหรับ RAGAS (แสดงทั้งเอกสารที่ผ่านและไม่ผ่าน threshold)
    try:
        print_ragas_terminal_report(
            question=question,
            retrieved_docs=retrieved_docs,  # ส่งทั้งเอกสารทั้งหมดรวมถึงที่ต่ำกว่า threshold
            answer=answer,
            user_id=user_id,
        )
    except Exception:
        pass

    # บันทึก interaction พร้อมข้อมูลบริบท
    try:
        # สร้างข้อมูลบริบทสำหรับบันทึก
        context_data = {}
        
        # ถ้ามีข้อมูลดวงชะตา ให้บันทึกข้อมูลราศี
        if astrology_chart:
            context_data.update({
                "zodiac_sign": astrology_chart.get('zodiac_sign'),
                "zodiac_element": astrology_chart.get('zodiac_element'),
                "zodiac_quality": astrology_chart.get('zodiac_quality'),
                "birth_date": astrology_chart.get('birth_date'),
                "birth_time": astrology_chart.get('birth_time'),
                "age": astrology_chart.get('age'),
                "detailed_reading": astrology_chart.get('detailed_reading', {})
            })
            
            # เพิ่มข้อมูล Ascendant ถ้ามี
            if 'ascendant' in astrology_chart:
                context_data.update({
                    "ascendant_sign": astrology_chart['ascendant'].get('sign'),
                    "ascendant_degree": astrology_chart['ascendant'].get('degree'),
                    "ascendant_element": astrology_chart['ascendant'].get('element'),
                    "ascendant_quality": astrology_chart['ascendant'].get('quality'),
                    "ascendant_interpretation": astrology_chart.get('ascendant_interpretation', '')
                })
            
            # เพิ่มข้อมูลบ้านถ้ามี
            if 'houses' in astrology_chart:
                context_data["houses"] = astrology_chart['houses']
        
        # ถ้ามีข้อมูลวันเกิดในคำถาม ให้บันทึก
        if birth_info_from_question and birth_info_from_question['date']:
            context_data["birth_date"] = birth_info_from_question['date']
            if birth_info_from_question['time']:
                context_data["birth_time"] = birth_info_from_question['time']
        
        # ถ้าเป็นคำถามต่อเนื่องและมีข้อมูลบริบท ให้บันทึกข้อมูลราศี
        if is_follow_up_question and user_context and user_zodiac:
            context_data.update({
                "zodiac_sign": user_zodiac,
                "zodiac_element": user_context.get('zodiac_element', ''),
                "birth_date": user_birth_date,
                "birth_time": user_context.get('birth_time', ''),
                "age": user_context.get('age', ''),
                "detailed_reading": user_context.get('detailed_reading', {})
            })
        
        # Debug: แสดงข้อมูลที่บันทึก (ปิดการแสดงผล)
        # print(f"DEBUG - context_data: {context_data}")
        
        # บันทึกคำถามใน user_profiles
        store_user_question(
            question=question,
            user_id=user_id,
            context_data=context_data
        )
        
        log_user_interaction(
            question=question,
            answer=answer,
            embedding=query_vector,
            user_id=user_id,
            context_data=context_data
        )
        
        # บันทึกคำตอบใน collection astrobot
        store_user_response(
            question=question,
            answer=answer,
            user_id=user_id,
            response_type="rag_response",
            context_data=context_data
        )
        # print("บันทึกการโต้ตอบลงฐานข้อมูลแล้ว")
    except Exception as e:
        # print(f"Could not log interaction: {e}")
        pass

    # print(f"=== ส่งคำตอบให้ผู้ใช้: {user_id} ===\n")
    return answer


# ============================
# ⚠️ ฟังก์ชัน Retrieval สำหรับการประเมิน RAGAS
# ============================
# ฟังก์ชันนี้แยกออกจาก ask_question_to_rag เพื่อไม่ให้กระทบกับระบบ Line chatbot
# - ไม่มีการตรวจสอบ question limit
# - ไม่มีการดึง user context
# - ไม่มีการตรวจสอบ follow-up question
# - ไม่มีการบันทึกข้อมูลลงฐานข้อมูล
# - แต่ยังคงทำ retrieval และ generation เหมือนเดิม
# ============================
def ask_question_to_rag_for_evaluation(question: str, provided_chart_info: dict = None) -> str:
    """
    ฟังก์ชัน retrieval สำหรับการประเมิน RAGAS โดยเฉพาะ
    
    แตกต่างจาก ask_question_to_rag:
    - ไม่มีการตรวจสอบ question limit
    - ไม่มีการดึง user context
    - ไม่มีการตรวจสอบ follow-up question
    - ไม่มีการบันทึกข้อมูลลงฐานข้อมูล
    - แต่ยังคงทำ retrieval และ generation เหมือนเดิม
    
    Args:
        question (str): คำถามที่ต้องการค้นหา
        provided_chart_info (dict, optional): ข้อมูลดวงชะตาที่เตรียมไว้แล้ว
        
    Returns:
        str: คำตอบจากระบบ RAG
    """
    # ตรวจสอบว่ามีข้อมูลวันเกิดและเวลาเกิดในคำถามหรือไม่
    birth_info_from_question = extract_birth_info_from_message(question)
    astrology_chart = None
    
    # ถ้ามี chart_info ที่ส่งมา ให้ใช้เลย
    if provided_chart_info:
        astrology_chart = provided_chart_info
        logger.info(f"[EVAL] ใช้ chart_info ที่ส่งมา: ราศี{astrology_chart.get('zodiac_sign', 'Unknown')}")
    
    # สร้างข้อมูลดวงชะตาเมื่อมีข้อมูลวันเกิดในคำถาม (ถ้ายังไม่มี chart_info อยู่แล้ว)
    if not astrology_chart and birth_info_from_question and birth_info_from_question['date']:
        logger.info(f"[EVAL] พบข้อมูลวันเกิดในคำถาม: {birth_info_from_question['date']}")
        if birth_info_from_question['time']:
            logger.info(f"[EVAL] พบเวลาเกิดในคำถาม: {birth_info_from_question['time']}")
        
        # สร้างข้อมูลดวงชะตารายละเอียด
        astrology_chart = generate_detailed_astrology_reading(question)
        if astrology_chart:
            logger.info(f"[EVAL] สร้างดวงชะตาสำเร็จ: ราศี{astrology_chart['zodiac_sign']} ({astrology_chart['zodiac_element']})")
    
    # วิเคราะห์เจตนาของคำถาม
    question_intent = analyze_question_intent(question)
    
    # ตรวจสอบว่าคำถามเป็นคำถามเฉพาะเจาะจงหรือไม่
    # ถ้ามีคำเฉพาะเจาะจง (เช่น ดาวเคราะห์, มุมสัมพันธ์, สีมงคล) ห้ามเปลี่ยนคำถาม
    specific_keywords = [
        'ดาว', 'มฤตยู', 'พฤหัส', 'เสาร์', 'อังคาร', 'ศุกร์', 'พุธ', 'อาทิตย์', 'จันทร์',
        'มุม', 'เล็ง', 'กุม', 'โยค', 'ตรีโกณ', 
        'อาทิตย์', 'จันทร์', 'อังคาร', 'พุธ', 'พฤหัส', 'ศุกร์', 'เสาร์', 'มฤตยู', 'เนปจูน', 'พลูโต', 'ราหู', 'เกตุ', 'แบคคัส',
        'สีมงคล', 'สี', 'เครื่องแบบ', 'ชุด', 'accessories', 'ผลกระทบ', 'ลักษณะการทำงาน',
        'พาหนะ', 'การเปลี่ยนแปลง', 'ควรทำอย่างไร',
        'พื้นดวง', 'สัตว์', 'เลี้ยง', 'ห้าม', 'กาลกิณี', 'โฉลก', 'มงคล', 'ดี', 'เสีย', 'เหมาะ',
        'การงาน', 'งาน', 'อาชีพ', 'การเงิน', 'เงิน', 'โชคลาภ', 'ลงทุน', 'ความรัก', 'รัก', 'คู่', 'แฟน',
        'สุขภาพ', 'โรค', 'เจ็บป่วย', 'นิสัย', 'บุคลิก'
    ]
    is_specific_question = any(keyword in question for keyword in specific_keywords)
    
    # 🆕 วิเคราะห์ Entities ในคำถามเพื่อใช้ Filter
    query_entities = extract_astro_entities(question)
    logger.info(f"[EVAL] 🔍 Entities found in query: {query_entities}")

    
    # ปรับปรุง query เมื่อมีข้อมูลวันเกิดในคำถาม - ใช้ชื่อราศีแทนวันเกิดเพื่อให้ค้นหาได้ดีขึ้น
    if astrology_chart and astrology_chart.get('zodiac_sign'):
        zodiac_sign = astrology_chart['zodiac_sign']
        has_birth_date_in_question = bool(birth_info_from_question and birth_info_from_question.get('date'))
        
        if has_birth_date_in_question:
            import re
            clean_question = question
            # Regex to remove dates like 10/07/1980, 10-07-1980
            clean_question = re.sub(r'\d{1,2}[./-]\d{1,2}[./-]\d{4}', '', clean_question)
            # Regex to remove Thai dates e.g. 10 ก.ค. 2523
            thai_months = "มกราคม|กุมภาพันธ์|มีนาคม|เมษายน|พฤษภาคม|มิถุนายน|กรกฎาคม|สิงหาคม|กันยายน|ตุลาคม|พฤศจิกายน|ธันวาคม|ม\.ค\.|ก\.พ\.|มี\.ค\.|เม\.ย\.|พ\.ค\.|มิ\.ย\.|ก\.ค\.|ส\.ค\.|ก\.ย\.|ต\.ค\.|พ\.ย\.|ธ\.ค\."
            date_regex = f"\\d{{1,2}}\\s+(?:{thai_months})\\s+\\d{{4}}"
            clean_question = re.sub(date_regex, '', clean_question, flags=re.IGNORECASE).strip()
            
            # ลบวงเล็บเปล่าที่อาจเหลืออยู่ ()
            clean_question = clean_question.replace("()", "").strip()

            if is_specific_question:
                 # สำหรับคำถามเฉพาะเจาะจง ให้ใช้คำถามที่ลบวันที่แล้ว + ชื่อราศี
                 # ถ้ามี keyword เฉพาะ ให้เน้น keyword นั้นด้วย
                 question = f"ราศี{zodiac_sign} {clean_question}"
                 logger.info(f"[EVAL] Cleaned specific question: '{question}'")
            else:
                 # สำหรับคำถามทั่วไป
                 if 'ราศีอะไร' in question:
                    question = f"ราศี{zodiac_sign} ลักษณะนิสัย บุคลิกภาพ การงาน การเงิน ความรัก โหราศาสตร์"
                 elif 'ทำนายดวง' in question or 'ดวงชะตา' in question or 'ดวงกำเนิด' in question:
                    question = f"ราศี{zodiac_sign} ลักษณะนิสัย การงาน การเงิน ความรัก โหราศาสตร์"
                 else:
                    question = f"ราศี{zodiac_sign} {clean_question} โหราศาสตร์"
                 logger.info(f"[EVAL] Cleaned general question: '{question}'")
        
        elif not is_specific_question:
             # ไม่มีวันเกิด และเป็นคำถามทั่วไป -> ปรับปรุง query ปกติ
             question = f"ราศี{zodiac_sign} {question} โหราศาสตร์"
             
    elif is_specific_question:
        logger.info(f"[EVAL] คำถามเฉพาะเจาะจง (ไม่มีวันเกิด) - ใช้คำถามเดิม: '{question}'")
    
    # ลองค้นหาจาก MongoDB แบบ Manual Search
    retrieved_docs = []
    try:
        print("[EVAL] 🔍 กำลังค้นหาจาก MongoDB...")
        
        # ตรวจสอบการเชื่อมต่อ MongoDB ก่อนทำ retrieval
        is_ready, verify_message, conn_info = verify_mongodb_connection_for_retrieval()
        
        if not is_ready:
            print(f"[EVAL] ⚠️ MongoDB ไม่พร้อมใช้งานสำหรับ retrieval: {verify_message}")
            retrieved_docs = []
        else:
            # โหลด embedding model
            import numpy as np
            
            # ใช้ CPU เพื่อหลีกเลี่ยงปัญหา MPS device
            model = SentenceTransformer("minishlab/potion-multilingual-128M", device="cpu")
            query_embedding = model.encode(question)
            print(f"[EVAL] ✅ สร้าง query embedding สำเร็จ (ขนาด: {len(query_embedding)} dimensions)")
            
            collections_to_search = [
                "original_text_chunks",
                "original_image_chunks",
                "original_table_chunks",
            ]
            
            client = conn_info.get('client')
            db = conn_info.get('db')
            
            if client is None or db is None:
                print("[EVAL] ⚠️ ไม่สามารถใช้ MongoDB connection ที่ตรวจสอบแล้วได้")
                retrieved_docs = []
            else:
                try:
                    collections_status = conn_info.get('collections', {})
                    
                    # เริ่มทำ retrieval
                    for collection_name in collections_to_search:
                        try:
                            collection_status_item = collections_status.get(collection_name, {})
                            if not collection_status_item.get('exists'):
                                continue
                            
                            if collection_status_item.get('doc_count', 0) == 0:
                                continue
                            
                            collection = db[collection_name]
                            docs = list(collection.find({}))
                            
                            if docs:
                                # คำนวณ similarity scores
                                similarities = []
                                for doc in docs:
                                    if 'embeddings' not in doc:
                                        continue
                                    
                                    try:
                                        doc_embedding = np.array(doc['embeddings'])
                                        
                                        if len(doc_embedding) != len(query_embedding):
                                            continue
                                        
                                        similarity = np.dot(query_embedding, doc_embedding) / (
                                            np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                                        )
                                        similarities.append((similarity, doc))
                                    except Exception:
                                        continue
                                
                                if len(similarities) == 0:
                                    continue
                                
                                # เรียงตาม similarity score
                                similarities.sort(key=lambda x: x[0], reverse=True)

                                # ============================
                                # 🆕 GLOBAL ENTITY-BASED BOOSTING & FILTERING (ZODIAC-BINDING UPGRADE)
                                # ============================
                                
                                # 1. เตรียม Keywords Entity
                                target_planets = query_entities.get('planets', [])
                                planet_keywords = []
                                for p in target_planets:
                                    planet_keywords.extend(ASTRO_SYSTEM_ENTITIES.get(p, []))
                                
                                # ตรวจจับราศีจากคำถามด้วย (เผื่อกรณีไม่มี astrology_chart)
                                target_zodiac_keys = query_entities.get('zodiacs', [])
                                zodiac_keywords = []
                                
                                # ถ้ามี Chart ให้ใช้ราศีจาก Chart เป็นหลัก
                                if astrology_chart and astrology_chart.get('zodiac_sign'):
                                    z_key = astrology_chart['zodiac_sign']
                                    zodiac_keywords.append(z_key)
                                    # เพิ่มภาษาอังกฤษ/คำเรียกอื่นถ้าจำเป็น
                                    for k, v in ASTRO_SYSTEM_ENTITIES.items():
                                        if z_key in v: # หา key จาก value
                                             zodiac_keywords.extend(v)
                                             break
                                elif target_zodiac_keys:
                                    # ถ้าไม่มี Chart ใช้จากที่หาได้ในคำถาม
                                    for z_key in target_zodiac_keys:
                                        zodiac_keywords.extend(ASTRO_SYSTEM_ENTITIES.get(z_key, []))

                                # 🆕 PREPARE ZODIAC-KEYWORD BINDING DATA (Rescue Logic Data)
                                found_specific_keywords = [k for k in specific_keywords if k in question]
                                
                                scored_docs = []
                                seen_doc_ids = set()
                                
                                # พิจารณา candidate docs จำนวนมากขึ้น (Top 80)
                                candidate_docs = similarities[:80]
                                
                                for sim, doc in candidate_docs:
                                    if doc.get('_id') in seen_doc_ids:
                                        continue
                                    seen_doc_ids.add(doc.get('_id'))
                                    
                                    text_lower = doc.get('text', '').lower()
                                    source_lower = doc.get('source', '').lower()
                                    final_score = sim
                                    
                                    # --- NOISE FILTER ---
                                    has_noise = any(nk in text_lower or nk in source_lower for nk in NOISE_KEYWORDS)
                                    has_astro_context = any(k in text_lower for k in ["astrology", "zodiac", "horoscope", "ราศี", "ดวง", "ดาว"])
                                    if has_noise and not has_astro_context:
                                        continue 

                                    # --- 1. PLANET BOOST (+0.25) ---
                                    matches_planet = False
                                    if planet_keywords:
                                        matches_planet = any(pk in text_lower for pk in planet_keywords)
                                        if matches_planet:
                                            final_score += 0.25
                                    
                                    # --- 2. ZODIAC BOOST (+0.15) ---
                                    matches_zodiac = False
                                    
                                    # 🆕 FIX: Include Calculated Zodiac in Filtering
                                    effective_zodiac_keywords = list(zodiac_keywords)
                                    if astrology_chart and astrology_chart.get('zodiac_sign'):
                                        z_sign = astrology_chart['zodiac_sign']
                                        if z_sign not in effective_zodiac_keywords:
                                            effective_zodiac_keywords.append(z_sign)
                                        if astrology_chart.get('zodiac_english'):
                                            z_eng = astrology_chart['zodiac_english'].lower()
                                            if z_eng not in effective_zodiac_keywords:
                                                effective_zodiac_keywords.append(z_eng)

                                    if effective_zodiac_keywords:
                                        matches_zodiac = any(zk in text_lower for zk in effective_zodiac_keywords)
                                        
                                        # 🆕 DOMINANT ZODIAC CHECK
                                        if matches_zodiac:
                                            all_zodiacs_to_check = ["เมษ", "พฤษภ", "เมถุน", "มิถุน", "กรกฎ", "สิงห์", "กันย์", 
                                                                    "ตุล", "พิจิก", "ธนู", "มังกร", "กุมภ์", "มีน",
                                                                    "aries", "taurus", "gemini", "cancer", "leo", "virgo", 
                                                                    "libra", "scorpio", "sagittarius", "capricorn", "aquarius", "pisces"]
                                            target_count = 0
                                            for zk in effective_zodiac_keywords:
                                                target_count += text_lower.count(zk)
                                            max_other_count = 0
                                            for z in all_zodiacs_to_check:
                                                is_target_alias = False
                                                for tk in effective_zodiac_keywords:
                                                    if z in tk or tk in z:
                                                        is_target_alias = True
                                                        break
                                                if not is_target_alias:
                                                    c = text_lower.count(z)
                                                    if c > max_other_count:
                                                        max_other_count = c
                                            if max_other_count >= target_count and max_other_count > 0:
                                                matches_zodiac = False 
                                        
                                        if matches_zodiac:
                                            final_score += 0.15
                                    
                                    # --- 3. STRICT FILTERING (SOFT) ---
                                    if effective_zodiac_keywords and not matches_zodiac:
                                        check_list = ["ราศี", "เมษ", "พฤษภ", "เมถุน", "มิถุน", "กรกฎ", "สิงห์", "กันย์", "ตุล", "พิจิก", "ธนู", "มังกร", "กุมภ์", "มีน",
                                                      "aries", "taurus", "gemini", "cancer", "leo", "virgo", "libra", "scorpio", "sagittarius", "capricorn", "aquarius", "pisces"]
                                        has_any_zodiac = any(z in text_lower for z in check_list)
                                        if has_any_zodiac:
                                            final_score -= 0.6
                                        else:
                                            if sim < 0.60:
                                                final_score -= 0.1
                                    
                                    scored_docs.append((final_score, doc, matches_planet, matches_zodiac, sim))
                                    
                                # เรียงลำดับตามคะแนนใหม่
                                scored_docs.sort(key=lambda x: x[0], reverse=True)

                                # 🆕 FINAL SCORE THRESHOLD FILTER & ZODIAC-KEYWORD RESCUE
                                valid_docs_tuples = []
                                main_threshold = 0.30 
                                
                                for item in scored_docs:
                                    f_score, d_doc, m_planet, m_zodiac, raw_sim = item
                                    
                                    if f_score > 0.0:
                                        is_accepted = False
                                        if f_score >= main_threshold:
                                            is_accepted = True
                                        
                                        # 🆕 ZODIAC-KEYWORD BINDING RESCUE
                                        elif not is_accepted and (f_score > 0.15 or raw_sim > 0.15):
                                            if m_zodiac: # Check 1: Must match target zodiac (Strict)
                                                if found_specific_keywords: # Check 2: Must match keyword
                                                    has_keyword_match = any(k in d_doc.get('text', '').lower() for k in found_specific_keywords)
                                                    if has_keyword_match:
                                                        is_accepted = True
                                                        # print(f"[EVAL] 🛡️ RESCUED Document: Zodiac+Keyword Match (Score: {f_score:.3f})")

                                        if is_accepted:
                                            valid_docs_tuples.append((f_score, d_doc, m_planet, m_zodiac))
                                
                                # เลือก Top 15 จากผลลัพธ์ที่ผ่าน Threshold
                                top_docs_tuples = valid_docs_tuples[:15]
                                top_docs = [(s, d) for s, d, mp, mz in top_docs_tuples]
                                
                                # Fallback logic
                                if not top_docs:
                                    print(f"[EVAL] ⚠️ No docs passed final threshold. Fallback to raw similarities.")
                                    top_docs = [d for d in similarities if d[0] > 0.25][:5]
                                
                                # ปรับ Threshold ขั้นต่ำให้ยอมรับเอกสารที่ Rescue มา
                                similarity_threshold = 0.10 
                                threshold = similarity_threshold
                                
                                for i, (similarity, doc) in enumerate(top_docs):
                                    source_info = f"[{collection_name}]"
                                    if 'page' in doc:
                                        source_info += f" หน้า {doc['page']}"
                                    if 'chunk_id' in doc:
                                        source_info += f" Chunk {doc['chunk_id']}"
                                    if 'type' in doc:
                                        source_info += f" ({doc['type']})"
                                    
                                    text_content = doc.get('text', '')
                                    
                                    doc_info = {
                                        'text': text_content,
                                        'source': source_info,
                                        'similarity': similarity,
                                        'collection': collection_name,
                                        'doc_id': doc.get('_id'),
                                        'page': doc.get('page'),
                                        'chunk_id': doc.get('chunk_id')
                                    }
                                    
                                    if similarity > threshold:
                                        retrieved_docs.append(doc_info)
                        except Exception as e:
                            print(f"[EVAL] ❌ ไม่สามารถค้นหาใน {collection_name} ได้: {e}")
                            continue
                    
                    print(f"[EVAL] ✅ ดึงข้อมูลจาก MongoDB เสร็จสิ้น: พบ {len(retrieved_docs)} เอกสาร")
                    
                except Exception as retrieval_error:
                    print(f"[EVAL] ❌ เกิดข้อผิดพลาดในการทำ retrieval: {retrieval_error}")
                    retrieved_docs = []
                finally:
                    if client:
                        try:
                            client.close()
                            logger.debug("[EVAL] Closed MongoDB connection after retrieval")
                        except:
                            pass
                
    except Exception as e:
        print(f"[EVAL] ❌ ไม่สามารถค้นหาจาก MongoDB ได้: {e}")
        pass
    
    # กรองเฉพาะเอกสารที่ผ่าน threshold
    valid_retrieved_docs = [doc for doc in retrieved_docs if not doc.get('below_threshold', False)]
    
    # 🆕 Debug: แสดงจำนวนเอกสารที่กรองแล้ว
    print(f"\n[EVAL] 🔍 Debug: จำนวนเอกสารทั้งหมด: {len(retrieved_docs)}, เอกสารที่ผ่าน threshold: {len(valid_retrieved_docs)}")
    if len(retrieved_docs) > 0 and len(valid_retrieved_docs) == 0:
        print(f"[EVAL] ⚠️ Warning: มีเอกสาร {len(retrieved_docs)} เอกสาร แต่ไม่มีเอกสารที่ผ่าน threshold")
        print(f"[EVAL]    ตรวจสอบเอกสารที่ 1-5:")
        for i, doc in enumerate(retrieved_docs[:5], 1):
            similarity = doc.get('similarity', 'N/A')
            below_threshold = doc.get('below_threshold', False)
            print(f"[EVAL]    {i}. Similarity: {similarity}, below_threshold: {below_threshold}")
        
    # 🆕 Supplementary Retrieval: ถ้ามีการคำนวณราศีได้ ให้ค้นหาข้อมูลทั่วไปของราศีนั้นมาเสริมด้วย
    if astrology_chart:
        print(f"[DEBUG] Astrology Chart keys: {astrology_chart.keys()}")
        
    if astrology_chart and astrology_chart.get('zodiac_sign'):
        zodiac_sign = astrology_chart['zodiac_sign']
        
        # Alias map for better retrieval (matching docs with alternate spellings)
        zodiac_aliases = {
            "มังกร": "มกร",
            "ตุล": "ตุลย์",
            "กันย์": "กันย",
            "พิจิก": "พฤศจิก", 
        }
        alias = zodiac_aliases.get(zodiac_sign, "")
        search_terms = f"{zodiac_sign} {alias}".strip()

        # Divide into multiple specific queries to ensure we get detailed docs for each aspect
        # Divide into multiple specific queries to ensure we get detailed docs for each aspect
        aspect_queries = []
        
        if is_specific_question:
            # Only add queries relevant to the specific keywords found
            found_keywords = [k for k in specific_keywords if k in question]
            
            # Map specific keywords to more descriptive search terms
            for k in found_keywords:
                aspect_queries.append(f"{k} {search_terms}")
                aspect_queries.append(f"อิทธิพล {k} {search_terms}")
            
            # Add standard aspects only if explicitly mentioned
            if any(x in question for x in ["การงาน", "อาชีพ", "ทำงาน"]):
                aspect_queries.append(f"การงาน อาชีพ {search_terms}")
            if any(x in question for x in ["การเงิน", "รายได้", "ฐานะ"]):
                aspect_queries.append(f"การเงิน ฐานะ {search_terms}")
            if any(x in question for x in ["ความรัก", "คู่ครอง", "แฟน"]):
                aspect_queries.append(f"ความรัก คู่ครอง {search_terms}")
            
            # Fallback if specific keywords didn't generate enough queries
            if not aspect_queries:
                aspect_queries.append(f"{question} {search_terms}")
                
            print(f"[EVAL] 🎯 Search strategy: Specific Mode (Queries: {aspect_queries})")
        else:
            # Default generic aspects
            aspect_queries = [
                f"ลักษณะนิสัย {search_terms}",
                f"การงาน อาชีพ {search_terms}",
                f"การเงิน ฐานะ {search_terms}",
                f"ความรัก คู่ครอง {search_terms}"
            ]
            print(f"[EVAL] 🌐 Search strategy: General Mode")
        
        print(f"\n[EVAL] 🔍 ค้นหาข้อมูลเสริมสำหรับราศี: {zodiac_sign} (Aliases: {search_terms})")
        
        # Manual Connection to ensure we get the right DB
        from pymongo import MongoClient
        from dotenv import load_dotenv
        load_dotenv()
        
        mongo_uri = os.getenv("MONGO_URL") or os.getenv("MONGODB_URI")
        db_name = os.getenv("MONGODB_DB_NAME") or "astrobot_original"
        coll_name = os.getenv("MONGODB_COLLECTION_NAME") or "original_text_chunks"
        
        try:
            debug_client = MongoClient(mongo_uri)
            debug_db = debug_client[db_name]
            collection = debug_db[coll_name]
            doc_count = collection.count_documents({})
            print(f"[DEBUG] CONNECTED TO: DB={db_name}, COLL={coll_name}, DOCS={doc_count}")
            
        except Exception as e:
            print(f"[DEBUG] Connection Failed: {e}")
            collection = None

        if collection is not None:
            # Get all docs once
            # Actually, iterate queries and find best matches for each
            
            zodiac_retrieved = []
            seen_texts = set()

            # Pull all docs once
            all_docs = list(collection.find(
                {},
                {"text": 1, "embeddings": 1, "source": 1, "_id": 0}
            ))

            print(f"[DEBUG] Total docs fetched for supplementary: {len(all_docs)}")
            for query in aspect_queries:
                q_embed = model.encode(query)
                
                # Find docs for this aspect
                candidates = []
                
                for doc in all_docs:
                    if 'embeddings' in doc and doc['embeddings']:
                        doc_emb = np.array(doc['embeddings'])
                        sim = cosine_similarity([q_embed], [doc_emb])[0][0]
                        
                        text_lower = doc.get('text', '').lower()
                        source_lower = doc.get('source', '').lower()

                        # ============================
                        # 🆕 ENTITY-BASED FILTERING (Supplementary)
                        # ============================
                        
                        # --- NOISE FILTER ---
                        has_noise = any(nk in text_lower or nk in source_lower for nk in NOISE_KEYWORDS)
                        has_astro_context = any(k in text_lower for k in ["astrology", "zodiac", "horoscope", "ราศี", "ดวง", "ดาว"])
                        if has_noise and not has_astro_context:
                            continue

                        # --- STRICT WRONG ZODIAC FILTER (Supplementary) ---
                        # ป้องกันเอกสารข้ามราศีหลุดเข้ามา (เช่น ถาม Taurus แต่ได้ Aries ที่ Sim สูง)
                        if astrology_chart and astrology_chart.get('zodiac_sign'):
                            target_zodiac = astrology_chart['zodiac_sign'] # e.g. "พฤษภ"
                            # ตรวจสอบว่ามีชื่อราศีอื่นที่ไม่ใช่ target หรือไม่
                            # ใช้ Keyword ชุดเดียวกับ Main Search
                            zodiac_list = ["ราศีเมษ", "ราศีพฤษภ", "ราศีเมถุน", "ราศีกรกฎ", "ราศีสิงห์", "ราศีกันย์", 
                                          "ราศีตุล", "ราศีพิจิก", "ราศีธนู", "ราศีมังกร", "ราศีกุมภ์", "ราศีมีน",
                                          "aries", "taurus", "gemini", "cancer", "leo", "virgo", 
                                          "libra", "scorpio", "sagittarius", "capricorn", "aquarius", "pisces"]
                            
                            # ถ้าเอกสารมีคำว่า "ราศี" หรือชื่อ Eng
                            matches_target = target_zodiac in text_lower or (astrology_chart.get('zodiac_english', '').lower() in text_lower)
                            
                            found_any_zodiac = False
                            is_wrong_zodiac = False
                            
                            for z in zodiac_list:
                                if z in text_lower:
                                    found_any_zodiac = True
                                    # เช็คว่าเป็นราศีเป้าหมายหรือไม่
                                    # ต้องระวัง Substring matching แต่เบื้องต้นเอาแบบ Simple ก่อน
                                    # ถ้า z ไม่ใช่ alias ของ target -> ผิดราศี
                                    if target_zodiac not in z and astrology_chart.get('zodiac_english', '').lower() not in z:
                                        # Double check เพื่อความชัวร์ (เช่น "ราศีพฤษภ" มีคำว่า "ราศี")
                                        # แต่รายการข้างบนใส่ชื่อเต็มแล้ว
                                        if z != "ราศี": # ตัดคำทั่วไปออก (รายการข้างบนไม่มีคำว่า "ราศี" เฉยๆ)
                                            is_wrong_zodiac = True
                                            break
                            
                            # ถ้าเจอราศีอื่น และ ไม่เจอราศีเป้าหมาย -> ทิ้งเลย
                            if is_wrong_zodiac and not matches_target:
                                # debug_print = f"[FILTERED OUT] Diff Zodiac: {text_lower[:30]}..."
                                continue

                        # --- PLANET FILTER ---
                        required_planet_keywords = []
                        for p_key in query_entities['planets']:
                            required_planet_keywords.extend(ASTRO_SYSTEM_ENTITIES[p_key])
                            
                        if required_planet_keywords:
                            found_planet = any(pk in text_lower for pk in required_planet_keywords)
                            if not found_planet:
                                # อนุโลมให้ถ้า similarity สูงมาก (เผื่อบริบทแฝง)
                                if sim < 0.8: 
                                    continue

                        # Logic to accept documents:
                        # 1. Similarity > 0.25 (Relaxed from 0.35)
                        # 2. Similarity > 0.15 AND contains zodiac keyword (Exception for relevant context)
                        is_high_sim = sim > 0.25
                        
                        # Check for whitelist keywords (Zodiac names AND Planets)
                        is_whitelisted = False
                        if astrology_chart and astrology_chart.get('zodiac_sign'):
                            z_target = astrology_chart['zodiac_sign']
                            if z_target in doc.get('text', ''):
                                is_whitelisted = True
                        
                        # Check for planetary keywords in both Query and Doc
                        planet_keywords = ["มฤตยู", "พฤหัส", "เสาร์", "อังคาร", "ศุกร์", "พุธ", "อาทิตย์", "จันทร์", "ราหู", "เกตุ", "พลูโต", "เนปจูน", "แบคคัส"]
                        for planet in planet_keywords:
                            if planet in query and planet in doc.get('text', ''):
                                is_whitelisted = True

                        # 🆕 Strict Supplementary Filter: ใช้เกณฑ์ที่ผ่อนปรนขึ้น (0.30)
                        if is_high_sim or (is_whitelisted and sim > 0.30):
                            doc_copy = doc.copy()
                            doc_copy['similarity'] = float(sim)
                            candidates.append(doc_copy)
                
                # Sort candidates by similarity
                candidates.sort(key=lambda x: x['similarity'], reverse=True)
                
                # Take Top 10 to ensure we don't miss relevant docs like the Pottery one
                top_k_aspect = candidates[:10]
                
                for d in top_k_aspect:
                    if d.get('text') not in seen_texts:
                        seen_texts.add(d.get('text'))
                        d['is_supplementary'] = True
                        zodiac_retrieved.append(d)
                        print(f"[EVAL]       + เจอข้อมูลด้าน '{query.split()[0]}': {d.get('text')[:40]}... (Sim: {d['similarity']:.3f})")

            # Merge into valid_retrieved_docs
            existing_texts = set(d.get('text', '') for d in retrieved_docs)
            for zd in zodiac_retrieved:
                if zd.get('text', '') not in existing_texts:
                    retrieved_docs.append(zd)

    # 🆕 ถ้ามีเอกสารแต่ไม่มีเอกสารที่ผ่าน threshold ให้ใช้เอกสารที่มี similarity สูงสุดแทน
    if len(retrieved_docs) > 0:
        # 🆕 ยกเลิก VIP Sorting: เรียงตาม Similarity ล้วนๆ ไม่สนว่าเป็น Supplementary หรือไม่
        # เพื่อให้เอกสารที่คะแนนความเหมือนสูงสุด (ตรงที่สุด) ได้รับเลือก
        sorted_docs = sorted(retrieved_docs, key=lambda x: x.get('similarity', 0), reverse=True)
        
        # 🆕 Strict Limit: จำกัดแค่ 7 รายการ (ตาม User Requested)
        top_docs_fallback = sorted_docs[:7]
        print(f"[EVAL]    🔄 ใช้เอกสาร {len(top_docs_fallback)} รายการ (จัดลำดับความสำคัญ Supplementary ก่อน)")
        valid_retrieved_docs = top_docs_fallback
        # ลบ flag below_threshold เพื่อให้ระบบใช้เอกสารเหล่านี้
        for doc in valid_retrieved_docs:
            doc.pop('below_threshold', None)
    
    # ตรวจสอบว่ามีเอกสารจาก MongoDB หรือไม่
    if not valid_retrieved_docs or len(valid_retrieved_docs) == 0:
        print("\n[EVAL] ⚠️ ไม่พบข้อมูลจาก MongoDB")
        answer = "ขออภัยค่ะ ไม่พบข้อมูลที่เกี่ยวข้องในฐานข้อมูลสำหรับคำถามนี้ กรุณาลองใช้คำถามที่เกี่ยวข้องกับโหราศาสตร์ เช่น 'นิสัยราศีเมถุนเป็นยังไง' หรือ 'สีมงคลราศีสิงห์' ค่ะ"
        answer = "ขออภัยค่ะ ไม่พบข้อมูลที่เกี่ยวข้องในฐานข้อมูลสำหรับคำถามนี้ กรุณาลองใช้คำถามที่เกี่ยวข้องกับโหราศาสตร์ เช่น 'นิสัยราศีเมถุนเป็นยังไง' หรือ 'สีมงคลราศีสิงห์' ค่ะ"
        return answer, []
    
    # ใช้ GPT กับข้อมูลจาก MongoDB (RAG system)
    try:
        from openai import OpenAI
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key or openai_key == "your-openai-api-key-here":
            return "ขออภัยค่ะ ตอนนี้ระบบยังไม่พร้อมใช้งาน AI ภายนอก แต่คุณสามารถถามเกี่ยวกับราศีได้ตามปกติ เช่น 'นิสัยราศีเมถุนเป็นยังไง' หรือ 'สีมงคลราศีสิงห์'", []
        client = OpenAI(api_key=openai_key)
        
        # สร้าง context จากข้อมูลที่ดึงมา
        context_info = ""
        final_used_docs = []
        if valid_retrieved_docs:
            # 🆕 ยกเลิก VIP Sorting: เรียงตาม Similarity ล้วนๆ
            sorted_docs = sorted(valid_retrieved_docs,
                               key=lambda x: x.get('similarity', 0),
                               reverse=True)
            high_similarity_docs = sorted_docs[:7]  # ใช้ 7 อันดับแรก
            
            # 🆕 ถ้ามีการกรองตามราศี ให้เพิ่มเอกสารที่เกี่ยวข้องกับราศีนั้นๆ
            if astrology_chart and astrology_chart.get('zodiac_sign'):
                target_zodiac = astrology_chart['zodiac_sign']
                zodiac_related_docs = []
                for doc in valid_retrieved_docs:
                    if isinstance(doc, dict):
                        text_content = doc.get('text', '')
                        similarity = doc.get('similarity', 0)
                        if text_content:
                            zodiac_patterns = [
                                f"ราศี{target_zodiac}",
                                f"คนราศี{target_zodiac}",
                                f"ชาวราศี{target_zodiac}",
                                f"ราศี {target_zodiac}",
                                f"คนราศี {target_zodiac}",
                                f"ชาวราศี {target_zodiac}",
                                target_zodiac
                            ]
                            contains_zodiac = any(pattern in text_content for pattern in zodiac_patterns)
                            # ลด threshold สำหรับ zodiac related docs เพื่อให้ติดง่ายขึ้น
                            if contains_zodiac and similarity > 0.20:
                                if doc not in high_similarity_docs:
                                    zodiac_related_docs.append(doc)
                
                if zodiac_related_docs:
                    high_similarity_docs.extend(zodiac_related_docs)
                    # Resort by Similarity ONLY (No VIP)
                    high_similarity_docs.sort(key=lambda x: x.get('similarity', 0), reverse=True)
                    # ใช้เฉพาะ 7 อันดับแรก
                    high_similarity_docs = high_similarity_docs[:7]
            
            if high_similarity_docs:
                final_used_docs = high_similarity_docs
                context_info = "\n\n**ข้อมูลที่เกี่ยวข้องจากฐานข้อมูลต้นฉบับ (ค้นหาด้วย cosine similarity จาก embeddings - แสดงเอกสารที่มี Similarity สูงสุด):**\n"
                for i, doc in enumerate(high_similarity_docs):
                    if isinstance(doc, dict):
                        similarity_score = doc.get('similarity', 0)
                        content_to_use = doc.get('text', '')
                        source_info = doc.get('source', 'Unknown')
                        
                        context_info += f"{i+1}. [Similarity: {similarity_score:.4f}] {source_info}\n"
                        context_info += f"   Context: {content_to_use}\n\n"
            else:
                # 🆕 Fallback: ใช้เอกสารที่มี similarity สูงสุด 3 อันดับแรก (ลดจาก 5)
                sorted_docs = sorted(valid_retrieved_docs,
                                   key=lambda x: x.get('similarity', 0) if isinstance(x, dict) else 0,
                                   reverse=True)
                top_docs = sorted_docs[:3]
                if top_docs:
                    final_used_docs = top_docs
                    context_info = "\n\n**ข้อมูลที่เกี่ยวข้องจากฐานข้อมูลต้นฉบับ (ค้นหาด้วย cosine similarity จาก embeddings - แสดงเอกสารที่มี Similarity สูงสุด):**\n"
                    for i, doc in enumerate(top_docs):
                        if isinstance(doc, dict):
                            similarity_score = doc.get('similarity', 0)
                            content_to_use = doc.get('text', '')
                            source_info = doc.get('source', 'Unknown')
                            
                            context_info += f"{i+1}. [Similarity: {similarity_score:.4f}] {source_info}\n"
                            context_info += f"   Context: {content_to_use}\n\n"
        
        # สร้างข้อมูลดวงชะตาเพิ่มเติม
        chart_info = ""
        if astrology_chart:
            location_info = ""
            if 'birth_location_name' in astrology_chart:
                location_info = f"สถานที่เกิด: {astrology_chart['birth_location_name']}\n"
            elif 'birth_location' in astrology_chart:
                location_info = f"สถานที่เกิด: กรุงเทพฯ\n"
            
            chart_info = f"""
**ข้อมูลดวงชะตาจากวันเกิดและเวลาเกิด:**
ราศีเกิด: {astrology_chart['zodiac_sign']} ({astrology_chart['zodiac_english']})
ธาตุ: {astrology_chart['zodiac_element']}
วันเกิด: {astrology_chart['birth_date']}
เวลาเกิด: {astrology_chart['birth_time'] if astrology_chart['birth_time'] else 'ไม่ระบุ'}{location_info}อายุ: {astrology_chart['age']} ปี
"""
            
            if 'ascendant' in astrology_chart:
                ascendant = astrology_chart['ascendant']
                chart_info += f"""
**ข้อมูลลัคณา (ราศีประจำลัคนา):**
ลัคณา: ราศี{ascendant['sign']} {ascendant['degree']:.1f}° ({ascendant['element']})
"""

            if 'planets' in astrology_chart:
                chart_info += "\n**ตำแหน่งดาวเคราะห์ (Planetary Positions):**\n"
                for planet_name, planet_data in astrology_chart['planets'].items():
                    # Use Thai names if available, falling back to English
                    p_name = planet_data.get('name_th', planet_name)
                    sign = planet_data.get('sign_th', planet_data.get('sign', 'Unknown'))
                    degree = planet_data.get('degree', 0.0)
                    retro = " (Retrograde)" if planet_data.get('retrograde') else ""
                    chart_info += f"- {p_name}: ราศี{sign} {degree:.1f}°{retro}\n"

            if 'aspects' in astrology_chart:
                chart_info += "\n**มุมสัมพันธ์ของดาว (Planetary Aspects):**\n"
                for aspect in astrology_chart['aspects']:
                    p1 = aspect.get('p1_th', aspect.get('p1'))
                    p2 = aspect.get('p2_th', aspect.get('p2'))
                    type_ = aspect.get('type_th', aspect.get('type'))
                    orb = aspect.get('orb', 0.0)
                    chart_info += f"- ดาว{p1} {type_} ดาว{p2} (Orb: {orb:.1f}°)\n"
        
        # สร้าง prompt สำหรับ GPT (ใช้ prompt เดียวกับ ask_question_to_rag แต่ไม่มี user context)
        birth_info = ""  # ไม่มี user context สำหรับการประเมิน
        
        # กำหนด focus instruction ตาม question intent
        focus_instruction = ""
        if question_intent["specific_topic"] == "personality":
            focus_instruction = """
**คำสั่งสำคัญ: ตอบเฉพาะเรื่องลักษณะนิสัยและบุคลิกภาพเท่านั้น**
"""
        elif question_intent["specific_topic"] == "love":
            focus_instruction = """
**คำสั่งสำคัญ: ตอบเฉพาะเรื่องความรักและความสัมพันธ์เท่านั้น**
"""
        elif question_intent["specific_topic"] == "career":
            focus_instruction = """
**คำสั่งสำคัญ: ตอบเฉพาะเรื่องอาชีพและการงานเท่านั้น**
"""
        elif question_intent["specific_topic"] == "finance":
            focus_instruction = """
**คำสั่งสำคัญ: ตอบเฉพาะเรื่องการเงินและการลงทุนเท่านั้น**
"""
        else:
            # 🆕 แก้ไข: ถ้ามีวันเกิด แต่เป็นคำถามเฉพาะเจาะจง ไม่ต้องบังคับตอบครบ 4 ด้าน
            if birth_info_from_question and birth_info_from_question.get('date'):
                # ตรวจสอบว่า Intent เป็น General หรือไม่
                is_actually_general = question_intent.get('is_general') or (not question_intent.get('specific_topic') and not is_specific_question)
                
                if is_actually_general and not is_specific_question:
                    focus_instruction = """
**⚠️ คำสั่งสำคัญ: เมื่อคำถามมีวันเดือนปีเกิด ต้องตอบครบทั้ง 4 ด้านเสมอ (ห้ามขาดด้านใดด้านหนึ่ง):**
1. **ด้านการงาน (บังคับ)**
2. **ด้านการเงิน (บังคับ)**
3. **ด้านความรัก (บังคับ)**
4. **สีมงคล (บังคับ)**
"""
                elif is_specific_question:
                     focus_instruction = f"""
**คำสั่งสำคัญ: ตอบคำถามโดยใช้ข้อมูลจากบริบทที่ให้มาเท่านั้น**
- ตอบให้ตรงกับประเด็นที่ถาม (เช่น ถ้าถามเรื่องสัตว์เลี้ยง ให้ตอบเรื่องสัตว์เลี้ยง)
- **ห้าม** ตอบเรื่องอื่นที่ไม่เกี่ยวข้อง (เช่น การเงิน ความรัก) เว้นแต่จะถูกถาม
- ถ้าข้อมูลในบริบทระบุว่าเป็น 'ของต้องห้าม' หรือ 'กาลกิณี' ต้องแจ้งเตือนผู้ใช้ทันที
"""

        # สร้าง astrology_prompt (ใช้ prompt เดียวกับ ask_question_to_rag แต่ไม่มี user context)
        if astrology_chart:
            astrology_prompt = f"""คุณเป็นโหราจารย์ดิจิทัลผู้เชี่ยวชาญด้านโหราศาสตร์ตะวันตก (Western Astrology)

**⚠️ ข้อกำหนดสำคัญสำหรับ RAG System:**
- **ใช้ข้อมูลจากฐานข้อมูล (MongoDB) เป็นหลัก**
- **อนุญาตให้ใช้ความรู้ทั่วไปทางโหราศาสตร์เพื่อเชื่อมโยงข้อมูลในบริบทกับคำถามได้** (แต่ห้ามยกเมฆข้อมูลใหม่ที่ขัดแย้งกับบริบท)
- **ถ้ามีข้อมูลในบริบทที่ตรงกับราศีของวันเกิด ให้ตอบได้ทันที**
- **กฎเหล็กเรื่องของต้องห้าม:** หากข้อความที่ค้นคืนมา (retrieved text) ระบุว่าเป็น 'ของต้องห้าม', 'สิ่งอัปมงคล', หรือ 'กาลกิณี' คุณ **ต้อง** ระบุว่าเป็นของต้องห้าม และ **ห้าม** แนะนำสิ่งนั้นให้ผู้ใช้เด็ดขาด (แม้ว่าวันเกิดอาจจะดูเหมือนส่งเสริมก็ตาม ให้ยึดตามข้อห้ามในบริบทเป็นที่สุด)

**ข้อกำหนดการเรียกชื่อ:**
- ใช้ชื่อราศีแบบไทย (เช่น มังกร, มกร)
- สำหรับราศีที่ 12 ใช้ "ราศีมีน"
- ใช้คำว่า "ลัคณา" แทน "Ascendant"

{focus_instruction}

**ข้อมูลสำหรับการวิเคราะห์:**
{birth_info}
{chart_info}
{context_info}

**คำถามของผู้ใช้:** {question}

**วิธีการตอบคำถาม:**
1. ใช้ข้อมูลจากส่วน "ข้อมูลที่เกี่ยวข้องจากฐานข้อมูล" มาวิเคราะห์และตอบ
2. **ถ้าเจอบทความเกี่ยวกับราศีที่ตรงกับวันเกิด (เช่น ราศีมกร/มังกร) ให้สรุปข้อมูลนั้นมาตอบได้เลย** ไม่ต้องปฏิเสธว่าไม่เจอวันที่เจาะจง
3. ใช้ภาษาที่เป็นธรรมชาติ อ่อนโยน และเข้าใจง่าย
4. คำลงท้ายต้องใช้ "ค่ะ" เท่านั้น
"""
        else:
            astrology_prompt = f"""คุณเป็นโหราจารย์ดิจิทัลผู้เชี่ยวชาญด้านโหราศาสตร์ตะวันตก (Western Astrology)

**⚠️ ข้อกำหนดสำคัญสำหรับ RAG System:**
- **ใช้ข้อมูลจากฐานข้อมูล (MongoDB) เป็นหลัก**
- **อนุญาตให้ใช้ความรู้ทั่วไปทางโหราศาสตร์เพื่อเชื่อมโยงข้อมูลในบริบทกับคำถามได้** (แต่ห้ามยกเมฆข้อมูลใหม่ที่ขัดแย้งกับบริบท)
- **ถ้ามีข้อมูลในบริบทที่ตรงกับราศีของวันเกิด ให้ตอบได้ทันที**

**ข้อกำหนดการเรียกชื่อ:**
- ใช้ชื่อราศีแบบไทย (เช่น มังกร, มกร)
- สำหรับราศีที่ 12 ใช้ "ราศีมีน"
- ใช้คำว่า "ลัคณา" แทน "Ascendant"

{focus_instruction}

**ข้อมูลสำหรับการวิเคราะห์:**
{birth_info}
{chart_info}
{context_info}

**คำถามของผู้ใช้:** {question}

**วิธีการตอบคำถาม:**
1. ใช้ข้อมูลจากส่วน "ข้อมูลที่เกี่ยวข้องจากฐานข้อมูล" มาวิเคราะห์และตอบ
2. **ถ้าเจอบทความเกี่ยวกับราศีที่ตรงกับวันเกิด (เช่น ราศีมกร/มังกร) ให้สรุปข้อมูลนั้นมาตอบได้เลย** ไม่ต้องปฏิเสธว่าไม่เจอวันที่เจาะจง
3. ใช้ภาษาที่เป็นธรรมชาติ อ่อนโยน และเข้าใจง่าย
4. คำลงท้ายต้องใช้ "ค่ะ" เท่านั้น
"""
        
        # สร้าง system prompt
        if astrology_chart:
            system_prompt = """คุณเป็นแชทบอทโหราศาสตร์ตะวันตกที่เชี่ยวชาญในการทำนายดวงชะตาจากวันเดือนปีเกิด 
ตอบคำถามด้วยภาษาที่เป็นมิตร เป็นธรรมชาติ และเข้าใจง่าย ใช้ชื่อราศีแบบไทย (อนุญาตให้ใช้ มกร/มังกร ได้)"""
        else:
            system_prompt = """คุณเป็นแชทบอทโหราศาสตร์ตะวันตกที่เชี่ยวชาญในการทำนายดวงชะตาจากวันเดือนปีเกิด 
ตอบคำถามด้วยภาษาที่เป็นมิตร เป็นธรรมชาติ และเข้าใจง่าย ใช้ชื่อราศีแบบไทย (อนุญาตให้ใช้ มกร/มังกร ได้)"""
        
        # ใช้ชื่อโมเดลจาก ENV
        openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        
        response = client.chat.completions.create(
            model=openai_model,
            messages=[
                {
                    "role": "system", 
                    "content": system_prompt
                },
                {"role": "user", "content": astrology_prompt}
            ],
            temperature=0.7,
            max_tokens=2000
        )
        answer = response.choices[0].message.content.strip()
        print(f"[EVAL] ✔ ได้รับค่าตอบจาก GPT (ความยาว: {len(answer)} ตัวอักษร)")
        
    except Exception as gpt_error:
        # Fallback: ตอบแบบพื้นฐานโดยไม่ใช้ LLM
        try:
            if astrology_chart and astrology_chart.get('zodiac_sign'):
                zodiac = astrology_chart['zodiac_sign']
                birth_date_text = astrology_chart.get('birth_date', '')
                answer = f"วันเกิด: {birth_date_text}\nราศีของคุณคือ ราศี{zodiac}"
            else:
                from .birth_date_parser import BirthDateParser
                parser = BirthDateParser()
                info = parser.extract_birth_info(question)
                if info and info.get('date'):
                    # Use keyword arguments to ensure strict safety
                    chart = parser.generate_birth_chart_info(
                        birth_date=info['date'], 
                        birth_time=info.get('time'), 
                        latitude=info.get('latitude', 13.7563), 
                        longitude=info.get('longitude', 100.5018)
                    )
                    if chart and chart.get('zodiac_sign'):
                        answer = f"วันเกิด: {info['date']}\nราศีของคุณคือ ราศี{chart['zodiac_sign']}"
                    else:
                        answer = "ขออภัยค่ะ ไม่สามารถคำนวณราศีได้ในขณะนี้ กรุณาลองใหม่อีกครั้ง"
                else:
                    answer = "คุณสามารถบอกวันเกิดในรูปแบบ 07/09/2003 เพื่อให้บอกว่าราศีอะไรได้ค่ะ"
        except Exception as e:
            print(f"[ERROR] Error in ask_question_to_rag_for_evaluation: {e}")
            answer = "ขออภัยค่ะ เกิดปัญหาในการประมวลผล กรุณาลองใหม่อีกครั้ง"
    
    # ⚠️ ไม่มีการบันทึกข้อมูลลงฐานข้อมูลสำหรับการประเมิน
    # ⚠️ ไม่มีการแสดงรายงาน terminal (เพื่อลด output)
    
    # Extract text content for Ragas evaluation
    retrieved_contexts_text = [d.get('text', '') for d in (final_used_docs if 'final_used_docs' in locals() else [])]
    
    # 🆕 Inject chart_info into contexts for Ragas Faithfulness check
    # Ragas needs to see the "source of truth" for the calculated data
    if chart_info:
        retrieved_contexts_text.append(f"*** Calculated Astrology Data ***\n{chart_info}")
        print(f"[EVAL] ➕ Injected chart_info into Ragas context ({len(chart_info)} chars)")

    # Debug: Print ALL retrieved contexts
    print(f"\n[EVAL] Final Retrieved Contexts ({len(retrieved_contexts_text)} docs):")
    for idx, txt in enumerate(retrieved_contexts_text):
        snippet = txt[:100].replace('\n', ' ')
        is_pottery = "เครื่องปั้นดินเผา" in txt
        marker = "!!! POTTERY !!!" if is_pottery else ""
        print(f"[EVAL]   [{idx+1}] {snippet}... {marker}")
        if is_pottery:
            print(f"[SUCCESS] Found Pottery Doc at Rank {idx+1}")
            
    return answer, retrieved_contexts_text