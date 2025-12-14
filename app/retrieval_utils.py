import os
import re
import logging
from datetime import datetime, timedelta, time as dt_time
from typing import Tuple
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from .birth_date_parser import generate_astrology_reading, generate_detailed_astrology_reading, extract_birth_info_from_message

# แก้ไขปัญหา MPS device - ใช้ CPU แทน
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# โหลด environment variables
load_dotenv()

# ตั้งค่า Logger
logger = logging.getLogger(__name__)

# ความยาวสูงสุดของ context ที่จะส่งให้ GPT
MAX_CONTEXT_LENGTH = 15000

# Import database configuration
from .multimodel_rag import ORIGINAL_DB_NAME, ORIGINAL_TEXT_COLLECTION, ORIGINAL_IMAGE_COLLECTION, ORIGINAL_TABLE_COLLECTION
from .birth_date_parser import (
    generate_astrology_reading, 
    generate_detailed_astrology_reading, 
    extract_birth_info_from_message,
    get_zodiac_data_from_mongodb
)

# Database สำหรับเก็บข้อมูลผู้ใช้ (user_profiles และ responses)
USER_DB_NAME = "astrobot"
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
            ORIGINAL_TEXT_COLLECTION,
            ORIGINAL_IMAGE_COLLECTION,
            ORIGINAL_TABLE_COLLECTION
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
        logger.error(f"เกิดข้อผิดพลาดในการตรวจสอบการเชื่อมต่อ MongoDB: {e}")
        if connection_info.get('client'):
            try:
                connection_info['client'].close()
            except:
                pass
        return False, f"เกิดข้อผิดพลาดในการตรวจสอบ MongoDB: {e}", connection_info

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
    chart_info: str = None,
):
    """
    แสดงผลสรุปบนเทอร์มินัลในรูปแบบอ่านง่าย เพื่อใช้ประกอบการประเมินด้วย RAGAS
    - สรุปผลการค้นหาและจำนวนเอกสาร
    - แหล่งที่มาพร้อม Similarity (ถ้ามี)
    - ความยาวคำตอบจาก GPT
    - แสดงข้อมูลจาก chart_info (ถ้ามี) เพื่อให้เห็นว่าข้อมูลราศีมาจากไหน
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

        # สรุปแหล่งที่มาของข้อมูล (เฉพาะ similarity >= 0.5000)
        filtered_docs = [doc for doc in valid_docs if isinstance(doc, dict) and doc.get("similarity", 0) >= 0.5000]
        if filtered_docs:
            print("=== สรุปแหล่งที่มาของข้อมูล ===")
            
            for i, doc in enumerate(filtered_docs, 1):
                try:
                    if isinstance(doc, dict):
                        source = doc.get("source", "Unknown source")
                        sim = doc.get("similarity")
                        text_content = doc.get("text", "")
                        
                        # กำหนด emoji ตามประเภทของเอกสาร
                        collection = doc.get("collection", "")
                        if "image" in collection:
                            emoji = "🖼️"
                        elif "table" in collection:
                            emoji = "📊"
                        else:
                            emoji = "📄"
                        
                        if sim is not None:
                            print(f"{emoji} เอกสารที่ {i}: {source} (Similarity: {sim:.4f})")
                        else:
                            print(f"{emoji} เอกสารที่ {i}: {source}")
                        
                        # แสดง context (เนื้อหา) โดยไม่จำกัดความยาว
                        if text_content:
                            print(f"   Context: {text_content}")
                        print()  # บรรทัดว่างระหว่างเอกสาร
                    else:
                        print(f"📄 เอกสารที่ {i}: ข้อมูลทั่วไป")
                except Exception:
                    print(f"❓ เอกสารที่ {i}: ไม่สามารถแสดงรายละเอียดได้")
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
            logger.warning("MONGO_URL ยังไม่ได้ตั้งค่าอย่างถูกต้อง ข้ามการบันทึกคำตอบ")
            return
        
        logger.info(f"🔄 Attempting to store response for user {user_id}, type: {response_type}")
        
        # 🆕 สร้าง embeddings สำหรับ question และ answer เพื่อใช้ใน Semantic Similarity
        try:
            model = SentenceTransformer("minishlab/potion-multilingual-128M", device="cpu")
            question_embedding = model.encode(question, convert_to_numpy=True).tolist()
            answer_embedding = model.encode(answer, convert_to_numpy=True).tolist()
            logger.debug(f"✅ Created embeddings for question and answer (dim: {len(question_embedding)})")
        except Exception as e:
            logger.warning(f"⚠️ ไม่สามารถสร้าง embeddings ได้: {e}")
            question_embedding = None
            answer_embedding = None
        
        mongo_client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000, connectTimeoutMS=5000)
        responses_collection = mongo_client[USER_DB_NAME]["responses"]
        profiles_collection = mongo_client[USER_DB_NAME]["user_profiles"]
        
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
        logger.info(f"✅ บันทึกคำตอบใน astrobot.responses สำเร็จ: {result.inserted_id}")
        
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
        logger.error(f"❌ ไม่สามารถบันทึกคำตอบใน astrobot.responses ได้: {e}")
        logger.error(f"📝 รายละเอียดข้อผิดพลาด - user_id: {user_id}, response_type: {response_type}")
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
        collection = mongo_client[USER_DB_NAME]["user_profiles"]
        
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
        profiles_collection = mongo_client[USER_DB_NAME]["user_profiles"]
        responses_collection = mongo_client[USER_DB_NAME]["responses"]
        
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
        responses_collection = mongo_client[USER_DB_NAME]["responses"]
        profiles_collection = mongo_client[USER_DB_NAME]["user_profiles"]
        
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
        logger.warning(f"เกิดข้อผิดพลาดในการคำนวณความคล้ายคลึงทางความหมาย: {e}")
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
        logger.warning(f"เกิดข้อผิดพลาดในการตรวจสอบ follow-up ด้วยความคล้ายคลึงทางความหมาย: {e}")
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
            logger.warning("ยังไม่ได้ตั้งค่า OpenAI API key ส่งคืนคำถามเดิม")
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
            logger.warning(f"เกิดข้อผิดพลาดในการประเมินตนเองของ query refinement: {eval_error}")
            print(f"\n⚠️  ไม่สามารถประเมินคุณภาพของคำถามที่ปรับปรุงแล้วได้: {eval_error}\n")
        
        return refined_question
        
    except Exception as e:
        logger.warning(f"เกิดข้อผิดพลาดในการปรับปรุงคำถามด้วย LLM: {e}, ส่งคืนคำถามเดิม")
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
            logger.warning("ยังไม่ได้ตั้งค่า OpenAI API key ใช้ความคล้ายคลึงทางความหมายแทน")
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
                logger.warning(f"เกิดข้อผิดพลาดในการประเมินตนเองของการตรวจจับ follow-up: {eval_error}")
                print(f"\n⚠️  ไม่สามารถประเมินความมั่นใจในการตัดสินใจ follow-up ได้: {eval_error}\n")
        
        return is_follow_up
        
    except Exception as e:
        logger.warning(f"เกิดข้อผิดพลาดในการตรวจสอบ follow-up ด้วย LLM: {e}, ใช้ความคล้ายคลึงทางความหมายแทน")
        # ถ้าเกิด error ให้ fallback ไปใช้ semantic similarity
        try:
            is_follow_up, _ = check_follow_up_question_with_semantic_similarity(
                question, user_context, similarity_threshold=0.25
            )
            return is_follow_up
        except:
            # ถ้า semantic similarity ก็ error ให้ return False
            return False

def ask_question_to_rag(question: str, user_id: str = "unknown", provided_chart_info: dict = None, return_retrieved_contexts: bool = False):
    # print(f"\n=== เริ่มการค้นหาข้อมูลสำหรับคำถาม: {question} ===")
    
    # เก็บคำถามเดิมไว้
    original_question = question
    retrieval_question = question  # ใช้คำถามเดิมสำหรับ retrieval
    refined_question_for_prompt = question  # ใช้คำถามเดิมสำหรับ prompt (จะ refine ถ้าเป็น follow-up)
    
    # ตรวจสอบจำนวนคำถามต่อเนื่องก่อน (ไม่จำกัดจำนวนครั้ง)
    is_allowed, current_count, limit_message = check_and_update_question_limit(user_id)
    if not is_allowed:
        logger.info(f"🚫 Question limit exceeded for user {user_id}: {current_count}/3")
        if return_retrieved_contexts:
            return limit_message, []
        return limit_message
    
    # ดึงข้อมูลบริบทการสนทนาของผู้ใช้ก่อน
    user_context = get_user_context(user_id)
    
    # ตรวจสอบว่าเป็นคำถามต่อเนื่องหรือไม่โดยใช้ LLM (ตาม diagram)
    print(f"\n{'='*60}")
    print(f"🔍 กำลังตรวจสอบว่าเป็น Follow-up Question...")
    print(f"{'='*60}")
    print(f"คำถามปัจจุบัน: {original_question}")
    is_follow_up_question = check_follow_up_question_with_llm(original_question, user_context)
    print(f"ผลการตรวจสอบ: {'YES (เป็น follow-up)' if is_follow_up_question else 'NO (ไม่ใช่ follow-up)'}")
    print(f"{'='*60}\n")
    logger.info(f"Follow-up detection (LLM): question='{original_question[:50]}...', is_follow_up={is_follow_up_question}")
    
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
        astrology_chart = generate_detailed_astrology_reading(original_question)
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
    question_intent = analyze_question_intent(original_question)
    
    # 🆕 ปรับปรุง query เมื่อมีข้อมูลราศี - ใช้ชื่อราศีเพื่อให้ค้นหาได้ดีขึ้น
    if astrology_chart and astrology_chart.get('zodiac_sign'):
        zodiac_sign = astrology_chart['zodiac_sign']
        # ตรวจสอบว่าคำถามมีวันเกิดหรือไม่ (เช่น "07/09/2003" หรือ "ทำนายดวง")
        has_birth_date_in_question = bool(birth_info_from_question and birth_info_from_question.get('date'))
        
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
    print(f"คำถามปัจจุบัน: {original_question}")
    is_follow_up_question = check_follow_up_question_with_llm(original_question, user_context)
    print(f"ผลการตรวจสอบ: {'YES (เป็น follow-up)' if is_follow_up_question else 'NO (ไม่ใช่ follow-up)'}")
    print(f"{'='*60}\n")
    logger.info(f"Follow-up detection (LLM): question='{original_question[:50]}...', is_follow_up={is_follow_up_question}")
    
    user_birth_date = user_context.get("birth_date") if user_context else None
    user_zodiac = user_context.get("zodiac_sign") if user_context else None
    
    # ตรวจสอบว่ามีข้อมูลวันเกิดและเวลาเกิดในคำถามหรือไม่ (เสมอ)
    birth_info_from_question = extract_birth_info_from_message(original_question)
    astrology_chart = None
    
    # ถ้ามี chart_info ที่ส่งมา ให้ใช้เลย (กรณีเรียกจาก generate_birth_chart_prediction)
    if provided_chart_info:
        astrology_chart = provided_chart_info
        logger.info(f"ใช้ chart_info ที่ส่งมา: ราศี{astrology_chart.get('zodiac_sign', 'N/A')}")
    
    # สร้างข้อมูลดวงชะตาเมื่อมีข้อมูลวันเกิดในคำถาม (ถ้ายังไม่มี chart_info อยู่แล้ว)
    if not astrology_chart and birth_info_from_question and birth_info_from_question['date']:
        logger.info(f"พบข้อมูลวันเกิดในคำถาม: {birth_info_from_question['date']}")
        if birth_info_from_question['time']:
            logger.info(f"พบเวลาเกิดในคำถาม: {birth_info_from_question['time']}")
        
        # สร้างข้อมูลดวงชะตารายละเอียด
        astrology_chart = generate_detailed_astrology_reading(original_question)
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
        zodiac_english = zodiac_english_map.get(user_zodiac, '')
        if zodiac_english:
            astrology_chart = {
                'zodiac_sign': user_zodiac,
                'zodiac_english': zodiac_english,
                'birth_date': user_birth_date
            }
            logger.info(f"ใช้ข้อมูลดวงชะตาจากบริบท: ราศี{user_zodiac}")
    
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
    question_intent = analyze_question_intent(original_question)
    
    # 🆕 ปรับปรุง query เมื่อมีข้อมูลราศี - ใช้ชื่อราศีเพื่อให้ค้นหาได้ดีขึ้น
    if astrology_chart and astrology_chart.get('zodiac_sign'):
        zodiac_sign = astrology_chart['zodiac_sign']
        # ตรวจสอบว่าคำถามมีวันเกิดหรือไม่ (เช่น "07/09/2003" หรือ "ทำนายดวง")
        has_birth_date_in_question = bool(birth_info_from_question and birth_info_from_question.get('date'))
        
    # ปรับปรุงคำถามให้ชัดเจนขึ้นสำหรับคำถามต่อเนื่องโดยใช้ LLM (สำหรับ prompt เท่านั้น)
    if is_follow_up_question and user_context:
        print(f"\n{'='*60}")
        print(f"🔄 กำลังปรับปรุงคำถามสำหรับ prompt (Refine Query for Prompt)...")
        print(f"{'='*60}")
        print(f"คำถามเดิม: {original_question}")
        refined_question = refine_follow_up_question_with_llm(original_question, user_context)
        if refined_question and refined_question != original_question:
            logger.info(f"Question refined for prompt: '{original_question[:50]}...' -> '{refined_question[:50]}...'")
            refined_question_for_prompt = refined_question
            print(f"✅ คำถามสำหรับ prompt: {refined_question_for_prompt}")
            print(f"✅ คำถามสำหรับ retrieval: {retrieval_question} (ใช้คำถามเดิม)")
            print(f"{'='*60}\n")
        else:
            print(f"คำถามไม่มีการเปลี่ยนแปลง (ไม่จำเป็นต้องปรับปรุง)")
            print(f"{'='*60}\n")
    else:
        if not is_follow_up_question:
            print(f"\n{'='*60}")
            print(f"ℹ️  ไม่ใช่ Follow-up Question - ไม่มีการ Refine Query")
            print(f"{'='*60}\n")
    
    # สร้าง query ที่เฉพาะเจาะจงกับราศีและคำถาม (ใช้ retrieval_question ที่เป็นคำถามเดิม)
    if astrology_chart and astrology_chart.get('zodiac_sign'):
        zodiac_sign = astrology_chart['zodiac_sign']
        has_birth_date_in_question = bool(birth_info_from_question and birth_info_from_question.get('date'))
        
        # ตรวจสอบว่าคำถามต้องการข้อมูลครบทั้ง 4 ด้านหรือไม่ (ใช้ retrieval_question)
        needs_all_aspects = bool(
            has_birth_date_in_question or
            'ทำนายดวง' in retrieval_question or 
            'ดวงชะตา' in retrieval_question or 
            'ดวงกำเนิด' in retrieval_question or 
            'ทำนาย' in retrieval_question or
            ('การงาน' in retrieval_question and 'การเงิน' in retrieval_question and 'ความรัก' in retrieval_question)
        )
        
        if needs_all_aspects:
            # ใช้ query ที่เฉพาะเจาะจงกับราศีนี้และเพิ่มคำสำคัญเกี่ยวกับ 4 ด้าน
            retrieval_question = f"ราศี{zodiac_sign} คนราศี{zodiac_sign} ชาวราศี{zodiac_sign} การงาน อาชีพ หน้าที่การงาน การเงิน การลงทุน การออม เงิน ความรัก ความสัมพันธ์ เนื้อคู่ คู่ สีมงคล สี"
            logger.info(f"ปรับปรุง query สำหรับคำถามที่มีวันเกิด/ทำนายดวง: ใช้ชื่อราศี '{zodiac_sign}' พร้อมคำสำคัญ 4 ด้าน -> '{retrieval_question}'")
        elif 'ราศีอะไร' in retrieval_question or ('ราศี' in retrieval_question and not has_birth_date_in_question):
            # ใช้ query ที่เฉพาะเจาะจงกับราศีนี้
            retrieval_question = f"ราศี{zodiac_sign} คนราศี{zodiac_sign} ชาวราศี{zodiac_sign}"
            logger.info(f"ปรับปรุง query สำหรับชื่อราศีโดยตรง: ใช้ชื่อราศี '{zodiac_sign}' -> '{retrieval_question}'")
        elif 'นิสัย' in retrieval_question or 'บุคลิก' in retrieval_question:
            retrieval_question = f"ราศี{zodiac_sign} คนราศี{zodiac_sign} ชาวราศี{zodiac_sign} ลักษณะนิสัย บุคลิกภาพ"
            logger.info(f"ปรับปรุง query สำหรับคำถามเกี่ยวกับนิสัย: ใช้ชื่อราศี '{zodiac_sign}' -> '{retrieval_question}'")
        else:
            # ถ้ามีชื่อราศีแต่ไม่มี keyword ชัดเจน ให้ใช้ชื่อราศีเป็นหลัก
            retrieval_question = f"ราศี{zodiac_sign} คนราศี{zodiac_sign} ชาวราศี{zodiac_sign} {retrieval_question}"
            logger.info(f"ปรับปรุง query สำหรับชื่อราศี: ใช้ชื่อราศี '{zodiac_sign}' -> '{retrieval_question}'")
    
    # ใช้ refined_question_for_prompt สำหรับการสร้าง prompt ให้ GPT
    # แต่ใช้ retrieval_question สำหรับการค้นหาข้อมูลจาก MongoDB
    question_for_prompt = refined_question_for_prompt
    
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
            query_embedding = model.encode(retrieval_question)
            print(f"✅ สร้าง query embedding สำเร็จ (ขนาด: {len(query_embedding)} dimensions)")
            print(f"📝 Query ที่ใช้สำหรับ retrieval: '{retrieval_question}'")
            
            # ✅ ค้นหาจาก original collections ใน ORIGINAL_DB_NAME
            collections_to_search = [
                ORIGINAL_TEXT_COLLECTION,
                ORIGINAL_IMAGE_COLLECTION,
                ORIGINAL_TABLE_COLLECTION,
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
                                        # ✅ embeddings ถูกสร้างจาก text (ใน multimodel_rag.py)
                                        doc_embedding = np.array(doc['embeddings'])
                                        
                                        # ตรวจสอบว่า dimensions ตรงกัน
                                        if len(doc_embedding) != len(query_embedding):
                                            docs_with_dimension_mismatch += 1
                                            if doc_idx < 3:
                                                print(f"   ⚠️ คำเตือน: ขนาดของ Embedding ไม่ตรงกัน (doc: {len(doc_embedding)}, query: {len(query_embedding)})")
                                            continue
                                        
                                        similarity = np.dot(query_embedding, doc_embedding) / (
                                            np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                                        )
                                        similarities.append((similarity, doc))
                                    except Exception as emb_error:
                                        if doc_idx < 3:
                                            print(f"   ❌ เกิดข้อผิดพลาดในการคำนวณ similarity สำหรับเอกสารที่ {doc_idx+1}: {emb_error}")
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
                                                    doc_info = {
                                                        'text': doc.get('text', ''),
                                                        'source': source_info,
                                                        'similarity': sim,
                                                        'collection': collection_name,
                                                        'doc_id': doc.get('_id'),
                                                        'fallback_query': True  # ระบุว่าเป็น fallback query
                                                    }
                                                    retrieved_docs.append(doc_info)
                                                    print(f"   ✅ เอกสาร fallback ที่ {i+1} (Similarity: {sim:.4f})")
                                    continue
                                
                                # เรียงตาม similarity score
                                similarities.sort(key=lambda x: x[0], reverse=True)
                                
                                # 🆕 กรองและปรับ similarity ตามว่ามีชื่อราศีในเนื้อหาหรือไม่ (ถ้ามีข้อมูลราศี)
                                if astrology_chart and astrology_chart.get('zodiac_sign'):
                                    zodiac_sign = astrology_chart['zodiac_sign']
                                    adjusted_similarities = []
                                    
                                    # ตรวจสอบว่าคำถามต้องการข้อมูลครบทั้ง 4 ด้านหรือไม่
                                    needs_all_aspects = bool(
                                        (birth_info_from_question and birth_info_from_question.get('date')) or
                                        'ทำนายดวง' in retrieval_question or 
                                        'ดวงชะตา' in retrieval_question or 
                                        'ดวงกำเนิด' in retrieval_question or 
                                        'ทำนาย' in retrieval_question or
                                        ('การงาน' in retrieval_question and 'การเงิน' in retrieval_question and 'ความรัก' in retrieval_question)
                                    )
                                    
                                    for sim, doc in similarities:
                                        doc_text = doc.get('text', '')
                                        # ตรวจสอบว่ามีชื่อราศีในเนื้อหาหรือไม่
                                        has_zodiac_in_text = (
                                            zodiac_sign in doc_text or 
                                            f"ราศี{zodiac_sign}" in doc_text or
                                            f"คนราศี{zodiac_sign}" in doc_text or
                                            f"ชาวราศี{zodiac_sign}" in doc_text
                                        )
                                        
                                        if not has_zodiac_in_text:
                                            # ถ้าไม่มีชื่อราศีในเนื้อหา ให้ลด similarity ลงมาก (0.4)
                                            adjusted_sim = max(0, sim - 0.4)
                                        else:
                                            # ไม่มีการเพิ่มคะแนน (เดิม +0.1 ถึง +0.3)
                                            adjusted_sim = sim
                                        
                                        adjusted_similarities.append((adjusted_sim, doc))
                                    
                                    # เรียงลำดับใหม่ตาม adjusted similarity
                                    adjusted_similarities.sort(key=lambda x: x[0], reverse=True)
                                    similarities = adjusted_similarities
                                    print(f"   🔍 ปรับ similarity (เฉพาะการลดคะแนนเมื่อไม่พบชื่อราศี) แล้ว")
                                
                                # เอาข้อมูลที่มี similarity สูงสุด 20 อันดับแรก (เพิ่มจาก 10 เป็น 20 เพื่อให้ได้ข้อมูลมากขึ้นสำหรับคำถามที่มีวันเกิด)
                                top_limit = 20 if (birth_info_from_question and birth_info_from_question.get('date')) else 10
                                top_docs = similarities[:top_limit]
                                print(f"   ✅ คำนวณ similarity สำเร็จ: {len(similarities)} เอกสาร (จาก {len(docs)} เอกสารทั้งหมด)")
                                
                                # แสดง similarity score ทั้งหมด (เฉพาะ top_limit อันดับแรก)
                                if similarities:
                                    print(f"   📊 Similarity scores ({top_limit} อันดับแรก):")
                                    for i, (sim, _) in enumerate(similarities[:top_limit], 1):
                                        print(f"      {i}. {sim:.4f}")
                                
                                # 🆕 คำนวณ threshold ก่อน (ลด threshold เพื่อให้ได้ข้อมูลมากขึ้น)
                                # ใช้ threshold ที่ต่ำกว่าเพื่อให้ได้ข้อมูลที่เกี่ยวข้องมากขึ้น
                                threshold = 0.3500 if (birth_info_from_question and birth_info_from_question.get('date')) else 0.3500
                                
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
                                    
                                    doc_text = doc.get('text', '')
                                    
                                    doc_info = {
                                        'text': doc_text,
                                        'source': source_info,
                                        'similarity': similarity,
                                        'collection': collection_name,
                                        'doc_id': doc.get('_id')
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
                    print(f"✅ ดึงข้อมูลจาก MongoDB เสร็จสิ้น: พบ {len(retrieved_docs)} เอกสารที่ผ่าน threshold")
                    
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
    # ✅ กรองเฉพาะเอกสารที่มี similarity > 0.35 (ลดจาก 0.5 เพื่อให้ระบบยอมรับข้อมูลได้มากขึ้น)
    valid_retrieved_docs = [
        doc for doc in retrieved_docs 
        if isinstance(doc, dict) and doc.get('similarity', 0) > 0.35
    ]
    
    # 🆕 Always Inject: ใส่ข้อมูลราศีจาก birth_date_parser เสมอ (ถ้ามี)
    # เพื่อให้มั่นใจว่ามีข้อมูลที่ถูกต้องแม่นยำที่สุดอยู่ใน Context อันดับแรก (เพราะให้ similarity 1.0)
    # ช่วยแก้ปัญหา Context Recall ตก และเพิ่ม Faithfulness
    if astrology_chart and astrology_chart.get('detailed_reading'):
        zodiac_sign = astrology_chart.get('zodiac_sign')
        reading = astrology_chart.get('detailed_reading')
        
        # แปลงข้อมูล dict เป็นข้อความ text
        reading_text = f"ข้อมูลลักษณะนิสัยชาวราศี{zodiac_sign}: {reading.get('ลักษณะนิสัย', '')}\n"
        reading_text += f"การงานชาวราศี{zodiac_sign}: {reading.get('การงาน', '')}\n"
        reading_text += f"การเงินชาวราศี{zodiac_sign}: {reading.get('การเงิน', '')}\n"
        reading_text += f"ความรักชาวราศี{zodiac_sign}: {reading.get('ความรัก', '')}\n"
        reading_text += f"สุขภาพชาวราศี{zodiac_sign}: {reading.get('สุขภาพ', '')}"
        
        synthetic_doc = {
            'text': reading_text,
            'similarity': 1.0,  # ให้คะแนนเต็มเพราะเป็นข้อมูลตรงตัว
            'source': f"[System] Zodiac Data for {zodiac_sign}",
            'collection': 'zodiac_personality',
            'doc_id': 'synthetic_injected'
        }
        valid_retrieved_docs.append(synthetic_doc)
        print(f"✅ เพิ่มข้อมูลราศี{zodiac_sign} จากฐานข้อมูลโดยตรง (Injection) เพื่อเป็น Context หลัก")

    # ✅ เรียงลำดับข้อมูลตาม similarity จากมากไปน้อย เพื่อให้ Context Precision ดีขึ้น
    # ข้อมูลที่ตรงที่สุด (หรือ fallback ที่ได้ 1.0) จะขึ้นก่อนเสมอ
    valid_retrieved_docs.sort(key=lambda x: x.get('similarity', 0), reverse=True)

    # ตรวจสอบว่ามีเอกสารจาก MongoDB หรือไม่
    # 🆕 ระบบ RAG ต้องใช้ข้อมูลจาก MongoDB ในการตอบคำถาม (ใช้ cosine similarity)
    # ✅ ใช้เฉพาะเอกสารที่มี similarity > 0.35 เท่านั้น
    if not valid_retrieved_docs or len(valid_retrieved_docs) == 0:
        print("\n⚠️ ไม่พบข้อมูลจาก MongoDB ที่มี similarity > 0.35 - ระบบ RAG ต้องใช้ข้อมูลจาก MongoDB ในการตอบคำถาม")
        
        # แสดงรายงานบนเทอร์มินัลสำหรับ RAGAS
        answer = "ขออภัยค่ะ ไม่พบข้อมูลที่เกี่ยวข้องในฐานข้อมูลสำหรับคำถามนี้ กรุณาลองใช้คำถามที่เกี่ยวข้องกับโหราศาสตร์ เช่น 'นิสัยราศีเมถุนเป็นยังไง' หรือ 'สีมงคลราศีสิงห์' ค่ะ"
        
        try:
            print_ragas_terminal_report(
                question=original_question,
                retrieved_docs=retrieved_docs,  # ส่งทั้งเอกสารทั้งหมดรวมถึงที่ต่ำกว่า threshold เพื่อแสดงในรายงาน
                answer=answer,
                user_id=user_id,
                chart_info=chart_info,  # ส่ง chart_info เพื่อแสดงว่าข้อมูลราศีมาจากไหน
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
                question=original_question,
                answer=answer,
                user_id=user_id,
                response_type="no_data_found",
                context_data=context_data
            )
        except Exception:
            pass
        
        if return_retrieved_contexts:
            # Return all retrieved docs even if below threshold, or just empty?
            # Ragas uses retrieved contexts. If we found nothing relevant (>0.5), we return empty list.
            return answer, []
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
            if return_retrieved_contexts:
                return "ขออภัยค่ะ ตอนนี้ระบบยังไม่พร้อมใช้งาน AI ภายนอก แต่คุณสามารถถามเกี่ยวกับราศีได้ตามปกติ เช่น 'นิสัยราศีเมถุนเป็นยังไง' หรือ 'สีมงคลราศีสิงห์'", []
            return "ขออภัยค่ะ ตอนนี้ระบบยังไม่พร้อมใช้งาน AI ภายนอก แต่คุณสามารถถามเกี่ยวกับราศีได้ตามปกติ เช่น 'นิสัยราศีเมถุนเป็นยังไง' หรือ 'สีมงคลราศีสิงห์'"
        client = OpenAI(api_key=openai_key)
        
        # ✅ สร้าง context จากเอกสารที่ค้นหาได้จาก original collections
        # ✅ ระบบ RAG ใช้ cosine similarity กับข้อมูลที่ embed แล้วจาก MongoDB
        # ✅ ใช้เฉพาะเอกสารที่ผ่าน threshold
        context_info = ""
        if valid_retrieved_docs:
            # 🆕 กรองเอกสารให้เฉพาะเจาะจงกับราศีที่ต้องการมากขึ้น
            if astrology_chart and astrology_chart.get('zodiac_sign'):
                zodiac_sign = astrology_chart['zodiac_sign']
                filtered_docs = []
                for doc in valid_retrieved_docs:
                    if isinstance(doc, dict):
                        doc_text = doc.get('text', '')
                        # ตรวจสอบว่ามีชื่อราศีในเนื้อหาหรือไม่
                        has_zodiac_in_text = (
                            zodiac_sign in doc_text or 
                            f"ราศี{zodiac_sign}" in doc_text or
                            f"คนราศี{zodiac_sign}" in doc_text or
                            f"ชาวราศี{zodiac_sign}" in doc_text
                        )
                        if has_zodiac_in_text:
                            filtered_docs.append(doc)
                    else:
                        filtered_docs.append(doc)
                
                # ถ้ามีเอกสารที่เกี่ยวกับราศีที่ต้องการ ให้ใช้เฉพาะเอกสารเหล่านั้น
                original_count = len(valid_retrieved_docs)
                if filtered_docs:
                    valid_retrieved_docs = filtered_docs
                    print(f"🔍 กรองเอกสารให้เฉพาะเจาะจงกับราศี{zodiac_sign}: {len(filtered_docs)} เอกสาร (จาก {original_count} เอกสาร)")
                else:
                    print(f"⚠️ ไม่พบเอกสารที่เกี่ยวกับราศี{zodiac_sign} ในเอกสารที่ผ่าน threshold - ใช้เอกสารทั้งหมด")
            
            context_info = "\n\n**ข้อมูลที่เกี่ยวข้องจากฐานข้อมูล (ค้นหาด้วย cosine similarity จาก embeddings) - ต้องใช้ข้อมูลนี้เท่านั้นในการตอบคำถาม (เฉพาะเอกสารที่มี similarity > 0.35):**\n"
            for i, doc in enumerate(valid_retrieved_docs):
                if isinstance(doc, dict):
                    similarity_score = doc.get('similarity', 0)
                    # ใช้ text จากเอกสารทั้งหมด (ไม่จำกัดความยาว)
                    content_to_use = doc.get('text', '')
                    context_info += f"{i+1}. [Similarity: {similarity_score:.4f}] {content_to_use}\n"
                else:
                    context_info += f"{i+1}. {doc}\n"
            print(f"✅ ใช้ข้อมูลจาก MongoDB (RAG): {len(valid_retrieved_docs)} เอกสาร (เฉพาะ similarity > 0.35)")
        else:
            # ถ้าไม่มีข้อมูลจาก MongoDB ที่มี similarity > 0.35 ให้แจ้งเตือน
            print(f"⚠️ ไม่พบข้อมูลจาก MongoDB ที่มี similarity > 0.35 - ระบบจะตอบตามข้อมูลที่มีอยู่ในฐานข้อมูลเท่านั้น")
            context_info = "\n\n**ข้อมูลที่เกี่ยวข้องจากฐานข้อมูล:**\nไม่พบข้อมูลที่เกี่ยวข้องในฐานข้อมูล กรุณาตรวจสอบว่ามีข้อมูลในฐานข้อมูลหรือไม่\n"
        
        # สร้างข้อมูลดวงชะตาเพิ่มเติม (สำหรับอ้างอิงเท่านั้น - ไม่ใช้ในการตอบคำถาม)
        # หมายเหตุ: chart_info นี้ใช้สำหรับแสดงใน terminal report เท่านั้น ไม่ส่งไปใน prompt
        chart_info = ""
        if astrology_chart:
            # ตรวจสอบว่ามีวันเกิดหรือไม่
            has_birth_date = bool(astrology_chart.get('birth_date'))
            
            # ข้อมูลสถานที่เกิด
            location_info = ""
            if 'birth_location_name' in astrology_chart:
                location_info = f"สถานที่เกิด: {astrology_chart['birth_location_name']}\n"
            elif 'birth_location' in astrology_chart and has_birth_date:
                location_info = f"สถานที่เกิด: กรุงเทพฯ\n"
            
            # สร้าง header ตามว่ามีวันเกิดหรือไม่
            if has_birth_date:
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
- ราศี{astrology_chart['zodiac_sign']} มีธาตุ{astrology_chart['zodiac_element']}

**คำสั่งสำคัญ:**
- ต้องใช้คำว่า "ลัคณา" แทน "Ascendant" ในทุกกรณี
- ต้องใช้ชื่อ "ราศี{astrology_chart['zodiac_sign']}" เท่านั้น ห้ามใช้ "ราศีปลา"
- ต้องใช้ชื่อ "ราศี{astrology_chart['zodiac_sign']}" ในทุกกรณี ห้ามใช้ชื่ออื่น

**ตัวอย่างการใช้งานที่ถูกต้อง:**
- ราศี{astrology_chart['zodiac_sign']} มีลักษณะอ่อนโยน
- คนราศี{astrology_chart['zodiac_sign']} มักจะ...
ธาตุ: {astrology_chart['zodiac_element']}
วันเกิด: {astrology_chart['birth_date']}
เวลาเกิด: {astrology_chart['birth_time'] if astrology_chart.get('birth_time') else 'ไม่ระบุ'}{location_info}อายุ: {astrology_chart.get('age', 'ไม่ระบุ')} ปี

การตีความดวงชะตา:
- ราศี{astrology_chart['zodiac_sign']} เป็นราศีธาตุ{astrology_chart['zodiac_element']}
- ลักษณะเด่นของราศี{astrology_chart['zodiac_sign']} คือ{astrology_chart.get('detailed_reading', {}).get('ลักษณะนิสัย', 'มีเอกลักษณ์เฉพาะตัว')[:50]}...
"""
            else:
                # กรณีไม่มีวันเกิด แต่มีชื่อราศี
                chart_info = f"""
**ข้อมูลราศี:**
ราศี: {astrology_chart['zodiac_sign']} ({astrology_chart['zodiac_english']})
**คำสั่งสำคัญ: ต้องใช้ชื่อ "ราศี{astrology_chart['zodiac_sign']}" เท่านั้น ห้ามใช้คำว่า "ราศีปลา" หรือชื่อสัตว์อื่นๆ**

**ตัวอย่างการใช้งานที่ถูกต้อง:**
- ราศี{astrology_chart['zodiac_sign']} มีลักษณะอ่อนโยน
- คนราศี{astrology_chart['zodiac_sign']} มักจะ...
- ราศี{astrology_chart['zodiac_sign']} เป็นราศีธาตุ{astrology_chart['zodiac_element']}

**คำสั่งเด็ดขาด: ห้ามใช้คำว่า "ราศีปลา" ในคำตอบเด็ดขาด ต้องใช้ "ราศี{astrology_chart['zodiac_sign']}" เท่านั้น**

**ข้อมูลเพิ่มเติม:**
- ราศี{astrology_chart['zodiac_sign']} มีธาตุ{astrology_chart['zodiac_element']}
- คุณภาพ: {astrology_chart.get('zodiac_quality', 'ไม่ระบุ')}

**คำสั่งสำคัญ:**
- ต้องใช้ชื่อ "ราศี{astrology_chart['zodiac_sign']}" เท่านั้น ห้ามใช้ "ราศีปลา"
- ต้องใช้ชื่อ "ราศี{astrology_chart['zodiac_sign']}" ในทุกกรณี ห้ามใช้ชื่ออื่น

**ตัวอย่างการใช้งานที่ถูกต้อง:**
- ราศี{astrology_chart['zodiac_sign']} มีลักษณะอ่อนโยน
- คนราศี{astrology_chart['zodiac_sign']} มักจะ...
ธาตุ: {astrology_chart['zodiac_element']}

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

            # REMOVED: detailed_reading and lucky_colors to enforce strict MongoDB RAG
            # We still keep the calculated zodiac sign and ascendant for identification purposes.



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
- **🚨 ต้องตอบครบทั้ง 4 ด้านเสมอ (ห้ามขาดด้านใดด้านหนึ่ง):**
  * **ด้านการงาน (บังคับ):** ต้องมีข้อมูลเกี่ยวกับอาชีพที่เหมาะ การทำงาน ความสำเร็จในหน้าที่การงาน ทักษะที่โดดเด่น
  * **ด้านการเงิน (บังคับ):** ต้องมีข้อมูลเกี่ยวกับการจัดการเงิน การลงทุน การออม การสร้างความมั่งคั่ง
  * **ด้านความรัก (บังคับ):** ต้องมีข้อมูลเกี่ยวกับความสัมพันธ์ การเข้ากันได้กับคนอื่น คำแนะนำสำหรับคนโสดและคนมีคู่
  * **ด้านสีมงคล (บังคับ):** ต้องมีข้อมูลเกี่ยวกับสีที่เหมาะกับราศี สีที่ควรหลีกเลี่ยง ความหมายของสี สีที่ควรใช้ในชีวิตประจำวัน
- ใช้ข้อมูลจากฐานข้อมูล (MongoDB) ในการตอบคำถาม
- ห้ามตอบเรื่องสุขภาพ
- ตอบเป็นข้อความต่อเนื่องแบบธรรมชาติ ไม่ใช้หัวข้อหรือหมวดหมู่
- **ตอบให้ครอบคลุมทั้ง 4 ด้าน - ไม่จำกัดความยาวของคำตอบ**
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

**⚠️ ข้อกำหนดสำคัญสำหรับ RAG System (ต้องปฏิบัติตามอย่างเคร่งครัด):**
- **🚨 กฎข้อแรกและสำคัญที่สุด: ต้องใช้ข้อมูลจากฐานข้อมูล MongoDB เท่านั้นในการตอบคำถาม**
- **🚨 ห้ามใช้ความรู้ทั่วไปของ GPT ในการตอบคำถาม - ต้องใช้เฉพาะข้อมูลที่แสดงใน "ข้อมูลที่เกี่ยวข้องจากฐานข้อมูล" เท่านั้น**
- **🚨 ห้ามสร้างข้อมูลขึ้นมาเอง - ต้องใช้เฉพาะข้อมูลจาก "ข้อมูลที่เกี่ยวข้องจากฐานข้อมูล" เท่านั้น**
- **🚨 ห้ามบอกว่า "ไม่พบข้อมูล" หรือ "ไม่มีข้อมูล" ถ้ามีข้อมูลใน "ข้อมูลที่เกี่ยวข้องจากฐานข้อมูล"**
- **🚨 ถ้ามีข้อมูลใน "ข้อมูลที่เกี่ยวข้องจากฐานข้อมูล" ให้ใช้ข้อมูลนั้นตอบทันที - ห้ามบอกว่า "ไม่พบข้อมูล"**
- **🚨 ถ้าข้อมูลใน "ข้อมูลที่เกี่ยวข้องจากฐานข้อมูล" ไม่ครบทุกด้าน ให้ตอบเฉพาะด้านที่มีข้อมูล - ห้ามบอกว่า "ไม่พบข้อมูล" สำหรับด้านที่ไม่มี**
- **อนุญาตให้ใช้ข้อมูลจาก chart_info หรือ birth_info เพื่อระบุราศีและพื้นดวงได้ แต่คำทำนายต้องอ้างอิงจากฐานข้อมูล**
- **🚨 ห้ามใช้ข้อมูลจากภายนอกใดๆ - ใช้เฉพาะข้อมูลจาก MongoDB เท่านั้น**
- ข้อมูลที่แสดงใน "ข้อมูลที่เกี่ยวข้องจากฐานข้อมูล" ถูกค้นหาด้วย cosine similarity จาก embeddings และเป็นข้อมูลที่เกี่ยวข้องกับคำถามมากที่สุด
- **ต้องอ้างอิงข้อมูลจากฐานข้อมูลโดยตรง** - ใช้ข้อความหรือประโยคจากข้อมูลที่ค้นหาได้
- **🚨 ห้ามบอกว่า "ไม่พบข้อมูล" หรือ "ไม่มีข้อมูล" ถ้ามีข้อมูลใน "ข้อมูลที่เกี่ยวข้องจากฐานข้อมูล"**
- **🚨 ถ้ามีข้อมูลใน "ข้อมูลที่เกี่ยวข้องจากฐานข้อมูล" ให้ใช้ข้อมูลนั้นตอบทันที - ห้ามบอกว่า "ไม่พบข้อมูล"**
- **🚨 ถ้าข้อมูลใน "ข้อมูลที่เกี่ยวข้องจากฐานข้อมูล" ไม่ครบทุกด้าน ให้ตอบเฉพาะด้านที่มีข้อมูล - ห้ามบอกว่า "ไม่พบข้อมูล" สำหรับด้านที่ไม่มี**
- **🚨 ห้ามบอกว่า "ข้อมูลในฐานข้อมูลไม่ได้กล่าวถึง..." - ถ้ามีข้อมูลใน "ข้อมูลที่เกี่ยวข้องจากฐานข้อมูล" ให้ใช้ข้อมูลนั้นตอบทันที**
- เริ่มต้นคำตอบด้วยการอ้างอิงข้อมูลจากฐานข้อมูล เช่น "ตามข้อมูลในฐานข้อมูล..." หรือ "จากข้อมูลที่เกี่ยวข้อง..."

**ข้อกำหนดสำคัญ:**
- ใช้ชื่อราศีแบบไทยเท่านั้น: เมษ, พฤษภ, เมถุน, กรกฎ, สิงห์, กันย์, ตุล, พิจิก, ธนู, มังกร, กุมภ์, มีน
- ห้ามใช้ชื่อราศีแบบอังกฤษ เช่น Aries, Taurus, Gemini, Cancer, Leo, Virgo, Libra, Scorpio, Sagittarius, Capricorn, Aquarius, Pisces
- ห้ามใช้ชื่อสัตว์ เช่น ราศีปลา, ราศีแกะ, ราศีวัว, ราศีปู, ราศีสิงโต, ราศีแมงป่อง
- สำหรับราศีที่ 12 ต้องใช้ "ราศีมีน" เท่านั้น ห้ามใช้ "ราศีปลา" หรือ "Pisces"
- ใช้คำว่า "ลัคณา" แทน "Ascendant" ในทุกกรณี


**ข้อมูลดวงชะตา (Chart Info):**
{chart_info}

{focus_instruction}

**ข้อมูลสำหรับการวิเคราะห์:**
{context_info}

**🚨 หมายเหตุสำคัญ (อ่านให้ละเอียด):**
- **ต้องใช้เฉพาะข้อมูลจาก "ข้อมูลที่เกี่ยวข้องจากฐานข้อมูล" เท่านั้นในการตอบคำถามทำนายดวง**
- **อนุญาตให้ใช้ข้อมูลจาก "ข้อมูลดวงชะตา (Chart Info)" เพื่อระบุราศีและวันเกิดของผู้ใช้ได้**
- **ห้ามใช้ความรู้ทั่วไปของ GPT ในการตอบคำถาม**
- **ห้ามสร้างข้อมูลขึ้นมาเอง - ต้องใช้เฉพาะข้อมูลจาก MongoDB เท่านั้น**
- **🚨 ห้ามบอกว่า "ไม่พบข้อมูล" หรือ "ไม่มีข้อมูล" ถ้ามีข้อมูลใน "ข้อมูลที่เกี่ยวข้องจากฐานข้อมูล"**
- **🚨 ถ้ามีข้อมูลใน "ข้อมูลที่เกี่ยวข้องจากฐานข้อมูล" ให้ใช้ข้อมูลนั้นตอบทันที - ห้ามบอกว่า "ไม่พบข้อมูล"**
- **🚨 ถ้าข้อมูลใน "ข้อมูลที่เกี่ยวข้องจากฐานข้อมูล" ไม่ครบทุกด้าน ให้ตอบเฉพาะด้านที่มีข้อมูล - ห้ามบอกว่า "ไม่พบข้อมูล" สำหรับด้านที่ไม่มี**
- **ห้ามใช้ข้อมูลจากภายนอกใดๆ - ใช้เฉพาะข้อมูลจาก MongoDB เท่านั้น**

**บริบทการสนทนาก่อนหน้า:**
{get_conversation_context(user_context)}

**คำถามของผู้ใช้:** {question_for_prompt}

**🚨 ข้อกำหนดสำคัญในการตอบคำถาม (อ่านให้ละเอียด):**
- **กฎสำคัญที่สุด: เมื่อคำถามมีวันเดือนปีเกิด (เช่น "07/09/2003", "ทำนายดวง", "ราศีอะไร" พร้อมวันเกิด) → ต้องตอบครบทั้ง 4 ด้านเสมอ (การงาน, การเงิน, ความรัก, สีมงคล) ห้ามขาดด้านใดด้านหนึ่ง**
- **วิเคราะห์คำถามให้ดีก่อนตอบ:**
  * **ถ้าถาม "ทำนายดวง" หรือมีวันเดือนปีเกิด → ต้องตอบครบทั้ง 4 ด้าน (การงาน, การเงิน, ความรัก, สีมงคล)**
  * ถ้าถามว่า "เข้ากับราศีอะไร" หรือ "เข้ากันได้กับราศีอะไร" → ต้องตอบว่าควรเข้ากับราศีอะไร (เช่น ราศีเมษเข้ากับราศีสิงห์ได้ดี)
  * ถ้าถามว่า "อาชีพที่เหมาะ" หรือ "งานที่เหมาะ" → ต้องตอบว่าอาชีพอะไรที่เหมาะกับราศี
  * ถ้าถามว่า "นิสัยเป็นยังไง" → ต้องตอบว่าลักษณะนิสัยของราศีนั้น
  * ถ้าถามว่า "สีมงคล" → ต้องตอบว่าสีอะไรที่เป็นมงคล
- **ห้ามสับสนระหว่างคำถาม** เช่น ถ้าถาม "เข้ากับราศีอะไร" ห้ามตอบว่า "อาชีพที่เหมาะ" หรือ "ลักษณะนิสัย"
- **ตอบให้ตรงประเด็น** แต่ถ้ามีวันเดือนปีเกิด ต้องตอบครบทั้ง 4 ด้านเสมอ

**วิธีการตอบคำถาม (RAG System) - อ่านให้ละเอียด:**
1. **🚨 กฎข้อแรก: อ่านข้อมูลจาก "ข้อมูลที่เกี่ยวข้องจากฐานข้อมูล" ก่อน และใช้เฉพาะข้อมูลนั้นเท่านั้นในการตอบ**
   - ดูข้อมูลในแต่ละ item ที่แสดงใน "ข้อมูลที่เกี่ยวข้องจากฐานข้อมูล"
   - **🚨 ต้องหาข้อมูลที่เกี่ยวกับราศี{astrology_chart['zodiac_sign'] if astrology_chart and astrology_chart.get('zodiac_sign') else 'ที่ถาม'} ใน "ข้อมูลที่เกี่ยวข้องจากฐานข้อมูล"**
   - **🚨 ถ้าพบข้อมูลที่เกี่ยวกับราศี{astrology_chart['zodiac_sign'] if astrology_chart and astrology_chart.get('zodiac_sign') else 'ที่ถาม'} ใน "ข้อมูลที่เกี่ยวข้องจากฐานข้อมูล" ให้ใช้ข้อมูลนั้นตอบทันที - ห้ามบอกว่า "ไม่พบข้อมูล" หรือ "ไม่สามารถช่วยได้"**
   - ใช้ข้อความโดยตรงจากข้อมูลที่เกี่ยวข้องกับคำถาม
   - ห้ามเพิ่มเติมข้อมูลที่ไม่ได้อยู่ในฐานข้อมูล
   - **ห้ามใช้ข้อมูลจาก chart_info หรือ birth_info ในการตอบคำถาม**
   - **ห้ามใช้ความรู้ทั่วไปของ GPT ในการตอบคำถาม**

2. **ขั้นตอนการตอบคำถาม:**
   a. อ่านคำถามให้เข้าใจว่าถามอะไร
   b. **🚨 ค้นหาข้อมูลที่เกี่ยวกับราศี{astrology_chart['zodiac_sign'] if astrology_chart and astrology_chart.get('zodiac_sign') else 'ที่ถาม'} ใน "ข้อมูลที่เกี่ยวข้องจากฐานข้อมูล"**
   c. **🚨 ถ้าพบข้อมูลที่เกี่ยวกับราศี{astrology_chart['zodiac_sign'] if astrology_chart and astrology_chart.get('zodiac_sign') else 'ที่ถาม'} ใน "ข้อมูลที่เกี่ยวข้องจากฐานข้อมูล" ให้ใช้ข้อมูลนั้นตอบทันที - ห้ามบอกว่า "ไม่พบข้อมูล" หรือ "ไม่สามารถช่วยได้"**
   d. **ใช้ข้อความจากข้อมูลที่ค้นหาได้โดยตรง** - ไม่ต้องแปลหรือสรุปใหม่มากเกินไป
   e. **ถ้าข้อมูลไม่ครบทุกด้าน ให้ตอบเฉพาะด้านที่มีข้อมูล - ห้ามบอกว่า "ไม่พบข้อมูล" สำหรับด้านที่ไม่มี**
   f. ตอบให้กระชับและตรงประเด็น - ห้ามยาวเกินไป
   g. **ห้ามใช้ข้อมูลจาก chart_info หรือ birth_info ในการตอบคำถาม**
   h. **ห้ามใช้ความรู้ทั่วไปของ GPT ในการตอบคำถาม**

3. **⚠️ สำหรับคำถามที่มีวันเดือนปีเกิด (บังคับ):** 
   - **ใช้เฉพาะข้อมูลที่พบในฐานข้อมูล (MongoDB) เท่านั้น** - ถ้ามีข้อมูลใน "ข้อมูลที่เกี่ยวข้องจากฐานข้อมูล" ให้ใช้ข้อมูลนั้นตอบทันที - ห้ามบอกว่า "ไม่พบข้อมูล" หรือ "ไม่มีข้อมูล"
   - **🚨 ห้ามบอกว่า "ไม่พบข้อมูล" หรือ "ไม่มีข้อมูล" ถ้ามีข้อมูลใน "ข้อมูลที่เกี่ยวข้องจากฐานข้อมูล"**
   - **🚨 ถ้าข้อมูลใน "ข้อมูลที่เกี่ยวข้องจากฐานข้อมูล" ไม่ครบทุกด้าน ให้ตอบเฉพาะด้านที่มีข้อมูล - ห้ามบอกว่า "ไม่พบข้อมูล" สำหรับด้านที่ไม่มี**
   - **ห้ามใช้ข้อมูลจาก chart_info หรือ birth_info ในการตอบคำถาม**
   - **ห้ามใช้ความรู้ทั่วไปของ GPT ในการตอบคำถาม**
   - ตอบเป็นข้อความต่อเนื่องแบบธรรมชาติ ไม่ใช้หัวข้อหรือหมวดหมู่
   - **ตอบให้ครอบคลุมทั้ง 4 ด้าน - ไม่จำกัดความยาวของคำตอบ**

4. **สำหรับคำถามเฉพาะด้าน:** 
   - **ใช้เฉพาะข้อมูลที่เกี่ยวข้องกับคำถามจากฐานข้อมูลเท่านั้น**
   - ตอบให้กระชับและตรงประเด็น - ไม่เกิน 200 คำ
   - ห้ามเพิ่มข้อมูลที่ไม่ได้อยู่ในฐานข้อมูล

5. **สำหรับคำถามเกี่ยวกับความเข้ากันได้ของราศี:** 
   - **ใช้เฉพาะข้อมูลที่ระบุในฐานข้อมูล** ว่าปราศีไหนเข้ากันได้
   - ห้ามสร้างรายชื่อราศีที่เข้ากันได้ขึ้นมาเอง

6. **สำหรับคำถามต่อเนื่อง:** 
   - ใช้ข้อมูลราศีที่มีอยู่แล้วและตอบคำถามเฉพาะเจาะจง
   - **ใช้เฉพาะข้อมูลจากฐานข้อมูล**

7. **กฎสำคัญ:**
   - **ห้ามสร้างข้อมูลขึ้นมาเอง** - ต้องใช้เฉพาะข้อมูลจากฐานข้อมูล
   - **อ้างอิงข้อมูลโดยตรง** - ใช้ข้อความจากฐานข้อมูล
   - **ตอบให้กระชับ** - ไม่เกิน 200-300 คำ
   - **ตรงประเด็น** - ตอบเฉพาะสิ่งที่ถาม ห้ามเพิ่มเติมข้อมูลที่ไม่เกี่ยวข้อง
   - ใช้ภาษาที่เป็นธรรมชาติ อ่อนโยน และเข้าใจง่าย
   - หลีกเลี่ยงคำทำนายเชิงโชคชะตาเด็ดขาด ใช้คำว่า "มีแนวโน้ม", "สะท้อนว่า"
   - ห้ามใช้ emoji หรือสัญลักษณ์พิเศษใดๆ

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
- **🚨 ห้ามบอกว่า "ไม่พบข้อมูล" หรือ "ไม่มีข้อมูล" ถ้ามีข้อมูลใน "ข้อมูลที่เกี่ยวข้องจากฐานข้อมูล"**
- **🚨 ถ้ามีข้อมูลใน "ข้อมูลที่เกี่ยวข้องจากฐานข้อมูล" ให้ใช้ข้อมูลนั้นตอบทันที - ห้ามบอกว่า "ไม่พบข้อมูล"**
- **🚨 ถ้าข้อมูลใน "ข้อมูลที่เกี่ยวข้องจากฐานข้อมูล" ไม่ครบทุกด้าน ให้ตอบเฉพาะด้านที่มีข้อมูล - ห้ามบอกว่า "ไม่พบข้อมูล" สำหรับด้านที่ไม่มี**
- **ห้ามใช้ความรู้โหราศาสตร์ทั่วไปในการให้คำแนะนำ - ต้องใช้เฉพาะข้อมูลจาก MongoDB เท่านั้น**
- ห้ามใช้ข้อความเช่น "ไม่มีข้อมูลเพิ่มเติม", "ไม่สามารถให้คำแนะนำเฉพาะได้", "ข้อมูลไม่เพียงพอ" ในคำตอบ
- **หากมีข้อมูลดวงชะตาแล้ว ให้ใช้ข้อมูลนั้นในการตอบคำถามทันที ไม่ต้องแจ้งเตือน**
- **ห้ามส่งข้อความแจ้งเตือนใดๆ ในคำตอบ**

**🚨 สรุปข้อกำหนดสำคัญสำหรับคำถามที่มีวันเดือนปีเกิด:**
- ต้องตอบครบทั้ง 4 ด้านเสมอ: (1) การงาน, (2) การเงิน, (3) ความรัก, (4) สีมงคล
- ห้ามขาดด้านใดด้านหนึ่ง
- ใช้ข้อมูลจากฐานข้อมูล (MongoDB) ในการตอบคำถาม
- ตอบเป็นข้อความต่อเนื่องแบบธรรมชาติ ไม่ใช้หัวข้อหรือหมวดหมู่

กรุณาตอบคำถามตามแนวทางที่กำหนดไว้ โดยใช้เฉพาะข้อมูลจากฐานข้อมูล MongoDB เท่านั้น ห้ามใช้ความรู้ทั่วไปของ GPT หรือข้อมูลจากภายนอกใดๆ"""
        else:
            astrology_prompt = f"""คุณเป็นโหราจารย์ดิจิทัลผู้เชี่ยวชาญด้านโหราศาสตร์ตะวันตก (Western Astrology) ที่มีความรู้ลึกซึ้งเกี่ยวกับดาวเคราะห์ ราศี และการตีความดวงกำเนิด

**บทบาทและความเชี่ยวชาญ:**
- คุณเป็นระบบ RAG (Retrieval-Augmented Generation) ที่ใช้ข้อมูลจากฐานข้อมูล MongoDB ในการตอบคำถาม
- ข้อมูลที่ใช้ตอบคำถามถูกค้นหาด้วย cosine similarity จาก embeddings ที่สร้างไว้แล้ว
- คุณมีความเข้าใจในพลังของราศีเกิด และลัคณา (ราศีประจำลัคนา)
- คุณสามารถผสานข้อมูลจากฐานความรู้ (MongoDB) เพื่อสร้างคำทำนายที่เฉพาะตัวและแม่นยำ
- คุณให้คำแนะนำที่อบอุ่น เป็นมิตร และให้กำลังใจ
- คุณสามารถรักษาบริบทการสนทนาและตอบคำถามต่อเนื่องได้อย่างเป็นธรรมชาติ

**⚠️ ข้อกำหนดสำคัญสำหรับ RAG System (ต้องปฏิบัติตามอย่างเคร่งครัด):**
- **🚨 กฎข้อแรกและสำคัญที่สุด: ต้องใช้ข้อมูลจากฐานข้อมูล MongoDB เท่านั้นในการตอบคำถาม**
- **🚨 ห้ามใช้ความรู้ทั่วไปของ GPT ในการตอบคำถาม - ต้องใช้เฉพาะข้อมูลที่แสดงใน "ข้อมูลที่เกี่ยวข้องจากฐานข้อมูล" เท่านั้น**
- **🚨 ห้ามสร้างข้อมูลขึ้นมาเอง - ต้องใช้เฉพาะข้อมูลจาก "ข้อมูลที่เกี่ยวข้องจากฐานข้อมูล" เท่านั้น**
- **🚨 ห้ามบอกว่า "ไม่พบข้อมูล" หรือ "ไม่มีข้อมูล" ถ้ามีข้อมูลใน "ข้อมูลที่เกี่ยวข้องจากฐานข้อมูล"**
- **🚨 ถ้ามีข้อมูลใน "ข้อมูลที่เกี่ยวข้องจากฐานข้อมูล" ให้ใช้ข้อมูลนั้นตอบทันที - ห้ามบอกว่า "ไม่พบข้อมูล"**
- **🚨 ถ้าข้อมูลใน "ข้อมูลที่เกี่ยวข้องจากฐานข้อมูล" ไม่ครบทุกด้าน ให้ตอบเฉพาะด้านที่มีข้อมูล - ห้ามบอกว่า "ไม่พบข้อมูล" สำหรับด้านที่ไม่มี**
- **อนุญาตให้ใช้ข้อมูลจาก chart_info หรือ birth_info เพื่อระบุราศีและพื้นดวงได้ แต่คำทำนายต้องอ้างอิงจากฐานข้อมูล**
- **🚨 ห้ามใช้ข้อมูลจากภายนอกใดๆ - ใช้เฉพาะข้อมูลจาก MongoDB เท่านั้น**
- ข้อมูลที่แสดงใน "ข้อมูลที่เกี่ยวข้องจากฐานข้อมูล" ถูกค้นหาด้วย cosine similarity จาก embeddings และเป็นข้อมูลที่เกี่ยวข้องกับคำถามมากที่สุด
- **ต้องอ้างอิงข้อมูลจากฐานข้อมูลโดยตรง** - ใช้ข้อความหรือประโยคจากข้อมูลที่ค้นหาได้
- **🚨 ห้ามบอกว่า "ไม่พบข้อมูล" หรือ "ไม่มีข้อมูล" ถ้ามีข้อมูลใน "ข้อมูลที่เกี่ยวข้องจากฐานข้อมูล"**
- **🚨 ถ้ามีข้อมูลใน "ข้อมูลที่เกี่ยวข้องจากฐานข้อมูล" ให้ใช้ข้อมูลนั้นตอบทันที - ห้ามบอกว่า "ไม่พบข้อมูล"**
- **🚨 ถ้าข้อมูลใน "ข้อมูลที่เกี่ยวข้องจากฐานข้อมูล" ไม่ครบทุกด้าน ให้ตอบเฉพาะด้านที่มีข้อมูล - ห้ามบอกว่า "ไม่พบข้อมูล" สำหรับด้านที่ไม่มี**
- **🚨 ห้ามบอกว่า "ข้อมูลในฐานข้อมูลไม่ได้กล่าวถึง..." - ถ้ามีข้อมูลใน "ข้อมูลที่เกี่ยวข้องจากฐานข้อมูล" ให้ใช้ข้อมูลนั้นตอบทันที**
- เริ่มต้นคำตอบด้วยการอ้างอิงข้อมูลจากฐานข้อมูล เช่น "ตามข้อมูลในฐานข้อมูล..." หรือ "จากข้อมูลที่เกี่ยวข้อง..."

**ข้อกำหนดสำคัญ:**
- ใช้ชื่อราศีแบบไทยเท่านั้น: เมษ, พฤษภ, เมถุน, กรกฎ, สิงห์, กันย์, ตุล, พิจิก, ธนู, มังกร, กุมภ์, มีน
- ห้ามใช้ชื่อราศีแบบอังกฤษ เช่น Aries, Taurus, Gemini, Cancer, Leo, Virgo, Libra, Scorpio, Sagittarius, Capricorn, Aquarius, Pisces
- ห้ามใช้ชื่อสัตว์ เช่น ราศีปลา, ราศีแกะ, ราศีวัว, ราศีปู, ราศีสิงโต, ราศีแมงป่อง
- สำหรับราศีที่ 12 ต้องใช้ "ราศีมีน" เท่านั้น ห้ามใช้ "ราศีปลา" หรือ "Pisces"
- ใช้คำว่า "ลัคณา" แทน "Ascendant" ในทุกกรณี


**ข้อมูลดวงชะตา (Chart Info):**
{chart_info}

{focus_instruction}

**ข้อมูลสำหรับการวิเคราะห์:**
{context_info}

**🚨 หมายเหตุสำคัญ (อ่านให้ละเอียด):**
- **ต้องใช้เฉพาะข้อมูลจาก "ข้อมูลที่เกี่ยวข้องจากฐานข้อมูล" เท่านั้นในการตอบคำถามทำนายดวง**
- **อนุญาตให้ใช้ข้อมูลจาก "ข้อมูลดวงชะตา (Chart Info)" เพื่อระบุราศีและวันเกิดของผู้ใช้ได้**
- **ห้ามใช้ความรู้ทั่วไปของ GPT ในการตอบคำถาม**
- **ห้ามสร้างข้อมูลขึ้นมาเอง - ต้องใช้เฉพาะข้อมูลจาก MongoDB เท่านั้น**
- **🚨 ห้ามบอกว่า "ไม่พบข้อมูล" หรือ "ไม่มีข้อมูล" ถ้ามีข้อมูลใน "ข้อมูลที่เกี่ยวข้องจากฐานข้อมูล"**
- **🚨 ถ้ามีข้อมูลใน "ข้อมูลที่เกี่ยวข้องจากฐานข้อมูล" ให้ใช้ข้อมูลนั้นตอบทันที - ห้ามบอกว่า "ไม่พบข้อมูล"**
- **🚨 ถ้าข้อมูลใน "ข้อมูลที่เกี่ยวข้องจากฐานข้อมูล" ไม่ครบทุกด้าน ให้ตอบเฉพาะด้านที่มีข้อมูล - ห้ามบอกว่า "ไม่พบข้อมูล" สำหรับด้านที่ไม่มี**
- **ห้ามใช้ข้อมูลจากภายนอกใดๆ - ใช้เฉพาะข้อมูลจาก MongoDB เท่านั้น**

**บริบทการสนทนาก่อนหน้า:**
{get_conversation_context(user_context)}

**คำถามของผู้ใช้:** {question_for_prompt}

**🚨 ข้อกำหนดสำคัญในการตอบคำถาม (อ่านให้ละเอียด):**
- **กฎสำคัญที่สุด: เมื่อคำถามมีวันเดือนปีเกิด (เช่น "07/09/2003", "ทำนายดวง", "ราศีอะไร" พร้อมวันเกิด) → ต้องตอบครบทั้ง 4 ด้านเสมอ (การงาน, การเงิน, ความรัก, สีมงคล) ห้ามขาดด้านใดด้านหนึ่ง**
- **วิเคราะห์คำถามให้ดีก่อนตอบ:**
  * **ถ้าถาม "ทำนายดวง" หรือมีวันเดือนปีเกิด → ต้องตอบครบทั้ง 4 ด้าน (การงาน, การเงิน, ความรัก, สีมงคล)**
  * ถ้าถามว่า "เข้ากับราศีอะไร" หรือ "เข้ากันได้กับราศีอะไร" → ต้องตอบว่าควรเข้ากับราศีอะไร (เช่น ราศีเมษเข้ากับราศีสิงห์ได้ดี)
  * ถ้าถามว่า "อาชีพที่เหมาะ" หรือ "งานที่เหมาะ" → ต้องตอบว่าอาชีพอะไรที่เหมาะกับราศี
  * ถ้าถามว่า "นิสัยเป็นยังไง" → ต้องตอบว่าลักษณะนิสัยของราศีนั้น
  * ถ้าถามว่า "สีมงคล" → ต้องตอบว่าสีอะไรที่เป็นมงคล
- **ห้ามสับสนระหว่างคำถาม** เช่น ถ้าถาม "เข้ากับราศีอะไร" ห้ามตอบว่า "อาชีพที่เหมาะ" หรือ "ลักษณะนิสัย"
- **ตอบให้ตรงประเด็น** แต่ถ้ามีวันเดือนปีเกิด ต้องตอบครบทั้ง 4 ด้านเสมอ

**วิธีการตอบคำถาม (RAG System) - อ่านให้ละเอียด:**
1. **🚨 กฎข้อแรก: อ่านข้อมูลจาก "ข้อมูลที่เกี่ยวข้องจากฐานข้อมูล" ก่อน และใช้เฉพาะข้อมูลนั้นเท่านั้นในการตอบ**
   - ดูข้อมูลในแต่ละ item ที่แสดงใน "ข้อมูลที่เกี่ยวข้องจากฐานข้อมูล"
   - **🚨 ต้องหาข้อมูลที่เกี่ยวกับราศี{astrology_chart['zodiac_sign'] if astrology_chart and astrology_chart.get('zodiac_sign') else 'ที่ถาม'} ใน "ข้อมูลที่เกี่ยวข้องจากฐานข้อมูล"**
   - **🚨 ถ้าพบข้อมูลที่เกี่ยวกับราศี{astrology_chart['zodiac_sign'] if astrology_chart and astrology_chart.get('zodiac_sign') else 'ที่ถาม'} ใน "ข้อมูลที่เกี่ยวข้องจากฐานข้อมูล" ให้ใช้ข้อมูลนั้นตอบทันที - ห้ามบอกว่า "ไม่พบข้อมูล" หรือ "ไม่สามารถช่วยได้"**
   - ใช้ข้อความโดยตรงจากข้อมูลที่เกี่ยวข้องกับคำถาม
   - ห้ามเพิ่มเติมข้อมูลที่ไม่ได้อยู่ในฐานข้อมูล
   - **ห้ามใช้ข้อมูลจาก chart_info หรือ birth_info ในการตอบคำถาม**
   - **ห้ามใช้ความรู้ทั่วไปของ GPT ในการตอบคำถาม**

2. **ขั้นตอนการตอบคำถาม:**
   a. อ่านคำถามให้เข้าใจว่าถามอะไร
   b. **🚨 ค้นหาข้อมูลที่เกี่ยวกับราศี{astrology_chart['zodiac_sign'] if astrology_chart and astrology_chart.get('zodiac_sign') else 'ที่ถาม'} ใน "ข้อมูลที่เกี่ยวข้องจากฐานข้อมูล"**
   c. **🚨 ถ้าพบข้อมูลที่เกี่ยวกับราศี{astrology_chart['zodiac_sign'] if astrology_chart and astrology_chart.get('zodiac_sign') else 'ที่ถาม'} ใน "ข้อมูลที่เกี่ยวข้องจากฐานข้อมูล" ให้ใช้ข้อมูลนั้นตอบทันที - ห้ามบอกว่า "ไม่พบข้อมูล" หรือ "ไม่สามารถช่วยได้"**
   d. **ใช้ข้อความจากข้อมูลที่ค้นหาได้โดยตรง** - ไม่ต้องแปลหรือสรุปใหม่มากเกินไป
   e. **ถ้าข้อมูลไม่ครบทุกด้าน ให้ตอบเฉพาะด้านที่มีข้อมูล - ห้ามบอกว่า "ไม่พบข้อมูล" สำหรับด้านที่ไม่มี**
   f. ตอบให้กระชับและตรงประเด็น - ห้ามยาวเกินไป
   g. **ห้ามใช้ข้อมูลจาก chart_info หรือ birth_info ในการตอบคำถาม**
   h. **ห้ามใช้ความรู้ทั่วไปของ GPT ในการตอบคำถาม**

3. **⚠️ สำหรับคำถามที่มีวันเดือนปีเกิด (บังคับ):** 
   - **ใช้เฉพาะข้อมูลที่พบในฐานข้อมูล (MongoDB) เท่านั้น** - ถ้ามีข้อมูลใน "ข้อมูลที่เกี่ยวข้องจากฐานข้อมูล" ให้ใช้ข้อมูลนั้นตอบทันที - ห้ามบอกว่า "ไม่พบข้อมูล" หรือ "ไม่มีข้อมูล"
   - **🚨 ห้ามบอกว่า "ไม่พบข้อมูล" หรือ "ไม่มีข้อมูล" ถ้ามีข้อมูลใน "ข้อมูลที่เกี่ยวข้องจากฐานข้อมูล"**
   - **🚨 ถ้าข้อมูลใน "ข้อมูลที่เกี่ยวข้องจากฐานข้อมูล" ไม่ครบทุกด้าน ให้ตอบเฉพาะด้านที่มีข้อมูล - ห้ามบอกว่า "ไม่พบข้อมูล" สำหรับด้านที่ไม่มี**
   - **ห้ามใช้ข้อมูลจาก chart_info หรือ birth_info ในการตอบคำถาม**
   - **ห้ามใช้ความรู้ทั่วไปของ GPT ในการตอบคำถาม**
   - ตอบเป็นข้อความต่อเนื่องแบบธรรมชาติ ไม่ใช้หัวข้อหรือหมวดหมู่
   - **ตอบให้ครอบคลุมทั้ง 4 ด้าน - ไม่จำกัดความยาวของคำตอบ**

4. **สำหรับคำถามเฉพาะด้าน:** 
   - **ใช้เฉพาะข้อมูลที่เกี่ยวข้องกับคำถามจากฐานข้อมูลเท่านั้น**
   - ตอบให้กระชับและตรงประเด็น - ไม่เกิน 200 คำ
   - ห้ามเพิ่มข้อมูลที่ไม่ได้อยู่ในฐานข้อมูล

5. **สำหรับคำถามเกี่ยวกับความเข้ากันได้ของราศี:** 
   - **ใช้เฉพาะข้อมูลที่ระบุในฐานข้อมูล** ว่าปราศีไหนเข้ากันได้
   - ห้ามสร้างรายชื่อราศีที่เข้ากันได้ขึ้นมาเอง

6. **สำหรับคำถามต่อเนื่อง:** 
   - ใช้ข้อมูลราศีที่มีอยู่แล้วและตอบคำถามเฉพาะเจาะจง
   - **ใช้เฉพาะข้อมูลจากฐานข้อมูล**

7. **กฎสำคัญ:**
   - **ห้ามสร้างข้อมูลขึ้นมาเอง** - ต้องใช้เฉพาะข้อมูลจากฐานข้อมูล
   - **อ้างอิงข้อมูลโดยตรง** - ใช้ข้อความจากฐานข้อมูล
   - **ตอบให้กระชับ** - ไม่เกิน 200-300 คำ
   - **ตรงประเด็น** - ตอบเฉพาะสิ่งที่ถาม ห้ามเพิ่มเติมข้อมูลที่ไม่เกี่ยวข้อง
   - ใช้ภาษาที่เป็นธรรมชาติ อ่อนโยน และเข้าใจง่าย
   - หลีกเลี่ยงคำทำนายเชิงโชคชะตาเด็ดขาด ใช้คำว่า "มีแนวโน้ม", "สะท้อนว่า"
   - ห้ามใช้ emoji หรือสัญลักษณ์พิเศษใดๆ

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
- **🚨 ห้ามบอกว่า "ไม่พบข้อมูล" หรือ "ไม่มีข้อมูล" ถ้ามีข้อมูลใน "ข้อมูลที่เกี่ยวข้องจากฐานข้อมูล"**
- **🚨 ถ้ามีข้อมูลใน "ข้อมูลที่เกี่ยวข้องจากฐานข้อมูล" ให้ใช้ข้อมูลนั้นตอบทันที - ห้ามบอกว่า "ไม่พบข้อมูล"**
- **🚨 ถ้าข้อมูลใน "ข้อมูลที่เกี่ยวข้องจากฐานข้อมูล" ไม่ครบทุกด้าน ให้ตอบเฉพาะด้านที่มีข้อมูล - ห้ามบอกว่า "ไม่พบข้อมูล" สำหรับด้านที่ไม่มี**
- **ห้ามใช้ความรู้โหราศาสตร์ทั่วไปในการให้คำแนะนำ - ต้องใช้เฉพาะข้อมูลจาก MongoDB เท่านั้น**
- ห้ามใช้ข้อความเช่น "ไม่มีข้อมูลเพิ่มเติม", "ไม่สามารถให้คำแนะนำเฉพาะได้", "ข้อมูลไม่เพียงพอ" ในคำตอบ

**🚨 สรุปข้อกำหนดสำคัญสำหรับคำถามที่มีวันเดือนปีเกิด:**
- ต้องตอบครบทั้ง 4 ด้านเสมอ: (1) การงาน, (2) การเงิน, (3) ความรัก, (4) สีมงคล
- ห้ามขาดด้านใดด้านหนึ่ง
- ใช้ข้อมูลจากฐานข้อมูล (MongoDB) ในการตอบคำถาม
- ตอบเป็นข้อความต่อเนื่องแบบธรรมชาติ ไม่ใช้หัวข้อหรือหมวดหมู่

กรุณาตอบคำถามตามแนวทางที่กำหนดไว้ โดยใช้เฉพาะข้อมูลจากฐานข้อมูล MongoDB เท่านั้น ห้ามใช้ความรู้ทั่วไปของ GPT หรือข้อมูลจากภายนอกใดๆ"""
        
        # สร้าง system prompt ที่เหมาะสม
        if astrology_chart:
            system_prompt = f"""คุณเป็นแชทบอทโหราศาสตร์ตะวันตกที่เชี่ยวชาญในการทำนายดวงชะตาจากวันเดือนปีเกิด 

**🚨 ข้อกำหนดสำคัญที่สุด (ต้องปฏิบัติตามอย่างเคร่งครัด):**
- **ห้ามดึงข้อมูลจากภายนอกใดๆ - ต้องใช้เฉพาะข้อมูลจากฐานข้อมูล MongoDB เท่านั้น**
- **ห้ามใช้ความรู้ทั่วไปของ GPT ในการตอบคำถาม**
- **ห้ามสร้างข้อมูลขึ้นมาเอง - ต้องใช้เฉพาะข้อมูลจาก "ข้อมูลที่เกี่ยวข้องจากฐานข้อมูล" เท่านั้น**
- **อนุญาตให้ใช้ข้อมูลจาก "ข้อมูลดวงชะตา (Chart Info)" เพื่อระบุราศีและพื้นดวงได้**

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

**🚨 ข้อกำหนดสำคัญที่สุด (ต้องปฏิบัติตามอย่างเคร่งครัด):**
- **ห้ามดึงข้อมูลจากภายนอกใดๆ - ต้องใช้เฉพาะข้อมูลจากฐานข้อมูล MongoDB เท่านั้น**
- **ห้ามใช้ความรู้ทั่วไปของ GPT ในการตอบคำถาม**
- **ห้ามสร้างข้อมูลขึ้นมาเอง - ต้องใช้เฉพาะข้อมูลจาก "ข้อมูลที่เกี่ยวข้องจากฐานข้อมูล" เท่านั้น**
- **อนุญาตให้ใช้ข้อมูลจาก "ข้อมูลดวงชะตา (Chart Info)" เพื่อระบุราศีและพื้นดวงได้**

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
        
        # กำหนด max_tokens ตามประเภทคำถาม
        # ถ้ามีวันเกิด ต้องตอบครบทั้ง 4 ด้าน จึงต้องใช้ tokens มากขึ้น
        if birth_info_from_question and birth_info_from_question.get('date'):
            max_tokens_value = 2000  # เพิ่ม tokens สำหรับตอบครบทั้ง 4 ด้าน (การงาน, การเงิน, ความรัก, สีมงคล)
        else:
            max_tokens_value = 400   # จำกัดความยาวสำหรับคำถามทั่วไป
        
        response = client.chat.completions.create(
            model=openai_model,
            messages=[
                {
                    "role": "system", 
                    "content": system_prompt
                },
                {"role": "user", "content": astrology_prompt}
            ],
            temperature=0.3,  # ลดลงเพื่อให้คำตอบสอดคล้องและใช้ข้อมูลจากฐานข้อมูลมากขึ้น
            max_tokens=max_tokens_value
        )
        answer = response.choices[0].message.content
        # print(f"ได้รับคำตอบจาก GPT (ความยาว: {len(answer)} ตัวอักษร)")
        
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
                    chart = parser.generate_birth_chart_info(info['date'], info.get('time'), info.get('latitude', 13.7563), info.get('longitude', 100.5018))
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
            question=original_question,
            retrieved_docs=retrieved_docs,  # ส่งทั้งเอกสารทั้งหมดรวมถึงที่ต่ำกว่า threshold
            answer=answer,
            user_id=user_id,
            chart_info=chart_info,  # ส่ง chart_info เพื่อแสดงว่าข้อมูลราศีมาจากไหน
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
            question=original_question,
            user_id=user_id,
            context_data=context_data
        )
        
        log_user_interaction(
            question=original_question,
            answer=answer,
            embedding=query_vector,
            user_id=user_id,
            context_data=context_data
        )
        
        # บันทึกคำตอบใน collection astrobot
        store_user_response(
            question=original_question,
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
    if return_retrieved_contexts:
        # Return list of texts
        # Note: valid_retrieved_docs matches the contexts used for generation
        contexts = [d.get('text', '') for d in valid_retrieved_docs] if 'valid_retrieved_docs' in locals() else []
        return answer, contexts
    return answer