import os
import logging
from datetime import datetime, timedelta, time as dt_time
from typing import Tuple
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from .birth_date_parser import generate_astrology_reading, generate_detailed_astrology_reading, extract_birth_info_from_message

# โหลด environment variables
load_dotenv()

# ตั้งค่า Logger
logger = logging.getLogger(__name__)

# Import database configuration
from config import SUMMARY_DB_NAME
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
            print("ไม่พบข้อมูลที่เกี่ยวข้อง -> ใช้ความรู้ทั่วไป (No-RAG)")
        print("==== เสร็จสิ้นการค้นหา ===\n")

        # แสดงข้อมูลที่ใช้จากฐานข้อมูล
        if total_found > 0:
            print(f"🗄️ ใช้ข้อมูลจากฐานข้อมูล: {total_found} เอกสาร")
            print("💬 กำลังส่งคำถามไปยัง GPT...")
        
        # GPT Response (แสดงแค่ความยาว ไม่แสดงคำตอบ)
        ans_len = len(answer) if isinstance(answer, str) else 0
        if ans_len > 0:
            print(f"✔ ได้รับค่าตอบจาก GPT (ความยาว: {ans_len} ตัวอักษร)\n")

        # สรุปแหล่งที่มาของข้อมูล
        if total_found:
            print("=== สรุปแหล่งที่มาของข้อมูล ===")
            for i, doc in enumerate(valid_docs, 1):
                try:
                    if isinstance(doc, dict):
                        source = doc.get("source", "Unknown source")
                        sim = doc.get("similarity")
                        
                        # กำหนด emoji ตามประเภทของเอกสาร
                        collection = doc.get("collection", "")
                        if "image" in collection:
                            emoji = "🖼️"
                        else:
                            emoji = "📄"
                        
                        if sim is not None:
                            print(f"{emoji} เอกสารที่ {i}: {source} (Similarity: {sim:.4f})")
                        else:
                            print(f"{emoji} เอกสารที่ {i}: {source}")
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
            logger.warning("MONGO_URL not configured properly, skipping response storage")
            return
        
        logger.info(f"🔄 Attempting to store response for user {user_id}, type: {response_type}")
        
        # 🆕 สร้าง embeddings สำหรับ question และ answer เพื่อใช้ใน Semantic Similarity
        try:
            model = SentenceTransformer("all-MiniLM-L6-v2")
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
        collection = mongo_client[SUMMARY_DB_NAME]["user_profiles"]
        
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
                "updated_at": user_profile.get("updated_at")
            })
        
        # ข้อมูลจาก responses collection
        if latest_response:
            context.update({
                "last_question": latest_response.get("question"),
                "last_response": latest_response.get("answer"),
                "last_response_type": latest_response.get("response_type"),
                "last_response_time": latest_response.get("created_at")
            })
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

# ✔️ ดึงข้อมูลจาก SUMMARY_DB_NAME เท่านั้น (ไม่ดึงจาก original)
def get_summary_content(doc_id, collection_name):
    """
    ดึงข้อมูลจาก SUMMARY_DB_NAME โดยใช้ doc_id
    """
    try:
        if not doc_id:
            return None
            
        mongo_uri = os.getenv("MONGO_URL")
        if not mongo_uri or mongo_uri == "mongodb+srv://your-username:your-password@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority":
            print("MONGO_URL not configured properly. Please set up your .env file with valid MongoDB connection string.")
            return None
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000, connectTimeoutMS=5000)
        collection = client[SUMMARY_DB_NAME][collection_name]
        
        summary_doc = collection.find_one({"_id": doc_id})
        client.close()
        
        if summary_doc:
            # print(f"ดึงข้อมูลจาก summary สำเร็จ: {collection_name}")
            return summary_doc
        else:
            # print(f"ไม่พบข้อมูลใน summary: {collection_name}")
            return None
            
    except Exception as e:
        # print(f"ไม่สามารถดึงข้อมูลจาก summary ได้: {e}")
        return None


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
            model = SentenceTransformer("all-MiniLM-L6-v2")
        
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
        model = SentenceTransformer("all-MiniLM-L6-v2")
        
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
    is_follow_up_question = check_follow_up_question_with_llm(question, user_context)
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
    
    # ปรับปรุงคำถามให้ชัดเจนขึ้นสำหรับคำถามต่อเนื่องโดยใช้ LLM
    if is_follow_up_question and user_context:
        refined_question = refine_follow_up_question_with_llm(question, user_context)
        if refined_question and refined_question != question:
            logger.info(f"Question refined: '{question[:50]}...' -> '{refined_question[:50]}...'")
            question = refined_question
    
    # ลองค้นหาจาก MongoDB แบบ Manual Search
    retrieved_docs = []
    try:
        # print("กำลังค้นหาจาก MongoDB แบบ Manual Search...")
        
        # โหลด embedding model
        import numpy as np
        
        model = SentenceTransformer("all-MiniLM-L6-v2")
        query_embedding = model.encode(question)
        # print(f"สร้าง query embedding สำเร็จ (ขนาด: {len(query_embedding)})")
        
        # ✅ ค้นหาจาก processed collections ใน SUMMARY_DB_NAME เท่านั้น (ใช้ summary และ embeddings)
        # ✅ ไม่ใช้ original collections (เก็บต้นฉบับเท่านั้น ไม่มี embeddings)
        # ต้องตรงกับชื่อ collection ที่ pipeline multimodel_rag สร้างไว้
        collections_to_search = [
            "processed_text_chunks",      # ✅ มี summary และ embeddings
            "processed_image_chunks",     # ✅ มี summary, embeddings (text), และ image_embeddings
            "processed_table_chunks",     # ✅ มี summary และ embeddings
        ]
        
        for collection_name in collections_to_search:
            try:
                # print(f"ค้นหาใน collection: {collection_name} (SUMMARY_DB)")
                mongo_uri = os.getenv("MONGO_URL")
                if not mongo_uri or mongo_uri == "mongodb+srv://your-username:your-password@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority":
                    # print("MONGO_URL not configured properly. Please set up your .env file with valid MongoDB connection string.")
                    continue
                client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000, connectTimeoutMS=5000)
                collection = client[SUMMARY_DB_NAME][collection_name]
                
                # ดึงข้อมูลทั้งหมด
                docs = list(collection.find({}))
                # print(f"จำนวนเอกสารใน {collection_name}: {len(docs)}")
                
                if docs:
                    # ✅ คำนวณ similarity scores (ใช้ embeddings ที่สร้างจาก summary)
                    # ✅ embeddings ใน processed chunks ถูกสร้างจาก summary text (ไม่ใช่ text ต้นฉบับ)
                    similarities = []
                    for doc in docs:
                        if 'embeddings' in doc:
                            # ✅ embeddings ถูกสร้างจาก summary text (ใน multimodel_rag.py)
                            doc_embedding = np.array(doc['embeddings'])
                            similarity = np.dot(query_embedding, doc_embedding) / (
                                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                            )
                            similarities.append((similarity, doc))
                    
                    # เรียงตาม similarity score
                    similarities.sort(key=lambda x: x[0], reverse=True)
                    
                    # เอาข้อมูลที่มี similarity สูงสุด 2 อันดับแรก
                    top_docs = similarities[:2]
                    # print(f"พบเอกสารที่เกี่ยวข้องใน {collection_name}: {len(top_docs)} เอกสาร")
                    
                    # แสดง similarity score สูงสุด
                    if top_docs:
                        max_similarity = top_docs[0][0]
                        # print(f"Similarity score สูงสุด: {max_similarity:.4f}")
                        pass
                    
                    for i, (similarity, doc) in enumerate(top_docs):
                        # เพิ่มข้อมูล source
                        source_info = f"[{collection_name}]"
                        if 'page' in doc:
                            source_info += f" หน้า {doc['page']}"
                        if 'chunk_id' in doc:
                            source_info += f" Chunk {doc['chunk_id']}"
                        if 'type' in doc:
                            source_info += f" ({doc['type']})"
                        
                        # ใช้ข้อมูลจาก summary database เท่านั้น
                        summary_content = get_summary_content(doc.get('_id'), collection_name)
                        
                        doc_info = {
                            'text': doc['text'],
                            'summary': doc.get('summary', ''),
                            'summary_content': summary_content,
                            'source': source_info,
                            'similarity': similarity,
                            'collection': collection_name,
                            'doc_id': doc.get('_id')
                        }
                        
                        # เพิ่มเอกสารทั้งหมด แต่ mark ว่าต่ำกว่า threshold หรือไม่
                        if similarity > 0.2:  # ลด threshold จาก 0.3 เป็น 0.2
                            # print(f"\nเอกสารที่ {i+1} จาก {collection_name} (Similarity: {similarity:.4f}):")
                            # print(f"   เนื้อหา: {doc['text'][:200]}...")
                            # print(f"   แหล่งที่มา: {source_info}")
                            retrieved_docs.append(doc_info)
                        else:
                            # print(f"เอกสารที่ {i+1} มี similarity ต่ำเกินไป: {similarity:.4f}")
                            # เพิ่มเอกสารที่ต่ำกว่า threshold เพื่อแสดงใน terminal
                            doc_info['below_threshold'] = True
                            retrieved_docs.append(doc_info)
                
                client.close()
                    
            except Exception as e:
                # print(f"ไม่สามารถค้นหาใน {collection_name} ได้: {e}")
                continue
                
    except Exception as e:
        # print(f"ไม่สามารถค้นหาจาก MongoDB ได้: {e}")
        # print("ใช้ GPT โดยตรงแทน")
        pass
    
    # หมายเหตุ: รายงานสรุปจะพิมพ์หลังจากได้คำตอบแล้ว เพื่อรวมความยาวคำตอบด้วย
    
    # ใช้ direct GPT เพราะ vector store ไม่มีข้อมูล
    query_vector = []
    # กำหนดธงสำหรับสร้างคำถามต่อเนื่องอัตโนมัติเมื่อมีข้อมูลวันเกิดในคำถาม
    should_create_chart = bool(birth_info_from_question and birth_info_from_question.get('date'))

    # ใช้ GPT โดยตรง (ไม่ใช้ RAG เพราะ vector store ไม่มีข้อมูล)
    try:
        from openai import OpenAI
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key or openai_key == "your-openai-api-key-here":
            # ถ้าไม่ตั้งค่า API key ให้ตอบแบบ fallback ทั่วไปแทนการเรียก LLM
            return "ขออภัยค่ะ ตอนนี้ระบบยังไม่พร้อมใช้งาน AI ภายนอก แต่คุณสามารถถามเกี่ยวกับราศีได้ตามปกติ เช่น 'นิสัยราศีเมถุนเป็นยังไง' หรือ 'สีมงคลราศีสิงห์'"
        client = OpenAI(api_key=openai_key)
        
        # ✅ สร้าง context จากเอกสารที่ค้นหาได้จาก processed collections เท่านั้น
        # ✅ ระบบใช้ summary embeddings ในการค้นหา และใช้ summary ในการสร้างคำตอบ
        # ✅ ไม่ใช้ original collections (ไม่มี embeddings)
        context_info = ""
        if retrieved_docs:
            context_info = "\n\nข้อมูลที่เกี่ยวข้องจากฐานข้อมูล astrobot_summary (ใช้ summary และ embeddings):\n"
            for i, doc in enumerate(retrieved_docs):
                if isinstance(doc, dict):
                    # ✅ ใช้ summary เป็นหลัก (เพราะ embeddings ถูกสร้างจาก summary)
                    # ✅ summary กระชับและมีความหมายมากกว่า text ต้นฉบับ
                    content_to_use = doc.get('summary', doc.get('text', ''))
                    context_info += f"{i+1}. {content_to_use[:300]}...\n"
                    
                    # เพิ่มข้อมูลจาก summary database ถ้ามี
                    if doc.get('summary_content'):
                        summary_text = doc['summary_content'].get('text', '')
                        if summary_text and len(summary_text) > 100:
                            context_info += f"   ข้อมูลเพิ่มเติมจาก summary: {summary_text[:200]}...\n"
                else:
                    context_info += f"{i+1}. {doc[:300]}...\n"
            # print(f"ใช้ข้อมูลจาก astrobot_summary: {len(retrieved_docs)} เอกสาร")
        else:
            # print("ไม่พบเอกสารที่เกี่ยวข้องใน astrobot_summary จะใช้ความรู้ทั่วไปในการตอบ")
            # print("ข้อแนะนำ: ลองใช้คำถามที่เกี่ยวข้องกับโหราศาสตร์มากขึ้น เช่น 'ราศี', 'ดาวเคราะห์', 'ดวงชะตา'")
            pass
        
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

            # เพิ่มข้อมูลรายละเอียดลักษณะนิสัย การงาน การเงิน ความรัก (เฉพาะ 4 ด้าน)
            if 'detailed_reading' in astrology_chart:
                detailed = astrology_chart['detailed_reading']
                chart_info += f"""

**การทำนายรายละเอียดสำหรับราศี{astrology_chart['zodiac_sign']}:**

ลักษณะนิสัยและบุคลิกภาพ:
{detailed.get('ลักษณะนิสัย', detailed.get('personality_traits', 'ไม่มีข้อมูลลักษณะนิสัย'))}

ด้านการงาน:
{detailed.get('การงาน', 'ไม่มีข้อมูลการงาน')}

ด้านการเงิน:
{detailed.get('การเงิน', 'ไม่มีข้อมูลการเงิน')}

ด้านความรัก:
"""
                if isinstance(detailed.get('ความรัก'), dict):
                    love_info = detailed['ความรัก']
                    chart_info += f"""
คนโสด: {love_info.get('คนโสด', 'ไม่มีข้อมูล')}
คนมีคู่:  {love_info.get('คนมีคู่', 'ไม่มีข้อมูล')}
"""
                else:
                    chart_info += f"{detailed.get('ความรัก', 'ไม่มีข้อมูลความรัก')}"


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
- คุณมีความเข้าใจในพลังของราศีเกิด และลัคณา (ราศีประจำลัคนา)
- คุณสามารถผสานข้อมูลจากฐานความรู้เพื่อสร้างคำทำนายที่เฉพาะตัว
- คุณให้คำแนะนำที่อบอุ่น เป็นมิตร และให้กำลังใจ
- คุณสามารถรักษาบริบทการสนทนาและตอบคำถามต่อเนื่องได้อย่างเป็นธรรมชาติ

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

**วิธีการตอบคำถาม:**
1. **สำหรับคำถามใหม่:** เริ่มต้นด้วยการระบุวันเกิดและราศีเกิดอย่างชัดเจน
2. **สำหรับคำถามทั่วไปเกี่ยวกับดวงชะตา:** ต้องตอบครบทั้ง 4 ด้าน (ลักษณะนิสัยและบุคลิกภาพ, การงาน, การเงิน, ความรัก) เพื่อให้คำทำนายที่สมบูรณ์
3. **สำหรับคำถามเฉพาะด้าน:** ตอบเฉพาะด้านที่ถามเท่านั้น (ถ้าถามเกี่ยวกับการงาน ก็ตอบเฉพาะการงาน เท่านั้น)
4. **สำหรับคำถามต่อเนื่อง:** ใช้ข้อมูลราศีที่มีอยู่แล้วและตอบคำถามเฉพาะเจาะจง
5. อธิบายลักษณะนิสัยตามราศีและธาตุ โดยอ้างอิงจากข้อมูลในฐานความรู้
6. **หากมีข้อมูล Ascendant:** ใช้ข้อมูล Ascendant เพื่อเพิ่มความแม่นยำในการทำนายบุคลิกภาพ
7. ใช้ภาษาที่เป็นธรรมชาติ อ่อนโยน และเข้าใจง่าย
8. หลีกเลี่ยงคำทำนายเชิงโชคชะตาเด็ดขาด ใช้คำว่า "มีแนวโน้ม", "สะท้อนว่า", "บ่งบอกถึงพลังของ..."
9. ตอบเป็นข้อความต่อเนื่องแบบธรรมชาติ ไม่ใช้หัวข้อหรือหมวดหมู่
10. ห้ามใช้ emoji หรือสัญลักษณ์พิเศษใดๆ
11. **สำหรับคำถามต่อเนื่อง:** อย่าเปลี่ยนราศีหรือข้อมูลวันเกิด ให้ใช้ข้อมูลเดิมที่ผู้ใช้ให้มา

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

กรุณาตอบคำถามตามแนวทางที่กำหนดไว้ โดยใช้ความรู้โหราศาสตร์ตะวันตกและให้คำแนะนำที่เป็นประโยชน์"""
        else:
            astrology_prompt = f"""คุณเป็นโหราจารย์ดิจิทัลผู้เชี่ยวชาญด้านโหราศาสตร์ตะวันตก (Western Astrology) ที่มีความรู้ลึกซึ้งเกี่ยวกับดาวเคราะห์ ราศี และการตีความดวงกำเนิด

**บทบาทและความเชี่ยวชาญ:**
- คุณมีความเข้าใจในพลังของราศีเกิด และลัคณา (ราศีประจำลัคนา)
- คุณสามารถผสานข้อมูลจากฐานความรู้เพื่อสร้างคำทำนายที่เฉพาะตัว
- คุณให้คำแนะนำที่อบอุ่น เป็นมิตร และให้กำลังใจ
- คุณสามารถรักษาบริบทการสนทนาและตอบคำถามต่อเนื่องได้อย่างเป็นธรรมชาติ

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

**วิธีการตอบคำถาม:**
1. **สำหรับคำถามใหม่:** เริ่มต้นด้วยการระบุวันเกิดและราศีเกิดอย่างชัดเจน
2. **สำหรับคำถามทั่วไปเกี่ยวกับดวงชะตา:** ต้องตอบครบทั้ง 4 ด้าน (ลักษณะนิสัยและบุคลิกภาพ, การงาน, การเงิน, ความรัก) เพื่อให้คำทำนายที่สมบูรณ์
3. **สำหรับคำถามเฉพาะด้าน:** ตอบเฉพาะด้านที่ถามเท่านั้น (ถ้าถามเกี่ยวกับการงาน ก็ตอบเฉพาะการงาน เท่านั้น)
4. **สำหรับคำถามต่อเนื่อง:** ใช้ข้อมูลราศีที่มีอยู่แล้วและตอบคำถามเฉพาะเจาะจง
5. อธิบายลักษณะนิสัยตามราศีและธาตุ โดยอ้างอิงจากข้อมูลในฐานความรู้
6. **หากมีข้อมูล Ascendant:** ใช้ข้อมูล Ascendant เพื่อเพิ่มความแม่นยำในการทำนายบุคลิกภาพ
7. ใช้ภาษาที่เป็นธรรมชาติ อ่อนโยน และเข้าใจง่าย
8. หลีกเลี่ยงคำทำนายเชิงโชคชะตาเด็ดขาด ใช้คำว่า "มีแนวโน้ม", "สะท้อนว่า", "บ่งบอกถึงพลังของ..."
9. ตอบเป็นข้อความต่อเนื่องแบบธรรมชาติ ไม่ใช้หัวข้อหรือหมวดหมู่
10. ห้ามใช้ emoji หรือสัญลักษณ์พิเศษใดๆ
11. **สำหรับคำถามต่อเนื่อง:** อย่าเปลี่ยนราศีหรือข้อมูลวันเกิด ให้ใช้ข้อมูลเดิมที่ผู้ใช้ให้มา

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

กรุณาตอบคำถามตามแนวทางที่กำหนดไว้ โดยใช้ความรู้โหราศาสตร์ตะวันตกและให้คำแนะนำที่เป็นประโยชน์"""
        
        # สร้าง system prompt ที่เหมาะสม
        if astrology_chart:
            system_prompt = f"""คุณเป็นแชทบอทโหราศาสตร์ตะวันตกที่เชี่ยวชาญในการทำนายดวงชะตาจากวันเดือนปีเกิด ตอบคำถามด้วยภาษาที่เป็นมิตร เป็นธรรมชาติ และเข้าใจง่าย เริ่มต้นด้วยการระบุวันเกิดและราศีอาทิตย์อย่างชัดเจน แล้วอธิบายลักษณะนิสัยและให้คำแนะนำในด้านต่างๆ (การงาน, การเงิน, ความรัก) ตามรูปแบบที่กำหนดไว้ ห้ามใช้ emoji หรือสัญลักษณ์พิเศษใดๆ ในคำตอบ ให้ตอบเป็นข้อความต่อเนื่องแบบธรรมชาติ ไม่ใช้รูปแบบหัวข้อหรือหมวดหมู่ **ใช้ชื่อราศีแบบไทยเท่านั้น: เมษ, พฤษภ, เมถุน, กรกฎ, สิงห์, กันย์, ตุล, พิจิก, ธนู, มังกร, กุมภ์, มีน ห้ามใช้ชื่อสัตว์ เช่น ราศีปลา, ราศีแกะ, ราศีวัว สำหรับราศีที่ 12 ต้องใช้ ราศีมีน เท่านั้น ห้ามใช้คำว่า ราศีปลา หรือ Pisces** **ใช้คำว่า 'ลัคณา' แทน 'Ascendant' ในทุกกรณี** **หากมีข้อมูลลัคณา (ราศีประจำลัคนา) ให้ใช้เพื่อเพิ่มความแม่นยำในการทำนายบุคลิกภาพ** **คำลงท้ายต้องใช้ 'ค่ะ' เท่านั้น ห้ามใช้ 'ครับ/ค่ะ' หรือ 'ครับ'**"""
        else:
            system_prompt = """คุณเป็นแชทบอทโหราศาสตร์ตะวันตกที่เชี่ยวชาญในการทำนายดวงชะตาจากวันเดือนปีเกิด ตอบคำถามด้วยภาษาที่เป็นมิตร เป็นธรรมชาติ และเข้าใจง่าย เริ่มต้นด้วยการระบุวันเกิดและราศีอาทิตย์อย่างชัดเจน แล้วอธิบายลักษณะนิสัยและให้คำแนะนำในด้านต่างๆ (การงาน, การเงิน, ความรัก) ตามรูปแบบที่กำหนดไว้ ห้ามใช้ emoji หรือสัญลักษณ์พิเศษใดๆ ในคำตอบ ให้ตอบเป็นข้อความต่อเนื่องแบบธรรมชาติ ไม่ใช้รูปแบบหัวข้อหรือหมวดหมู่ **ใช้ชื่อราศีแบบไทยเท่านั้น: เมษ, พฤษภ, เมถุน, กรกฎ, สิงห์, กันย์, ตุล, พิจิก, ธนู, มังกร, กุมภ์, มีน ห้ามใช้ชื่อสัตว์ เช่น ราศีปลา, ราศีแกะ, ราศีวัว สำหรับราศีที่ 12 ต้องใช้ ราศีมีน เท่านั้น ห้ามใช้คำว่า ราศีปลา หรือ Pisces** **ใช้คำว่า 'ลัคณา' แทน 'Ascendant' ในทุกกรณี** **หากมีข้อมูลลัคณา (ราศีประจำลัคนา) ให้ใช้เพื่อเพิ่มความแม่นยำในการทำนายบุคลิกภาพ** **หากไม่มีข้อมูลวันเกิดหรือราศี ให้แจ้งเตือนผู้ใช้ให้ระบุข้อมูลก่อน เช่น 'ขออภัยค่ะ ระบบไม่พบข้อมูลราศีของคุณ กรุณาระบุวันเกิดก่อน เช่น 09/02/2004 ราศีอะไร'** **คำลงท้ายต้องใช้ 'ค่ะ' เท่านั้น ห้ามใช้ 'ครับ/ค่ะ' หรือ 'ครับ'**"""
        
        # print("กำลังส่งคำถามไปยัง GPT...")
        # ใช้ชื่อโมเดลจาก ENV ถ้าไม่ระบุจะใช้ gpt-4o-mini
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
            temperature=0.8,  # ลดลงเล็กน้อยเพื่อความสมดุลระหว่างความหลากหลายและความสอดคล้อง
            max_tokens=1000   # จำกัดความยาวเพื่อให้คำตอบกระชับ
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

    # แสดงรายงานบนเทอร์มินัลสำหรับ RAGAS
    try:
        print_ragas_terminal_report(
            question=question,
            retrieved_docs=retrieved_docs,
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