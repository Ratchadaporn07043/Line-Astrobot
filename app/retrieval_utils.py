import os
import logging
from datetime import datetime
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
    
    Args:
        question (str): คำถามของผู้ใช้ (ใช้สำหรับอัปเดต user_profiles เท่านั้น)
        answer (str): คำตอบที่ส่งให้ผู้ใช้
        user_id (str): ID ของผู้ใช้
        response_type (str): ประเภทของคำตอบ (rag_response, birth_chart, quick_reply, etc.)
        context_data (dict): ข้อมูลบริบทเพิ่มเติม
    """
    try:
        mongo_uri = os.getenv("MONGO_URL")
        if not mongo_uri or mongo_uri == "mongodb+srv://your-username:your-password@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority":
            logger.warning("MONGO_URL not configured properly, skipping response storage")
            return
        
        logger.info(f"🔄 Attempting to store response for user {user_id}, type: {response_type}")
        
        mongo_client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000, connectTimeoutMS=5000)
        responses_collection = mongo_client["astrobot"]["responses"]
        profiles_collection = mongo_client["astrobot"]["user_profiles"]
        
        # สร้างข้อมูลสำหรับบันทึกใน responses (ไม่เก็บคำถาม)
        response_data = {
            "user_id": user_id,
            "answer": answer,
            "response_type": response_type,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
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
    """
    บันทึกคำถามของผู้ใช้ใน user_profiles collection
    
    Args:
        question (str): คำถามของผู้ใช้
        user_id (str): ID ของผู้ใช้
        context_data (dict): ข้อมูลบริบทเพิ่มเติม
    """
    try:
        mongo_uri = os.getenv("MONGO_URL")
        if not mongo_uri or mongo_uri == "mongodb+srv://your-username:your-password@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority":
            logger.warning("MONGO_URL not configured properly, skipping question storage")
            return
        
        logger.info(f"🔄 Attempting to store question for user {user_id}")
        
        mongo_client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000, connectTimeoutMS=5000)
        profiles_collection = mongo_client["astrobot"]["user_profiles"]
        
        # สร้างข้อมูลสำหรับบันทึกคำถาม
        question_data = {
            "user_id": user_id,
            "question": question,
            "question_timestamp": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        # เพิ่มข้อมูลบริบทถ้ามี
        if context_data:
            question_data.update(context_data)
        
        # อัปเดตหรือสร้างโปรไฟล์ใหม่
        result = profiles_collection.update_one(
            {"user_id": user_id},
            {"$set": question_data},
            upsert=True
        )
        
        logger.info(f"✅ Successfully stored question in user_profiles for user {user_id}")
        logger.info(f"📊 Question data: user_id={user_id}, question_length={len(question)}")
        
        mongo_client.close()
        
    except Exception as e:
        logger.error(f"❌ Failed to store question in user_profiles: {e}")
        logger.error(f"📝 Error details - user_id: {user_id}")
        import traceback
        logger.error(f"🔍 Full traceback: {traceback.format_exc()}")

# ✔️ บันทึกหรืออัปเดต user_profiles พร้อมบริบทการสนทนา
def log_user_interaction(
    question: str,
    answer: str,
    embedding: list,
    user_id: str = "unknown",
    context_data: dict = None
):
    # print(f"\n=== บันทึกการโต้ตอบลง MongoDB ===")
    # print(f"User ID: {user_id}")
    # print(f"คำถาม: {question}")
    # print(f"คำตอบ: {answer[:100]}...")
    # print(f"Embedding size: {len(embedding)}")
    
    mongo_uri = os.getenv("MONGO_URL")
    if not mongo_uri or mongo_uri == "mongodb+srv://your-username:your-password@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority":
        # print("MONGO_URL not configured properly. Please set up your .env file with valid MongoDB connection string.")
        return
    mongo_client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000, connectTimeoutMS=5000)
    collection = mongo_client[SUMMARY_DB_NAME]["user_profiles"]

    existing_profile = collection.find_one({"user_id": user_id})
    # print(f"พบโปรไฟล์เดิม: {existing_profile is not None}")

    update_data = {
        "question": question,
        "embedding": embedding,
        "response": answer,
        "updated_at": datetime.utcnow()
    }

    if not existing_profile:
        update_data["created_at"] = datetime.utcnow()
        # print("สร้างโปรไฟล์ใหม่")

    # บันทึกข้อมูลบริบท
    if context_data:
        update_data.update(context_data)
        # print(f"บันทึกข้อมูลบริบท: {context_data}")

    if existing_profile and "birth_date" in existing_profile:
        update_data["birth_date"] = existing_profile["birth_date"]
        # print(f"ข้อมูลวันเกิด: {existing_profile['birth_date']}")

    result = collection.update_one(
        {"user_id": user_id},
        {"$set": update_data},
        upsert=True
    )

    # print(f"บันทึก user_profiles สำหรับ {user_id} แล้ว (Matched: {result.matched_count}, Modified: {result.modified_count})")
    # print(f"=== เสร็จสิ้นการบันทึก ===\n")

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

# ตรวจสอบและอัปเดตจำนวนคำถามต่อเนื่อง
def check_and_update_question_limit(user_id: str, max_questions: int = 3):
    """
    ตรวจสอบและอัปเดตจำนวนคำถามต่อเนื่องของผู้ใช้
    
    Args:
        user_id (str): ID ของผู้ใช้
        max_questions (int): จำนวนคำถามสูงสุดที่อนุญาต (ค่าเริ่มต้น: 3)
        
    Returns:
        tuple: (is_allowed, current_count, message)
    """
    try:
        mongo_uri = os.getenv("MONGO_URL")
        if not mongo_uri or mongo_uri == "mongodb+srv://your-username:your-password@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority":
            logger.warning("MONGO_URL not configured properly, allowing question")
            return True, 0, ""
            
        mongo_client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000, connectTimeoutMS=5000)
        collection = mongo_client[SUMMARY_DB_NAME]["user_profiles"]
        
        user_profile = collection.find_one({"user_id": user_id})
        current_date = datetime.utcnow().date()
        
        if not user_profile:
            # ผู้ใช้ใหม่ - อนุญาตให้ถามได้
            update_data = {
                "user_id": user_id,
                "daily_question_count": 1,
                "last_question_date": current_date,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            collection.update_one(
                {"user_id": user_id},
                {"$set": update_data},
                upsert=True
            )
            logger.info(f"✅ New user {user_id} - Question 1/3 allowed")
            return True, 1, ""
        
        # ตรวจสอบวันที่ของคำถามล่าสุด
        last_question_date = user_profile.get("last_question_date")
        daily_question_count = user_profile.get("daily_question_count", 0)
        
        if last_question_date:
            # แปลงเป็น date object ถ้าเป็น string
            if isinstance(last_question_date, str):
                last_question_date = datetime.fromisoformat(last_question_date).date()
            elif hasattr(last_question_date, 'date'):
                last_question_date = last_question_date.date()
        
        # รีเซ็ตจำนวนคำถามถ้าเป็นวันใหม่
        if not last_question_date or last_question_date < current_date:
            daily_question_count = 0
            logger.info(f"🔄 Reset question count for user {user_id} - new day")
        
        # ตรวจสอบจำนวนคำถาม
        if daily_question_count >= max_questions:
            remaining_hours = 24 - (datetime.utcnow().hour)
            message = f"""ขออภัยครับ คุณได้ถามคำถามครบ {max_questions} คำถามแล้วในวันนี้

คุณสามารถถามคำถามใหม่ได้ในวันพรุ่งนี้ หรือรออีก {remaining_hours} ชั่วโมง

ขอบคุณที่ใช้บริการโหราศาสตร์ครับ! 🌟"""
            
            logger.info(f"🚫 User {user_id} exceeded question limit: {daily_question_count}/{max_questions}")
            return False, daily_question_count, message
        
        # อัปเดตจำนวนคำถาม
        new_count = daily_question_count + 1
        update_data = {
            "daily_question_count": new_count,
            "last_question_date": current_date,
            "updated_at": datetime.utcnow()
        }
        
        collection.update_one(
            {"user_id": user_id},
            {"$set": update_data}
        )
        
        remaining_questions = max_questions - new_count
        logger.info(f"✅ User {user_id} - Question {new_count}/{max_questions} allowed (remaining: {remaining_questions})")
        
        return True, new_count, ""
        
    except Exception as e:
        logger.error(f"❌ Error checking question limit for user {user_id}: {e}")
        # ในกรณีที่เกิดข้อผิดพลาด ให้อนุญาตให้ถามได้
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
def enhance_question_context(question: str, user_context: dict = None) -> str:
    """
    ปรับปรุงคำถามให้ชัดเจนขึ้นสำหรับคำถามต่อเนื่อง โดยใช้ข้อมูลบริบท
    และข้อมูลจาก collection responses รวมถึงข้อมูลการสนทนาก่อนหน้า
    
    Args:
        question (str): คำถามเดิม
        user_context (dict): ข้อมูลบริบทของผู้ใช้
        
    Returns:
        str: คำถามที่ปรับปรุงแล้ว
    """
    question_lower = question.lower()
    
    # ตรวจสอบคำถามต่อเนื่องที่อ้างอิงถึงราศี
    if any(word in question_lower for word in ["ราศีนี้", "ราศีของฉัน", "ราศีของผม", "ราศีของเรา", "คนราศีนี้", "ราศี", "ดวงชะตา", "นิสัย", "ลักษณะ", "ดาวเคราะห์", "บ้าน", "แอสเปค", "โหราศาสตร์", "ดวง", "สีมงคล", "สัย", "เป็นไง", "เป็นอย่างไร", "ยังไง", "อย่างไร", "ความรัก", "อาชีพ", "การงาน", "สุขภาพ", "การเงิน", "เหมาะ", "ดี", "เป็น", "เป็นคน", "คน", "ดวง", "โหราศาสตร์", "ดาวเคราะห์", "บ้าน", "แอสเปค", "ของคน", "ของ", "เป็นยังไง", "เป็นอย่างไร", "ยังไง", "อย่างไร", "เป็นไง"]):
        # print(f"พบคำถามต่อเนื่อง: {question}")
        
        # ใช้ข้อมูลราศีจากบริบทก่อน
        zodiac = None
        if user_context and user_context.get("zodiac_sign"):
            zodiac = user_context["zodiac_sign"]
            # print(f"ใช้ราศีจากบริบท: {zodiac}")
        elif user_context and user_context.get("birth_date"):
            # ถ้าไม่มีราศีในบริบท แต่มีวันเกิด ให้คำนวณใหม่
            try:
                from datetime import datetime
                birth_date = datetime.strptime(user_context["birth_date"], "%d/%m/%Y")
                day, month = birth_date.day, birth_date.month
                zodiac = calculate_zodiac_from_date(day, month)
                # print(f"คำนวณราศีจากวันเกิด: {zodiac}")
            except:
                pass
        
        # ตรวจสอบข้อมูลราศีจาก recent_conversations ถ้าไม่มีใน user_context
        if not zodiac and user_context and user_context.get("recent_conversations"):
            for conv in user_context["recent_conversations"]:
                context_data = conv.get("context_data", {})
                if context_data.get("zodiac_sign"):
                    zodiac = context_data["zodiac_sign"]
                    # print(f"ใช้ราศีจาก recent_conversations: {zodiac}")
                    break
        
        # ตรวจสอบข้อมูลราศีจาก last_conversation ถ้าไม่มีใน user_context
        if not zodiac and user_context and user_context.get("last_conversation"):
            last_conv = user_context["last_conversation"]
            # วิเคราะห์คำตอบก่อนหน้าเพื่อหาข้อมูลราศี
            if last_conv.get("answer"):
                answer_text = last_conv["answer"]
                # ค้นหาข้อมูลราศีในคำตอบก่อนหน้า
                zodiac_keywords = ["ราศีเมษ", "ราศีพฤษภ", "ราศีมิถุน", "ราศีกรกฎ", "ราศีสิงห์", "ราศีกันย์", 
                                 "ราศีตุล", "ราศีพิจิก", "ราศีธนู", "ราศีมังกร", "ราศีกุมภ์", "ราศีมีน",
                                 "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo", 
                                 "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"]
                
                for keyword in zodiac_keywords:
                    if keyword in answer_text:
                        # แปลงเป็นชื่อราศีไทย
                        if keyword in ["ราศีเมษ", "Aries"]:
                            zodiac = "เมษ"
                        elif keyword in ["ราศีพฤษภ", "Taurus"]:
                            zodiac = "พฤษภ"
                        elif keyword in ["ราศีมิถุน", "Gemini"]:
                            zodiac = "มิถุน"
                        elif keyword in ["ราศีกรกฎ", "Cancer"]:
                            zodiac = "กรกฎ"
                        elif keyword in ["ราศีสิงห์", "Leo"]:
                            zodiac = "สิงห์"
                        elif keyword in ["ราศีกันย์", "Virgo"]:
                            zodiac = "กันย์"
                        elif keyword in ["ราศีตุล", "Libra"]:
                            zodiac = "ตุล"
                        elif keyword in ["ราศีพิจิก", "Scorpio"]:
                            zodiac = "พิจิก"
                        elif keyword in ["ราศีธนู", "Sagittarius"]:
                            zodiac = "ธนู"
                        elif keyword in ["ราศีมังกร", "Capricorn"]:
                            zodiac = "มังกร"
                        elif keyword in ["ราศีกุมภ์", "Aquarius"]:
                            zodiac = "กุมภ์"
                        elif keyword in ["ราศีมีน", "Pisces"]:
                            zodiac = "มีน"
                        break
        
        if zodiac:
            # แทนที่คำถามให้ชัดเจน
            enhanced = question.replace("ราศีนี้", f"ราศี{zodiac}").replace("ราศีของฉัน", f"ราศี{zodiac}").replace("ราศีของผม", f"ราศี{zodiac}").replace("ราศีของเรา", f"ราศี{zodiac}").replace("คนราศีนี้", f"คนราศี{zodiac}")
            
            # เพิ่มการจัดการคำถามทั่วไป
            if "คนราศีนี้" in question_lower and zodiac not in question:
                enhanced = f"คนราศี{zodiac} {question}"
            elif "ราศี" in question_lower and zodiac not in question:
                enhanced = f"ราศี{zodiac} {question}"
            elif "ดวงชะตา" in question_lower and zodiac not in question:
                enhanced = f"ดวงชะตาราศี{zodiac} {question}"
            elif "นิสัย" in question_lower and zodiac not in question:
                enhanced = f"ลักษณะนิสัยราศี{zodiac} {question}"
            elif "ลักษณะ" in question_lower and zodiac not in question:
                enhanced = f"ลักษณะนิสัยราศี{zodiac} {question}"
            elif "สัย" in question_lower and zodiac not in question:
                enhanced = f"ลักษณะนิสัยราศี{zodiac} {question}"
            elif "ดาวเคราะห์" in question_lower and zodiac not in question:
                enhanced = f"ดาวเคราะห์ราศี{zodiac} {question}"
            elif "บ้าน" in question_lower and zodiac not in question:
                enhanced = f"บ้านราศี{zodiac} {question}"
            elif "แอสเปค" in question_lower and zodiac not in question:
                enhanced = f"แอสเปคราศี{zodiac} {question}"
            elif "โหราศาสตร์" in question_lower and zodiac not in question:
                enhanced = f"โหราศาสตร์ราศี{zodiac} {question}"
            elif "ดวง" in question_lower and zodiac not in question:
                enhanced = f"ดวงราศี{zodiac} {question}"
            elif "สีมงคล" in question_lower and zodiac not in question:
                enhanced = f"สีมงคลราศี{zodiac} {question}"
            elif "ความรัก" in question_lower and zodiac not in question:
                enhanced = f"ความรักราศี{zodiac} {question}"
            elif "อาชีพ" in question_lower and zodiac not in question:
                enhanced = f"อาชีพราศี{zodiac} {question}"
            elif "การงาน" in question_lower and zodiac not in question:
                enhanced = f"การงานราศี{zodiac} {question}"
            elif "สุขภาพ" in question_lower and zodiac not in question:
                enhanced = f"สุขภาพราศี{zodiac} {question}"
            elif "การเงิน" in question_lower and zodiac not in question:
                enhanced = f"การเงินราศี{zodiac} {question}"
            elif "เหมาะ" in question_lower and zodiac not in question:
                enhanced = f"ราศี{zodiac} {question}"
            elif "ดี" in question_lower and zodiac not in question:
                enhanced = f"ราศี{zodiac} {question}"
            elif "เป็น" in question_lower and zodiac not in question:
                enhanced = f"ราศี{zodiac} {question}"
            elif "เป็นคน" in question_lower and zodiac not in question:
                enhanced = f"ราศี{zodiac} {question}"
            elif "คน" in question_lower and zodiac not in question and "ราศี" in question_lower:
                enhanced = f"ราศี{zodiac} {question}"
            elif "ของคน" in question_lower and zodiac not in question:
                enhanced = f"ราศี{zodiac} {question}"
            elif "ของ" in question_lower and zodiac not in question and "ราศี" in question_lower:
                enhanced = f"ราศี{zodiac} {question}"
            
            # print(f"ใช้ราศี: {zodiac}")
            # print(f"แปลงคำถาม: {question} → {enhanced}")
            return enhanced
        else:
            # print(f"ไม่มีข้อมูลราศีสำหรับคำนวณ")
            pass
    
    return question

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
def generate_follow_up_questions(context_data: dict = None) -> list:
    """
    สร้างคำถามต่อเนื่องอัตโนมัติตามบริบทของผู้ใช้
    
    Args:
        context_data (dict): ข้อมูลบริบทของผู้ใช้หรือข้อมูลดวงชะตา
        
    Returns:
        list: รายการคำถามต่อเนื่อง
    """
    if not context_data or not context_data.get("zodiac_sign"):
        return []
    
    zodiac = context_data.get("zodiac_sign")
    follow_up_questions = [
        f"นิสัยคนราศี{zodiac}เป็นไง",
        f"อยากทราบเรื่องความรักของราศี{zodiac}",
        f"สีมงคลของราศี{zodiac}มีอะไรบ้าง",
        f"อาชีพที่เหมาะกับราศี{zodiac}",
        f"จุดแข็งและจุดอ่อนของราศี{zodiac}",
        f"วิธีดูแลสุขภาพสำหรับราศี{zodiac}",
        f"ความสัมพันธ์กับราศีอื่นๆ ของราศี{zodiac}",
        f"การเงินและการลงทุนสำหรับราศี{zodiac}"
    ]
    
    return follow_up_questions

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

# ✔️ ทำ Retrieval Phase และถาม GPT พร้อม Western Astrology Context
def ask_question_to_rag(question: str, user_id: str = "unknown") -> str:
    # print(f"\n=== เริ่มการค้นหาข้อมูลสำหรับคำถาม: {question} ===")
    
    # ตรวจสอบจำนวนคำถามต่อเนื่องก่อน
    is_allowed, current_count, limit_message = check_and_update_question_limit(user_id, max_questions=3)
    if not is_allowed:
        logger.info(f"🚫 Question limit exceeded for user {user_id}: {current_count}/3")
        return limit_message
    
    # ตรวจสอบว่าเป็นคำถามต่อเนื่องหรือไม่
    is_follow_up_question = any(word in question.lower() for word in [
        "ราศีนี้", "ราศีของฉัน", "ราศีของผม", "ราศีของเรา", "คนราศีนี้", 
        "นิสัย", "ลักษณะ", "สัย", "เป็นไง", "เป็นอย่างไร", "ยังไง", "อย่างไร",
        "ความรัก", "อาชีพ", "การงาน", "สุขภาพ", "การเงิน", "สีมงคล", "เหมาะ", "ดี", "เป็น",
        "เป็นคน", "คน", "ดวง", "โหราศาสตร์", "ดาวเคราะห์", "บ้าน", "แอสเปค",
        "ของคน", "ของ", "เป็นยังไง", "เป็นอย่างไร", "ยังไง", "อย่างไร", "เป็นไง"
    ])
    
    # ดึงข้อมูลบริบทการสนทนาของผู้ใช้ก่อน
    user_context = get_user_context(user_id)
    user_birth_date = user_context.get("birth_date") if user_context else None
    user_zodiac = user_context.get("zodiac_sign") if user_context else None
    
    # ตรวจสอบว่ามีข้อมูลวันเกิดและเวลาเกิดในคำถามหรือไม่ (เฉพาะเมื่อไม่ใช่คำถามต่อเนื่อง)
    birth_info_from_question = None
    if not is_follow_up_question:
        birth_info_from_question = extract_birth_info_from_message(question)
    astrology_chart = None
    
    # Debug: แสดงข้อมูลการตัดสินใจ (ปิดการแสดงผล)
    # print(f"DEBUG - คำถาม: {question}")
    # print(f"DEBUG - is_follow_up_question: {is_follow_up_question}")
    # print(f"DEBUG - user_context: {user_context is not None}")
    # print(f"DEBUG - user_zodiac: {user_zodiac}")
    # print(f"DEBUG - birth_info_from_question: {birth_info_from_question}")
    
    # สร้างข้อมูลดวงชะตาเฉพาะเมื่อ:
    # 1. มีข้อมูลวันเกิดในคำถาม และ
    # 2. ไม่ใช่คำถามต่อเนื่อง
    should_create_chart = (birth_info_from_question and birth_info_from_question['date'] and 
                          not is_follow_up_question)
    
    if should_create_chart:
        # print(f"พบข้อมูลวันเกิดในคำถามใหม่: {birth_info_from_question['date']}")
        if birth_info_from_question['time']:
            # print(f"พบเวลาเกิดในคำถาม: {birth_info_from_question['time']}")
            pass
        
        # สร้างข้อมูลดวงชะตารายละเอียด
        astrology_chart = generate_detailed_astrology_reading(question)
        if astrology_chart:
            # print(f"สร้างดวงชะตาสำเร็จ: ราศี{astrology_chart['zodiac_sign']} ({astrology_chart['zodiac_element']})")
            pass
    elif user_context and user_zodiac and is_follow_up_question:
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
    elif birth_info_from_question and birth_info_from_question['date'] and not user_context and not is_follow_up_question:
        # สำหรับคำถามใหม่ที่ไม่มีบริบท ให้สร้างข้อมูลดวงชะตา
        # print(f"สร้างข้อมูลดวงชะตาใหม่: {birth_info_from_question['date']}")
        astrology_chart = generate_detailed_astrology_reading(question)
    
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
    
    # ปรับปรุงคำถามให้ชัดเจนขึ้นสำหรับคำถามต่อเนื่อง
    enhanced_question = enhance_question_context(question, user_context)
    if enhanced_question != question:
        # print(f"ปรับปรุงคำถาม: {enhanced_question}")
        question = enhanced_question
    
    # การจัดการคำถามต่อเนื่องถูกจัดการใน enhance_question_context แล้ว
    
    # ลองค้นหาจาก MongoDB แบบ Manual Search
    retrieved_docs = []
    try:
        # print("กำลังค้นหาจาก MongoDB แบบ Manual Search...")
        
        # โหลด embedding model
        import numpy as np
        
        model = SentenceTransformer("all-MiniLM-L6-v2")
        query_embedding = model.encode(question)
        # print(f"สร้าง query embedding สำเร็จ (ขนาด: {len(query_embedding)})")
        
        # ค้นหาจาก collections ทั้งหมดที่มีข้อมูลใน SUMMARY_DB_NAME (สำหรับ embeddings)
        collections_to_search = ["text_chunks", "image_chunks", "table_chunks"]
        
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
                    # คำนวณ similarity scores
                    similarities = []
                    for doc in docs:
                        if 'embeddings' in doc:
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
                        if similarity > 0.2:  # ลด threshold จาก 0.3 เป็น 0.2
                            # print(f"\nเอกสารที่ {i+1} จาก {collection_name} (Similarity: {similarity:.4f}):")
                            # print(f"   เนื้อหา: {doc['text'][:200]}...")
                            
                            # เพิ่มข้อมูล source
                            source_info = f"[{collection_name}]"
                            if 'page' in doc:
                                source_info += f" หน้า {doc['page']}"
                            if 'chunk_id' in doc:
                                source_info += f" Chunk {doc['chunk_id']}"
                            if 'type' in doc:
                                source_info += f" ({doc['type']})"
                            
                            # print(f"   แหล่งที่มา: {source_info}")
                            
                            # ใช้ข้อมูลจาก summary database เท่านั้น
                            summary_content = get_summary_content(doc.get('_id'), collection_name)
                            
                            retrieved_docs.append({
                                'text': doc['text'],
                                'summary': doc.get('summary', ''),
                                'summary_content': summary_content,
                                'source': source_info,
                                'similarity': similarity,
                                'collection': collection_name,
                                'doc_id': doc.get('_id')
                            })
                        else:
                            # print(f"เอกสารที่ {i+1} มี similarity ต่ำเกินไป: {similarity:.4f}")
                            pass
                
                client.close()
                    
            except Exception as e:
                # print(f"ไม่สามารถค้นหาใน {collection_name} ได้: {e}")
                continue
                
    except Exception as e:
        # print(f"ไม่สามารถค้นหาจาก MongoDB ได้: {e}")
        # print("ใช้ GPT โดยตรงแทน")
        pass
    
    # แสดงสรุปผลการค้นหา
    # print(f"\n=== สรุปผลการค้นหา ===")
    # print(f"เอกสารที่พบทั้งหมด: {len(retrieved_docs)} เอกสาร")
    # if retrieved_docs:
    #     print(f"พบข้อมูลที่เกี่ยวข้อง สามารถใช้ RAG ได้")
    # else:
    #     print(f"ไม่พบข้อมูลที่เกี่ยวข้อง ใช้ความรู้ทั่วไป")
    # print(f"=== เสร็จสิ้นการค้นหา ===\n")
    
    # ใช้ direct GPT เพราะ vector store ไม่มีข้อมูล
    query_vector = []

    # ใช้ GPT โดยตรง (ไม่ใช้ RAG เพราะ vector store ไม่มีข้อมูล)
    try:
        from openai import OpenAI
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key or openai_key == "your-openai-api-key-here":
            # print("OPENAI_API_KEY not configured properly. Please set up your .env file with valid OpenAI API key.")
            return "ขออภัยค่ะ ระบบยังไม่ได้ตั้งค่า API key สำหรับ OpenAI กรุณาติดต่อผู้ดูแลระบบ"
        client = OpenAI(api_key=openai_key)
        
        # สร้าง context จากเอกสารที่ค้นหาได้จาก astrobot_summary เท่านั้น
        context_info = ""
        if retrieved_docs:
            context_info = "\n\nข้อมูลที่เกี่ยวข้องจากฐานข้อมูล astrobot_summary:\n"
            for i, doc in enumerate(retrieved_docs):
                if isinstance(doc, dict):
                    # ใช้ summary ถ้ามี หรือใช้ text ถ้าไม่มี summary
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
                location_info = f"สถานที่เกิด: {astrology_chart['birth_location']['location']}\n"
            
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
- ราศี{astrology_chart['zodiac_sign']} มีคุณภาพ Mutable

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

            # เพิ่มข้อมูลรายละเอียดการงาน การเงิน ความรัก
            if 'detailed_reading' in astrology_chart:
                detailed = astrology_chart['detailed_reading']
                chart_info += f"""

**การทำนายรายละเอียดสำหรับราศี{astrology_chart['zodiac_sign']}:**

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

                chart_info += f"""

ด้านสุขภาพ:
{detailed.get('สุขภาพ', 'ไม่มีข้อมูลสุขภาพ')}

สีมงคล:
{detailed.get('สีมงคล', 'ไม่มีข้อมูลสีมงคล')}
"""

            # เพิ่มข้อมูลสีมงคล
            if 'lucky_colors' in astrology_chart and astrology_chart['lucky_colors']:
                chart_info += f"""

สีมงคล:
สีดี: {', '.join(astrology_chart['lucky_colors'])}
สีไม่ดี: {', '.join(astrology_chart['bad_colors']) if astrology_chart['bad_colors'] else 'ไม่มีข้อมูล'}
"""

            # เพิ่มข้อมูลโชคลาภ
            if 'omen_info' in astrology_chart:
                omen = astrology_chart['omen_info']
                chart_info += f"""

**โชคลาภและวาสนา:**
ชื่อ: {omen.get('ชื่อ', 'ไม่มีข้อมูล')}
ความหมาย: {omen.get('ความหมาย', 'ไม่มีข้อมูล')}
"""


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
**คำสั่งสำคัญ: ตอบครบทุกด้านตามปกติ**
- ให้ข้อมูลครบถ้วนในด้านต่างๆ (ลักษณะนิสัย, ความรัก, การงาน, การเงิน, สุขภาพ, สีมงคล)
- เริ่มต้นด้วยการระบุวันเกิดและราศีอาทิตย์อย่างชัดเจน
"""

        astrology_prompt = f"""คุณเป็นโหราจารย์ดิจิทัลผู้เชี่ยวชาญด้านโหราศาสตร์ตะวันตก (Western Astrology) ที่มีความรู้ลึกซึ้งเกี่ยวกับดาวเคราะห์ ราศี และการตีความดวงกำเนิด

**บทบาทและความเชี่ยวชาญ:**
- คุณมีความเข้าใจในพลังของราศีอาทิตย์ (Sun Sign) และลัคณา (ราศีประจำลัคนา)
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
1. **สำหรับคำถามใหม่:** เริ่มต้นด้วยการระบุวันเกิดและราศีอาทิตย์อย่างชัดเจน
2. **สำหรับคำถามต่อเนื่อง:** ใช้ข้อมูลราศีที่มีอยู่แล้วและตอบคำถามเฉพาะเจาะจง
3. อธิบายลักษณะนิสัยตามราศีและธาตุ โดยอ้างอิงจากข้อมูลในฐานความรู้
4. **หากมีข้อมูล Ascendant:** ใช้ข้อมูล Ascendant เพื่อเพิ่มความแม่นยำในการทำนายบุคลิกภาพ
5. ใช้ภาษาที่เป็นธรรมชาติ อ่อนโยน และเข้าใจง่าย
6. หลีกเลี่ยงคำทำนายเชิงโชคชะตาเด็ดขาด ใช้คำว่า "มีแนวโน้ม", "สะท้อนว่า", "บ่งบอกถึงพลังของ..."
7. ตอบเป็นข้อความต่อเนื่องแบบธรรมชาติ ไม่ใช้หัวข้อหรือหมวดหมู่
8. ห้ามใช้ emoji หรือสัญลักษณ์พิเศษใดๆ
9. **สำหรับคำถามต่อเนื่อง:** อย่าเปลี่ยนราศีหรือข้อมูลวันเกิด ให้ใช้ข้อมูลเดิมที่ผู้ใช้ให้มา

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

**ข้อห้ามสำคัญ:**
- ห้ามแจ้งว่าขาดข้อมูลหรือไม่สามารถให้คำแนะนำเฉพาะได้
- ห้ามใช้ข้อความเช่น "ไม่มีข้อมูลเพิ่มเติม", "ไม่สามารถให้คำแนะนำเฉพาะได้", "ข้อมูลไม่เพียงพอ"
- ให้ใช้ข้อมูลที่มีอยู่และให้คำแนะนำที่เป็นประโยชน์เสมอ
- หากข้อมูลบางส่วนไม่ครบ ให้ใช้ความรู้โหราศาสตร์ทั่วไปในการให้คำแนะนำ

กรุณาตอบคำถามตามแนวทางที่กำหนดไว้ โดยใช้ความรู้โหราศาสตร์ตะวันตกและให้คำแนะนำที่เป็นประโยชน์
"""
        
        # print("กำลังส่งคำถามไปยัง GPT...")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system", 
                    "content": "คุณเป็นแชทบอทโหราศาสตร์ตะวันตกที่เชี่ยวชาญในการทำนายดวงชะตาจากวันเดือนปีเกิด ตอบคำถามด้วยภาษาที่เป็นมิตร เป็นธรรมชาติ และเข้าใจง่าย เริ่มต้นด้วยการระบุวันเกิดและราศีอาทิตย์อย่างชัดเจน แล้วอธิบายลักษณะนิสัยและให้คำแนะนำในด้านต่างๆ (การงาน, การเงิน, ความรัก, สุขภาพ, สีมงคล) ตามรูปแบบที่กำหนดไว้ ห้ามใช้ emoji หรือสัญลักษณ์พิเศษใดๆ ในคำตอบ ให้ตอบเป็นข้อความต่อเนื่องแบบธรรมชาติ ไม่ใช้รูปแบบหัวข้อหรือหมวดหมู่ **ใช้ชื่อราศีแบบไทยเท่านั้น: เมษ, พฤษภ, เมถุน, กรกฎ, สิงห์, กันย์, ตุล, พิจิก, ธนู, มังกร, กุมภ์, มีน ห้ามใช้ชื่อสัตว์ เช่น ราศีปลา, ราศีแกะ, ราศีวัว สำหรับราศีที่ 12 ต้องใช้ ราศีมีน เท่านั้น ห้ามใช้คำว่า ราศีปลา หรือ Pisces** **ใช้คำว่า 'ลัคณา' แทน 'Ascendant' ในทุกกรณี** **หากมีข้อมูลลัคณา (ราศีประจำลัคนา) ให้ใช้เพื่อเพิ่มความแม่นยำในการทำนายบุคลิกภาพ** **ห้ามแจ้งว่าขาดข้อมูลหรือไม่สามารถให้คำแนะนำเฉพาะได้ ให้ใช้ข้อมูลที่มีอยู่และให้คำแนะนำที่เป็นประโยชน์เสมอ** **คำลงท้ายต้องใช้ 'ค่ะ' เท่านั้น ห้ามใช้ 'ครับ/ค่ะ' หรือ 'ครับ'**"
                },
                {"role": "user", "content": astrology_prompt}
            ],
            temperature=0.8,  # ลดลงเล็กน้อยเพื่อความสมดุลระหว่างความหลากหลายและความสอดคล้อง
            max_tokens=1000   # จำกัดความยาวเพื่อให้คำตอบกระชับ
        )
        answer = response.choices[0].message.content
        # print(f"ได้รับคำตอบจาก GPT (ความยาว: {len(answer)} ตัวอักษร)")
        
        # เพิ่มข้อมูลจำนวนคำถามที่เหลือ
        remaining_questions = 3 - current_count
        if remaining_questions > 0:
            answer += f"\n\n💫 คุณสามารถถามคำถามได้อีก {remaining_questions} คำถามในวันนี้"
        elif remaining_questions == 0:
            answer += f"\n\n⚠️ นี่เป็นคำถามสุดท้ายของคุณในวันนี้"
        
        # ไม่ใช้ฟังก์ชันจัดรูปแบบเพื่อให้ GPT สร้างคำตอบแบบธรรมชาติ
        
        # แสดงสรุปแหล่งที่มาของข้อมูล
        if retrieved_docs:
            print(f"=== สรุปแหล่งที่มาของข้อมูล ===")
            for i, doc in enumerate(retrieved_docs):
                if isinstance(doc, dict):
                    print(f"เอกสารที่ {i+1}: {doc['source']} (Similarity: {doc['similarity']:.4f})")
                else:
                    print(f"เอกสารที่ {i+1}: ข้อมูลทั่วไป")
            print(f"=== เสร็จสิ้นการสรุปแหล่งที่มา ===")
        
        # ไม่เพิ่ม emoji ใดๆ เพื่อให้คำตอบสะอาดตา
        
        # สร้างคำถามต่อเนื่องอัตโนมัติสำหรับคำถามใหม่
        if should_create_chart and astrology_chart:
            # ถ้าเป็นคำถามใหม่ที่มีข้อมูลวันเกิด ให้สร้างคำถามต่อเนื่อง
            follow_up_questions = generate_follow_up_questions(astrology_chart)
            if follow_up_questions:
                # เพิ่มคำถามต่อเนื่องที่ท้ายคำตอบ
                answer += f"\n\nหากต้องการทราบข้อมูลเพิ่มเติม สามารถถามได้ เช่น:\n"
                for i, q in enumerate(follow_up_questions[:3]):  # แสดงแค่ 3 คำถาม
                    answer += f"• {q}\n"
            
    except Exception as gpt_error:
        # print(f"DEBUG - Error in GPT processing: {gpt_error}")
        # print(f"DEBUG - Error type: {type(gpt_error)}")
        # import traceback
        # print(f"DEBUG - Traceback: {traceback.format_exc()}")
        answer = "ขออภัยค่ะ เกิดปัญหาในการประมวลผล กรุณาลองใหม่อีกครั้ง"

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