import logging
from fastapi import Request
from linebot.v3.messaging import TextMessage
from dotenv import load_dotenv
from datetime import datetime
from pymongo import MongoClient
import os
import re

# ใช้ฟังก์ชัน Retrieval จาก utils
from .retrieval_utils import ask_question_to_rag, store_user_response, store_user_question, check_and_update_question_limit
# ใช้ฟังก์ชัน Birth Date Parser
from .birth_date_parser import extract_birth_date_from_message, generate_birth_chart_prediction
# ใช้ฟังก์ชัน Content Filter
from .content_filter import check_content_safety

load_dotenv()

# ตั้งค่า Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ฟังก์ชัน extract_birth_date_from_message ถูกย้ายไปที่ birth_date_parser.py แล้ว
def get_or_create_user_profile(user_id: str, user_message: str = None):
    """ตรวจสอบ/สร้าง user profile ด้วยวันเกิด"""
    mongo_uri = os.getenv("MONGO_URL")
    logger.info(f"🌐 Checking user {user_id}")

    try:
        client = MongoClient(mongo_uri)
        collection = client["astrobot"]["user_profiles"]

        user = collection.find_one({"user_id": user_id})
        logger.info(f"🔎 User found: {user is not None}")

        if user_message:
            # ใช้ฟังก์ชันแยกวันเกิด (แยกเสมอไม่ว่าจะมีข้อมูลอยู่แล้วหรือไม่)
            birth_date = extract_birth_date_from_message(user_message)
            logger.info(f"Extracted birth_date: {birth_date}")
            
            if birth_date:
                # ตรวจสอบว่าวันเกิดใหม่หรือไม่
                current_birth_date = user.get("birth_date") if user else None
                is_new_birth_date = current_birth_date != birth_date
                
                if is_new_birth_date:
                    logger.info(f"Updating birth_date from {current_birth_date} to {birth_date}")
                else:
                    logger.info(f"Same birth_date: {birth_date}")
                
                profile_data = {
                    "user_id": user_id,
                    "birth_date": birth_date,
                    "updated_at": datetime.utcnow(),
                    "raw_message": user_message
                }
                
                # เพิ่ม created_at เฉพาะเมื่อสร้างใหม่
                if not user:
                    profile_data["created_at"] = datetime.utcnow()

                result = collection.update_one(
                    {"user_id": user_id},
                    {"$set": profile_data},
                    upsert=True
                )
                logger.info(f"Saved profile for {user_id}")
                
                # ตรวจสอบว่ามีคำขอทำนายดวงกำเนิดหรือไม่
                if any(keyword in user_message.lower() for keyword in ['ทำนายดวงกำเนิด', 'ดวงกำเนิด', 'ทำนายดวง', 'ดูดวงกำเนิด', 'ราศีอะไร', 'ราศี', 'ดวงชะตา']):
                    try:
                        logger.info(f"กำลังสร้างคำทำนายดวงกำเนิดสำหรับ: {user_message}")
                        birth_chart_prediction = generate_birth_chart_prediction(user_message, user_id)
                        if birth_chart_prediction and not birth_chart_prediction.startswith("ไม่สามารถ"):
                            logger.info(f"สร้างคำทำนายดวงกำเนิดสำเร็จ (ความยาว: {len(birth_chart_prediction)} ตัวอักษร)")
                            
                            # บันทึกคำถามใน user_profiles
                            store_user_question(
                                question=user_message,
                                user_id=user_id,
                                context_data={"birth_date": birth_date}
                            )
                            
                            # บันทึกคำตอบใน collection astrobot
                            store_user_response(
                                question=user_message,
                                answer=birth_chart_prediction,
                                user_id=user_id,
                                response_type="birth_chart",
                                context_data={"birth_date": birth_date}
                            )
                            
                            return birth_chart_prediction
                        else:
                            logger.warning(f"ไม่สามารถสร้างคำทำนายดวงกำเนิดได้: {birth_chart_prediction}")
                    except Exception as e:
                        logger.warning(f"Error generating birth chart prediction: {e}")
                
                # สร้างข้อมูลดวงชะตาเพื่อแสดงข้อมูล Ascendant (ถ้ามีเวลาเกิด)
                ascendant_info = ""
                try:
                    from .birth_date_parser import BirthDateParser
                    parser = BirthDateParser()
                    birth_info = parser.extract_birth_info(user_message)
                    if birth_info and birth_info.get('time'):
                        chart_info = parser.generate_birth_chart_info(
                            birth_info['date'], 
                            birth_info.get('time'), 
                            birth_info.get('latitude', 13.7563), 
                            birth_info.get('longitude', 100.5018)
                        )
                        
                        if chart_info and 'ascendant' in chart_info:
                            ascendant = chart_info['ascendant']
                            ascendant_info = f"""

🌟 **ข้อมูลลัคณา (Ascendant)**
ราศีลัคณา: {ascendant['sign']} {ascendant['degree']:.1f}°
ธาตุ: {ascendant['element']}
คุณภาพ: {ascendant['quality']}

{chart_info.get('ascendant_interpretation', '')}"""
                            logger.info(f"✅ Generated ascendant info: {ascendant['sign']} {ascendant['degree']:.1f}°")
                except Exception as e:
                    logger.warning(f"Error generating ascendant info: {e}")

                # ตอบคำถามโหราศาสตร์ทันที
                try:
                    logger.info(f"กำลังตอบคำถามโหราศาสตร์สำหรับ: {user_message}")
                    from .retrieval_utils import ask_question_to_rag
                    astrology_answer = ask_question_to_rag(user_message, user_id)
                    logger.info(f"ได้รับคำตอบโหราศาสตร์ (ความยาว: {len(astrology_answer)} ตัวอักษร)")
                    
                    # เพิ่มข้อมูลลัคณาในคำตอบถ้ามี
                    if ascendant_info:
                        astrology_answer += ascendant_info
                        logger.info("✅ Added ascendant info to astrology answer")
                    
                    # บันทึกคำถามใน user_profiles
                    store_user_question(
                        question=user_message,
                        user_id=user_id,
                        context_data={"birth_date": birth_date}
                    )
                    
                    # บันทึกคำตอบใน collection astrobot
                    store_user_response(
                        question=user_message,
                        answer=astrology_answer,
                        user_id=user_id,
                        response_type="astrology_qa",
                        context_data={"birth_date": birth_date}
                    )
                    
                    return astrology_answer
                except Exception as e:
                    logger.warning(f"Could not get astrology answer: {e}")
                    
                    welcome_message = f"""ขอบคุณที่ให้ข้อมูลครับ!
วันเกิดของคุณ: {birth_date}{ascendant_info}

ตอนนี้คุณสามารถถามเรื่องต่างๆ ได้แล้ว เช่น:
ดูดวงตามราศี
ลักษณะนิสัยตามวันเกิด  
ความเข้ากันได้กับคนอื่น
คำแนะนำสำหรับวันนี้
ทำนายดวงกำเนิด

ลองถามอะไรก็ได้นะครับ!"""
                    
                    # บันทึกคำถามใน user_profiles
                    store_user_question(
                        question=user_message,
                        user_id=user_id,
                        context_data={"birth_date": birth_date}
                    )
                    
                    # บันทึกคำตอบใน collection astrobot
                    store_user_response(
                        question=user_message,
                        answer=welcome_message,
                        user_id=user_id,
                        response_type="welcome_message",
                        context_data={"birth_date": birth_date}
                    )
                    
                    return welcome_message
            
            # ถ้าไม่พบวันเกิดในข้อความ แต่ผู้ใช้มี profile อยู่แล้ว ให้ผ่านไปให้ RAG จัดการ
            if user and user.get("birth_date"):
                logger.info(f"User has existing profile with birth_date: {user.get('birth_date')}")
                return None  # ให้ผ่านไปให้ RAG จัดการ
            
            error_message = """ขออภัยครับ ยังไม่สามารถแยกวันเกิดจากข้อความได้

กรุณาระบุวันเกิดในรูปแบบ:
07/09/2003
15/03/1990  
วันที่ 7 เดือน 9 ปี 2003
7 มกราคม 2003

ลองพิมพ์ใหม่นะครับ"""
            
            # บันทึกคำถามใน user_profiles
            store_user_question(
                question=user_message,
                user_id=user_id,
                context_data={"error_type": "birth_date_parse_failed"}
            )
            
            # บันทึกคำตอบใน collection astrobot
            store_user_response(
                question=user_message,
                answer=error_message,
                user_id=user_id,
                response_type="error_message",
                context_data={"error_type": "birth_date_parse_failed"}
            )
            
            return error_message
        
        else:
            welcome_message = """ยินดีต้อนรับสู่โลกแห่งโหราศาสตร์! 

กรุณาบอกวันเกิดของคุณ เช่น:
07/09/2003
วันที่ 7 เดือน 9 ปี 2003
7 มกราคม 2003

พิมพ์วันเกิดของคุณได้เลยครับ"""
            
            # บันทึกคำถามใน user_profiles
            store_user_question(
                question=user_message or "initial_contact",
                user_id=user_id,
                context_data={"user_status": "new_user"}
            )
            
            # บันทึกคำตอบใน collection astrobot
            store_user_response(
                question=user_message or "initial_contact",
                answer=welcome_message,
                user_id=user_id,
                response_type="initial_welcome",
                context_data={"user_status": "new_user"}
            )
            
            return welcome_message
            
    except Exception as e:
        logger.error(f"Database error: {e}")
        error_message = "ขออภัยครับ เกิดปัญหาในระบบ กรุณาลองใหม่อีกครั้ง"
        
        # บันทึกคำถามใน user_profiles
        store_user_question(
            question=user_message or "unknown",
            user_id=user_id,
            context_data={"error_type": "database_error", "error_details": str(e)}
        )
        
        # บันทึกคำตอบใน collection astrobot
        store_user_response(
            question=user_message or "unknown",
            answer=error_message,
            user_id=user_id,
            response_type="system_error",
            context_data={"error_type": "database_error", "error_details": str(e)}
        )
        
        return error_message

def generate_reply_message(event):
    """ตอบกลับข้อความจาก LINE"""
    user_text = event.message.text.strip()
    user_id = event.source.user_id if event.source and hasattr(event.source, 'user_id') else "unknown"
    logger.info(f"📨 Message from {user_id}: {user_text}")

    # ตรวจสอบความปลอดภัยของเนื้อหาก่อน
    is_safe, safety_message = check_content_safety(user_text)
    if not is_safe:
        logger.warning(f"Content filtered for user {user_id}: {safety_message}")
        
        # บันทึกคำถามใน user_profiles
        store_user_question(
            question=user_text,
            user_id=user_id,
            context_data={"filter_reason": "unsafe_content"}
        )
        
        # บันทึกคำตอบใน collection astrobot
        store_user_response(
            question=user_text,
            answer=safety_message,
            user_id=user_id,
            response_type="content_filtered",
            context_data={"filter_reason": "unsafe_content"}
        )
        
        return TextMessage(text=safety_message)

    # ตรวจสอบ/สร้างโปรไฟล์ก่อนใช้งาน
    profile_status = get_or_create_user_profile(user_id=user_id, user_message=user_text)
    if profile_status:
        return TextMessage(text=profile_status)

    # ตรวจสอบจำนวนคำถามต่อเนื่อง (เฉพาะเมื่อไม่ใช่การสร้างโปรไฟล์)
    is_allowed, current_count, limit_message = check_and_update_question_limit(user_id, max_questions=3)
    if not is_allowed:
        logger.info(f"🚫 Question limit exceeded for user {user_id}: {current_count}/3")
        
        # บันทึกคำถามใน user_profiles
        store_user_question(
            question=user_text,
            user_id=user_id,
            context_data={"question_count": current_count, "max_questions": 3}
        )
        
        # บันทึกคำตอบใน collection astrobot
        store_user_response(
            question=user_text,
            answer=limit_message,
            user_id=user_id,
            response_type="question_limit_exceeded",
            context_data={"question_count": current_count, "max_questions": 3}
        )
        
        return TextMessage(text=limit_message)

    # ถ้ามี profile แล้ว ให้ถามตอบได้ผ่าน RAG
    try:
        # ตรวจสอบว่ามีคำขอทำนายดวงกำเนิดหรือไม่
        if any(keyword in user_text.lower() for keyword in ['ทำนายดวงกำเนิด', 'ดวงกำเนิด', 'ทำนายดวง', 'ดูดวงกำเนิด']):
            logger.info(f"กำลังสร้างคำทำนายดวงกำเนิดสำหรับ: {user_text}")
            birth_chart_prediction = generate_birth_chart_prediction(user_text, user_id)
            if birth_chart_prediction and not birth_chart_prediction.startswith("ไม่สามารถ"):
                logger.info(f"สร้างคำทำนายดวงกำเนิดสำเร็จ (ความยาว: {len(birth_chart_prediction)} ตัวอักษร)")
                reply_text = birth_chart_prediction
                
                # บันทึกคำถามใน user_profiles
                store_user_question(
                    question=user_text,
                    user_id=user_id,
                    context_data={"prediction_type": "birth_chart"}
                )
                
                # บันทึกคำตอบใน collection astrobot
                store_user_response(
                    question=user_text,
                    answer=reply_text,
                    user_id=user_id,
                    response_type="birth_chart_prediction",
                    context_data={"prediction_type": "birth_chart"}
                )
            else:
                logger.warning(f"ไม่สามารถสร้างคำทำนายดวงกำเนิดได้: {birth_chart_prediction}")
                reply_text = birth_chart_prediction or "ไม่สามารถสร้างคำทำนายดวงกำเนิดได้ กรุณาระบุวันเกิดที่ชัดเจน"
                
                # บันทึกคำถามใน user_profiles
                store_user_question(
                    question=user_text,
                    user_id=user_id,
                    context_data={"error_type": "prediction_failed"}
                )
                
                # บันทึกคำตอบใน collection astrobot
                store_user_response(
                    question=user_text,
                    answer=reply_text,
                    user_id=user_id,
                    response_type="birth_chart_error",
                    context_data={"error_type": "prediction_failed"}
                )
        else:
            logger.info(f"กำลังประมวลผลคำถาม: {user_text}")
            reply_text = ask_question_to_rag(user_text, user_id=user_id)
            logger.info(f"ได้รับคำตอบ (ความยาว: {len(reply_text)} ตัวอักษร)")
    except Exception as e:
        logger.error(f"Error in processing: {e}")
        reply_text = "ขออภัยครับ เกิดปัญหาในการประมวลผล กรุณาลองใหม่อีกครั้ง"
        
        # บันทึกคำถามใน user_profiles
        store_user_question(
            question=user_text,
            user_id=user_id,
            context_data={"error_type": "processing_error", "error_details": str(e)}
        )
        
        # บันทึกคำตอบใน collection astrobot
        store_user_response(
            question=user_text,
            answer=reply_text,
            user_id=user_id,
            response_type="processing_error",
            context_data={"error_type": "processing_error", "error_details": str(e)}
        )
    
    return TextMessage(text=reply_text)