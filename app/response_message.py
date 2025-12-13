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
from .birth_date_parser import extract_birth_date_from_message, generate_birth_chart_prediction, detect_zodiac_sign_in_message
# ใช้ฟังก์ชัน Content Filter
from .content_filter import check_content_safety
from .multimodel_rag import ORIGINAL_DB_NAME

# Database สำหรับเก็บข้อมูลผู้ใช้ (user_profiles และ responses)
USER_DB_NAME = "astrobot"

load_dotenv()

# ตั้งค่า Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# แสดงคำตอบในเทอร์มินัลแบบอ่านง่าย
def log_pretty_answer(user_id: str, title: str, answer_text: str):
    try:
        header = "\n\n🟦================ คำตอบที่ส่งให้ผู้ใช้ ================\n"
        meta = f"ผู้ใช้: {user_id}\nประเภท: {title}\nความยาว: {len(answer_text)} ตัวอักษร\n"
        body_header = "────────────────────────────────────────────────────\n"
        footer = "\n🟦====================================================\n"
        logging.info(header + meta + body_header + (answer_text or "") + footer)
    except Exception:
        pass

# ฟังก์ชัน extract_birth_date_from_message ถูกย้ายไปที่ birth_date_parser.py แล้ว
def get_or_create_user_profile(user_id: str, user_message: str = None):
    """ตรวจสอบ/สร้าง user profile ด้วยวันเกิด"""
    mongo_uri = os.getenv("MONGO_URL")
    logger.info(f"🌐 Checking user {user_id}")

    try:
        client = MongoClient(mongo_uri)
        collection = client[USER_DB_NAME]["user_profiles"]

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
                            
                            # บันทึกคำถามใน user_profiles (เก็บบริบทเท่านั้น ไม่บันทึก response ต้นทาง)
                            store_user_question(
                                question=user_message,
                                user_id=user_id,
                                context_data={"birth_date": birth_date}
                            )
                            
                            # log_pretty_answer(user_id, "birth_chart", birth_chart_prediction)
                            return birth_chart_prediction
                        else:
                            logger.warning(f"ไม่สามารถสร้างคำทำนายดวงกำเนิดได้: {birth_chart_prediction}")
                    except Exception as e:
                        logger.warning(f"เกิดข้อผิดพลาดในการสร้างคำทำนายดวงกำเนิด: {e}")
                
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
                    logger.warning(f"เกิดข้อผิดพลาดในการสร้างข้อมูล Ascendant: {e}")

                # ตอบคำถามโหราศาสตร์ทันที
                try:
                    logger.info(f"กำลังตอบคำถามโหราศาสตร์สำหรับ: {user_message}")
                    from .retrieval_utils import ask_question_to_rag
                    from .birth_date_parser import BirthDateParser
                    
                    # สร้าง chart_info เพื่อส่งไปยัง ask_question_to_rag
                    parser = BirthDateParser()
                    birth_info_extracted = parser.extract_birth_info(user_message)
                    chart_info_for_rag = None
                    
                    if birth_info_extracted and birth_info_extracted.get('date'):
                        chart_info_for_rag = parser.generate_birth_chart_info(
                            birth_info_extracted['date'],
                            birth_info_extracted.get('time'),
                            birth_info_extracted.get('latitude', 13.7563),
                            birth_info_extracted.get('longitude', 100.5018)
                        )
                        if chart_info_for_rag:
                            logger.info(f"สร้าง chart_info สำหรับ RAG สำเร็จ: ราศี{chart_info_for_rag['zodiac_sign']}")
                    
                    # ส่ง chart_info ไปกับคำถามถ้ามี
                    if chart_info_for_rag:
                        astrology_answer = ask_question_to_rag(user_message, user_id, provided_chart_info=chart_info_for_rag)
                    else:
                        astrology_answer = ask_question_to_rag(user_message, user_id)
                    
                    logger.info(f"ได้รับคำตอบโหราศาสตร์ (ความยาว: {len(astrology_answer)} ตัวอักษร)")
                    
                    # เพิ่มข้อมูลลัคณาในคำตอบถ้ามี และไม่ใช่ข้อความแจ้งเตือน
                    is_error_message = (
                        astrology_answer.startswith("ขออภัยค่ะ ระบบไม่พบข้อมูล") or
                        astrology_answer.startswith("ขออภัยค่ะ ระบบไม่พบข้อมูลบริบท") or
                        astrology_answer.startswith("ขออภัยค่ะ ระบบไม่พบข้อมูลราศี") or
                        astrology_answer.startswith("ขออภัยครับ")  # คำสั่งจำกัดคำถาม
                    )
                    
                    if ascendant_info and not is_error_message:
                        astrology_answer += ascendant_info
                        logger.info("✅ Added ascendant info to astrology answer")
                    elif ascendant_info and is_error_message:
                        logger.info("⚠️ ข้ามการเพิ่มข้อมูล Ascendant เนื่องจากเป็นข้อความข้อผิดพลาด")
                    
                    # บันทึกเฉพาะคำตอบสุดท้ายเท่านั้น (astrology_answer) จะถูกบันทึกโดยชั้นล่างใน ask_question_to_rag
                    # อัปเดตบริบทคำถามล่าสุดไว้ในโปรไฟล์
                    store_user_question(
                        question=user_message,
                        user_id=user_id,
                        context_data={"birth_date": birth_date}
                    )
                    
                    # log_pretty_answer(user_id, "astrology_qa", astrology_answer)
                    return astrology_answer
                except Exception as e:
                    logger.warning(f"ไม่สามารถรับคำตอบโหราศาสตร์ได้: {e}")
                    
                    welcome_message = f"""ขอบคุณที่ให้ข้อมูลครับ!
วันเกิดของคุณ: {birth_date}{ascendant_info}

ตอนนี้คุณสามารถถามเรื่องต่างๆ ได้แล้ว เช่น:
ดูดวงตามราศี
ลักษณะนิสัยตามวันเกิด  
ความเข้ากันได้กับคนอื่น
คำแนะนำสำหรับวันนี้
ทำนายดวงกำเนิด

ลองถามอะไรก็ได้นะครับ!"""
                    
                    # ข้อความต้อนรับนี้เป็นคำตอบสุดท้าย กรณีนี้ไม่มีชั้นล่างบันทึก จึงยังคงบันทึกได้
                    store_user_question(
                        question=user_message,
                        user_id=user_id,
                        context_data={"birth_date": birth_date}
                    )
                    store_user_response(
                        question=user_message,
                        answer=welcome_message,
                        user_id=user_id,
                        response_type="welcome_message",
                        context_data={"birth_date": birth_date}
                    )
                    
                    # log_pretty_answer(user_id, "welcome_message", welcome_message)
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
        logger.error(f"ข้อผิดพลาดฐานข้อมูล: {e}")
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
        logger.warning(f"กรองเนื้อหาสำหรับผู้ใช้ {user_id}: {safety_message}")
        
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
        
        # log_pretty_answer(user_id, "content_filtered", safety_message)
        return TextMessage(text=safety_message)

    # ตรวจสอบ/สร้างโปรไฟล์ก่อนใช้งาน
    profile_status = get_or_create_user_profile(user_id=user_id, user_message=user_text)
    if profile_status:
        return TextMessage(text=profile_status)

    # ตรวจสอบว่ามีชื่อราศีในข้อความหรือไม่ (ก่อนตรวจสอบวันเกิด)
    # ถ้ามีชื่อราศีชัดเจนและไม่มีวันเกิด ให้ตอบคำถามเกี่ยวกับราศีนั้นโดยตรง
    detected_zodiac = detect_zodiac_sign_in_message(user_text)
    birth_date = extract_birth_date_from_message(user_text)
    
    # ถ้าพบชื่อราศีแต่ไม่พบวันเกิด ให้สร้าง chart_info จากชื่อราศี
    chart_info_for_zodiac = None
    if detected_zodiac and not birth_date:
        logger.info(f"พบชื่อราศีในข้อความ: {detected_zodiac['sign']} แต่ไม่พบวันเกิด - จะตอบคำถามเกี่ยวกับราศีโดยตรง")
        from .birth_date_parser import BirthDateParser
        # สร้าง chart_info จากชื่อราศีที่พบ
        chart_info_for_zodiac = {
            'zodiac_sign': detected_zodiac['sign'],
            'zodiac_element': detected_zodiac['element'],
            'zodiac_quality': detected_zodiac['quality'],
            'zodiac_english': detected_zodiac['english_name']
        }

    # ตรวจสอบจำนวนคำถามต่อเนื่อง (ไม่จำกัดจำนวนครั้ง)
    is_allowed, current_count, limit_message = check_and_update_question_limit(user_id)
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
        
        # log_pretty_answer(user_id, "question_limit_exceeded", limit_message)
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
                
                # เก็บบริบทคำถามไว้ แต่ไม่บันทึก response ต้นทาง เพื่อให้เก็บเฉพาะคำตอบสุดท้าย
                store_user_question(
                    question=user_text,
                    user_id=user_id,
                    context_data={"prediction_type": "birth_chart"}
                )
            else:
                logger.warning(f"ไม่สามารถสร้างคำทำนายดวงกำเนิดได้: {birth_chart_prediction}")
                reply_text = birth_chart_prediction or "ไม่สามารถสร้างคำทำนายดวงกำเนิดได้ กรุณาระบุวันเกิดที่ชัดเจน"
                
                # กรณีล้มเหลว ข้อความนี้เป็นคำตอบสุดท้าย จึงยังคงบันทึกได้
                store_user_question(
                    question=user_text,
                    user_id=user_id,
                    context_data={"error_type": "prediction_failed"}
                )
                store_user_response(
                    question=user_text,
                    answer=reply_text,
                    user_id=user_id,
                    response_type="birth_chart_error",
                    context_data={"error_type": "prediction_failed"}
                )
        else:
            logger.info(f"กำลังประมวลผลคำถาม: {user_text}")
            # ถ้ามี chart_info จากชื่อราศี ให้ส่งไปกับคำถาม
            if chart_info_for_zodiac:
                logger.info(f"ส่ง chart_info จากชื่อราศี: ราศี{chart_info_for_zodiac['zodiac_sign']}")
                reply_text = ask_question_to_rag(user_text, user_id=user_id, provided_chart_info=chart_info_for_zodiac)
            else:
                reply_text = ask_question_to_rag(user_text, user_id=user_id)
            # ป้องกันกรณีที่คำตอบไม่ใช่สตริง หรือเป็น None
            if not isinstance(reply_text, str):
                logger.warning(f"reply_text ไม่ใช่ str (type={type(reply_text)}), แปลงเป็น string")
                reply_text = "" if reply_text is None else str(reply_text)
            logger.info(f"ได้รับคำตอบ (ความยาว: {len(reply_text)} ตัวอักษร)")
    except Exception as e:
        import traceback
        logger.error(f"เกิดข้อผิดพลาดในการประมวลผล: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        # บันทึกบริบทสำคัญช่วยดีบัก
        try:
            logger.error(f"DEBUG context -> user_id={user_id}, text_len={len(user_text)}, has_openai_key={bool(os.getenv('OPENAI_API_KEY'))}, model={os.getenv('OPENAI_MODEL', 'gpt-4o-mini')}")
        except Exception:
            pass
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
    
    # try:
    #     log_pretty_answer(user_id, "final_reply", reply_text)
    # except Exception:
    #     pass
    return TextMessage(text=reply_text)