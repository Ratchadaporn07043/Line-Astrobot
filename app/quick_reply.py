# quick_reply_module.py
import os
import json
import logging
from datetime import datetime
from linebot.v3.messaging.models import (
    TextMessage, QuickReply, QuickReplyItem, MessageAction
)
from .retrieval_utils import store_user_response, store_user_question

# ตั้งค่า logger
logger = logging.getLogger(__name__)

# โหลดข้อมูล
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

with open(os.path.join(BASE_DIR, "omenData.json"), encoding='utf-8') as f:
    omenData = json.load(f)
with open(os.path.join(BASE_DIR, "luckyColorData.json"), encoding='utf-8') as f:
    colorData = json.load(f)
with open(os.path.join(BASE_DIR, "zodiacData.json"), encoding='utf-8') as f:
    zodiacData = json.load(f)
with open(os.path.join(BASE_DIR, "zodiacNextMonth.json"), encoding='utf-8') as f:
    nextMonth = json.load(f)

weekdays = ["อาทิตย์", "จันทร์", "อังคาร", "พุธ", "พฤหัสบดี", "ศุกร์", "เสาร์"]
zodiacs = ["ชวด", "ฉลู", "ขาล", "เถาะ", "มะโรง", "มะเส็ง", "มะเมีย", "มะแม", "วอก", "ระกา", "จอ", "กุน"]
userSession = {}

def getZodiac(day, month):
    zodiac_range = [
        (1, 20, 2, 18, 'กุมภ์'), (2, 19, 3, 20, 'มีน'), (3, 21, 4, 19, 'เมษ'),
        (4, 20, 5, 20, 'พฤษภ'), (5, 21, 6, 20, 'เมถุน'), (6, 21, 7, 22, 'กรกฎ'),
        (7, 23, 8, 22, 'สิงห์'), (8, 23, 9, 22, 'กันย์'), (9, 23, 10, 22, 'ตุล'),
        (10, 23, 11, 21, 'พิจิก'), (11, 22, 12, 21, 'ธนู')
    ]
    for m1, d1, m2, d2, name in zodiac_range:
        if (month == m1 and day >= d1) or (month == m2 and day <= d2):
            return name
    return 'มังกร'

def handle_quick_reply(event):
    try:
        user_id = event.source.user_id
        msg = event.message.text.strip()

        if msg == "ดูดวงพื้นฐาน":
            userSession[user_id] = {'mode': 'basic'}
            categories = ["การงาน", "การเงิน", "สุขภาพ", "ความรัก"]
            quick = QuickReply(items=[QuickReplyItem(action=MessageAction(label=cat, text=cat)) for cat in categories])
            return TextMessage(text="กรุณาเลือกหมวดหมู่ที่ต้องการดูดวง", quick_reply=quick)

        elif msg == "ดูดวงรายเดือน":
            userSession[user_id] = {'mode': 'monthly'}
            zodiac_signs = [
                "มังกร (22 ธ.ค. - 19 ม.ค.)",
                "กุมภ์ (20 ม.ค. - 18 ก.พ.)", 
                "มีน (19 ก.พ. - 20 มี.ค.)",
                "เมษ (21 มี.ค. - 19 เม.ย.)",
                "พฤษภ (20 เม.ย. - 20 พ.ค.)",
                "เมถุน (21 พ.ค. - 20 มิ.ย.)",
                "กรกฎ (21 มิ.ย. - 22 ก.ค.)",
                "สิงห์ (23 ก.ค. - 22 ส.ค.)",
                "กันย์ (23 ส.ค. - 22 ก.ย.)",
                "ตุล (23 ก.ย. - 22 ต.ค.)",
                "พิจิก (23 ต.ค. - 21 พ.ย.)",
                "ธนู (22 พ.ย. - 21 ธ.ค.)"
            ]
            quick = QuickReply(items=[QuickReplyItem(action=MessageAction(label=f"ราศี{r}", text=f"ราศี{r}")) for r in zodiac_signs])
            return TextMessage(text="กรุณาเลือกราศีของคุณ", quick_reply=quick)

        elif msg == "เคล็ดลับเสริมดวง":
            userSession[user_id] = {'mode': 'color'}
            quick = QuickReply(items=[QuickReplyItem(action=MessageAction(label=f"วัน{d}", text=f"วัน{d}")) for d in weekdays])
            return TextMessage(text="กรุณาเลือกวันเกิดของคุณ", quick_reply=quick)

        elif msg == "ทำนายดวงเฉพาะด้าน":
            userSession[user_id] = {'mode': 'aspect'}
            categories = ["การงาน", "การเงิน", "สุขภาพ", "ความรัก"]
            quick = QuickReply(items=[QuickReplyItem(action=MessageAction(label=cat, text=cat)) for cat in categories])
            return TextMessage(text="กรุณาเลือกหมวดหมู่ที่ต้องการทำนาย", quick_reply=quick)

        # Handle category selection for basic horoscope
        elif userSession.get(user_id, {}).get("mode") == "basic" and msg in ["การงาน", "การเงิน", "สุขภาพ", "ความรัก"]:
            userSession[user_id]['category'] = msg
            quick = QuickReply(items=[QuickReplyItem(action=MessageAction(label=f"วัน{d}", text=f"วัน{d}")) for d in weekdays])
            return TextMessage(text="กรุณาเลือกวันเกิดของคุณ", quick_reply=quick)

        # Handle category selection for aspect horoscope
        elif userSession.get(user_id, {}).get("mode") == "aspect" and msg in ["การงาน", "การเงิน", "สุขภาพ", "ความรัก"]:
            userSession[user_id]['category'] = msg
            return TextMessage(text="กรุณากรอกวันเกิดของคุณในรูปแบบ dd/mm/yyyy เช่น 01/01/2000")

        elif userSession.get(user_id, {}).get("mode") == "aspect" and '/' in msg:
            try:
                # ใช้ BirthDateParser เพื่อแยกข้อมูลวันเกิดและเวลาเกิด
                from .birth_date_parser import BirthDateParser
                parser = BirthDateParser()
                birth_info = parser.extract_birth_info(msg)
                
                if not birth_info or not birth_info['date']:
                    return TextMessage(text="❌ กรุณากรอกวันเกิดให้ถูกต้อง เช่น 01/01/2000")
                
                # แปลงวันเกิด
                bday = datetime.strptime(birth_info['date'], '%d/%m/%Y')
                zodiac = getZodiac(bday.day, bday.month)
                
                # สร้างคำถามสำหรับ AI ตามหมวดหมู่ที่เลือก
                session = userSession.get(user_id, {})
                category = session.get('category', '')
                
                if category == "การงาน":
                    ai_question = f"ทำนายดวงการงานสำหรับคนที่เกิดวันที่ {birth_info['date']} (ราศี{zodiac})"
                elif category == "การเงิน":
                    ai_question = f"ทำนายดวงการเงินสำหรับคนที่เกิดวันที่ {birth_info['date']} (ราศี{zodiac})"
                elif category == "สุขภาพ":
                    ai_question = f"ทำนายดวงสุขภาพสำหรับคนที่เกิดวันที่ {birth_info['date']} (ราศี{zodiac})"
                elif category == "ความรัก":
                    ai_question = f"ทำนายดวงความรักสำหรับคนที่เกิดวันที่ {birth_info['date']} (ราศี{zodiac})"
                else:
                    ai_question = f"ทำนายดวงเฉพาะด้านสำหรับคนที่เกิดวันที่ {birth_info['date']} (ราศี{zodiac})"
                
                # ใช้ AI สร้างคำตอบ
                from .retrieval_utils import ask_question_to_rag
                response_text = ask_question_to_rag(ai_question, user_id)
                
                # บันทึกคำถามใน user_profiles
                store_user_question(
                    question=ai_question,
                    user_id=user_id,
                    context_data={"zodiac": zodiac, "birth_date": birth_info['date'], "birth_time": birth_info.get('time'), "mode": "aspect", "category": category}
                )
                
                # บันทึกคำตอบใน collection astrobot
                store_user_response(
                    question=ai_question,
                    answer=response_text,
                    user_id=user_id,
                    response_type="quick_reply_aspect_ai",
                    context_data={"zodiac": zodiac, "birth_date": birth_info['date'], "birth_time": birth_info.get('time'), "mode": "aspect", "category": category}
                )
                
                return TextMessage(text=response_text)
            except Exception as e:
                return TextMessage(text="❌ กรุณากรอกวันเกิดให้ถูกต้อง เช่น 01/01/2000")

        elif msg.startswith("วัน") and userSession.get(user_id, {}).get("mode") == "color":
            d = msg.replace("วัน", "").strip()
            if d in weekdays:
                # สร้างคำถามสำหรับ AI
                ai_question = f"เคล็ดลับเสริมดวงและสีมงคลสำหรับคนที่เกิดวัน{d}"
                
                # ใช้ AI สร้างคำตอบ
                from .retrieval_utils import ask_question_to_rag
                response_text = ask_question_to_rag(ai_question, user_id)
                
                # บันทึกคำถามใน user_profiles
                store_user_question(
                    question=ai_question,
                    user_id=user_id,
                    context_data={"day": d, "mode": "color"}
                )
                
                # บันทึกคำตอบใน collection astrobot
                store_user_response(
                    question=ai_question,
                    answer=response_text,
                    user_id=user_id,
                    response_type="quick_reply_color_ai",
                    context_data={"day": d, "mode": "color"}
                )
                
                return TextMessage(text=response_text)

        elif msg.startswith("ราศี") and userSession.get(user_id, {}).get("mode") == "monthly":
            # Extract zodiac sign name from text that includes date range
            zodiac_text = msg.replace("ราศี", "").strip()
            # Remove date range in parentheses to get just the zodiac name
            zodiac = zodiac_text.split(" (")[0].strip()
            
            # สร้างคำถามสำหรับ AI
            ai_question = f"ดูดวงรายเดือนสำหรับราศี{zodiac} ครอบคลุมการงาน การเงิน สุขภาพ และความรัก"
            
            # ใช้ AI สร้างคำตอบ
            from .retrieval_utils import ask_question_to_rag
            response_text = ask_question_to_rag(ai_question, user_id)
            
            # บันทึกคำถามใน user_profiles
            store_user_question(
                question=ai_question,
                user_id=user_id,
                context_data={"zodiac": zodiac, "mode": "monthly"}
            )
            
            # บันทึกคำตอบใน collection astrobot
            store_user_response(
                question=ai_question,
                answer=response_text,
                user_id=user_id,
                response_type="quick_reply_monthly_ai",
                context_data={"zodiac": zodiac, "mode": "monthly"}
            )
            
            return TextMessage(text=response_text)

        elif msg.startswith("วัน") and userSession.get(user_id, {}).get("mode") == "basic":
            d = msg.replace("วัน", "").strip()
            if d in weekdays:
                userSession[user_id]['day'] = d
                quick = QuickReply(items=[QuickReplyItem(action=MessageAction(label=f"ปี{z}", text=f"ปี{z}")) for z in zodiacs])
                return TextMessage(text="กรุณาเลือกปีนักษัตรของคุณ", quick_reply=quick)

        elif msg.startswith("ปี") and userSession.get(user_id, {}).get("mode") == "basic":
            zodiac = msg
            session = userSession.get(user_id, {})
            day = session.get('day')
            category = session.get('category', '')
            
            if day and zodiac:
                # สร้างคำถามสำหรับ AI ตามหมวดหมู่ที่เลือก
                if category == "การงาน":
                    ai_question = f"ดูดวงพื้นฐานการงานสำหรับคนที่เกิดวัน{day} ปีนักษัตร{zodiac}"
                elif category == "การเงิน":
                    ai_question = f"ดูดวงพื้นฐานการเงินสำหรับคนที่เกิดวัน{day} ปีนักษัตร{zodiac}"
                elif category == "สุขภาพ":
                    ai_question = f"ดูดวงพื้นฐานสุขภาพสำหรับคนที่เกิดวัน{day} ปีนักษัตร{zodiac}"
                elif category == "ความรัก":
                    ai_question = f"ดูดวงพื้นฐานความรักสำหรับคนที่เกิดวัน{day} ปีนักษัตร{zodiac}"
                else:
                    ai_question = f"ดูดวงพื้นฐานสำหรับคนที่เกิดวัน{day} ปีนักษัตร{zodiac}"
                
                # ใช้ AI สร้างคำตอบ
                from .retrieval_utils import ask_question_to_rag
                response_text = ask_question_to_rag(ai_question, user_id)
                
                # บันทึกคำถามใน user_profiles
                store_user_question(
                    question=ai_question,
                    user_id=user_id,
                    context_data={"day": day, "zodiac": zodiac, "mode": "basic", "category": category}
                )
                
                # บันทึกคำตอบใน collection astrobot
                store_user_response(
                    question=ai_question,
                    answer=response_text,
                    user_id=user_id,
                    response_type="quick_reply_basic_ai",
                    context_data={"day": day, "zodiac": zodiac, "mode": "basic", "category": category}
                )
                
                return TextMessage(text=response_text)

    except Exception as e:
        return TextMessage(text=f"❌ เกิดข้อผิดพลาด: {str(e)}")

    return None

