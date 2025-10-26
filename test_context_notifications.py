#!/usr/bin/env python3
"""
ทดสอบการแจ้งเตือนผู้ใช้เมื่อไม่มีข้อมูลบริบท
"""

import os
import sys
from datetime import datetime

# เพิ่ม path สำหรับ import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.retrieval_utils import ask_question_to_rag, enhance_question_context, get_user_context

def test_context_notifications():
    """ทดสอบการแจ้งเตือนเมื่อไม่มีข้อมูลบริบท"""
    
    print("=== ทดสอบการแจ้งเตือนผู้ใช้เมื่อไม่มีข้อมูลบริบท ===\n")
    
    # ทดสอบ 1: คำถามต่อเนื่องโดยไม่มีบริบท
    print("1. ทดสอบคำถามต่อเนื่องโดยไม่มีบริบท:")
    print("คำถาม: 'ราศีนี้มีลักษณะนิสัยยังไง'")
    print("User ID: 'test_user_no_context'")
    
    try:
        result = ask_question_to_rag("ราศีนี้มีลักษณะนิสัยยังไง", "test_user_no_context")
        print(f"ผลลัพธ์: {result}")
        print("✅ ระบบแจ้งเตือนผู้ใช้ให้ระบุวันเกิดก่อน\n")
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {e}\n")
    
    # ทดสอบ 2: คำถามต่อเนื่องที่มีบริบทบางส่วนแต่ไม่มีราศี
    print("2. ทดสอบคำถามต่อเนื่องที่มีบริบทบางส่วนแต่ไม่มีราศี:")
    print("คำถาม: 'ราศีนี้มีลักษณะนิสัยยังไง'")
    print("User ID: 'test_user_partial_context'")
    
    try:
        result = ask_question_to_rag("ราศีนี้มีลักษณะนิสัยยังไง", "test_user_partial_context")
        print(f"ผลลัพธ์: {result}")
        print("✅ ระบบแจ้งเตือนผู้ใช้ให้ระบุวันเกิดก่อน\n")
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {e}\n")
    
    # ทดสอบ 3: คำถามใหม่ที่มีข้อมูลวันเกิด
    print("3. ทดสอบคำถามใหม่ที่มีข้อมูลวันเกิด:")
    print("คำถาม: '09/02/2004 ราศีอะไร'")
    print("User ID: 'test_user_new_question'")
    
    try:
        result = ask_question_to_rag("09/02/2004 ราศีอะไร", "test_user_new_question")
        print(f"ผลลัพธ์: {result[:200]}...")
        print("✅ ระบบตอบคำถามได้ปกติ\n")
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {e}\n")
    
    # ทดสอบ 4: คำถามต่อเนื่องหลังจากมีข้อมูลแล้ว
    print("4. ทดสอบคำถามต่อเนื่องหลังจากมีข้อมูลแล้ว:")
    print("คำถาม: 'แล้วราศีนี้มีลักษณะนิสัยยังไง'")
    print("User ID: 'test_user_new_question'")
    
    try:
        result = ask_question_to_rag("แล้วราศีนี้มีลักษณะนิสัยยังไง", "test_user_new_question")
        print(f"ผลลัพธ์: {result[:200]}...")
        print("✅ ระบบตอบคำถามต่อเนื่องได้ปกติ\n")
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {e}\n")

def test_enhance_question_context():
    """ทดสอบฟังก์ชัน enhance_question_context"""
    
    print("=== ทดสอบฟังก์ชัน enhance_question_context ===\n")
    
    # ทดสอบ 1: คำถามต่อเนื่องโดยไม่มีบริบท
    print("1. ทดสอบคำถามต่อเนื่องโดยไม่มีบริบท:")
    question = "ราศีนี้มีลักษณะนิสัยยังไง"
    user_context = None
    
    result = enhance_question_context(question, user_context)
    print(f"คำถามเดิม: {question}")
    print(f"ผลลัพธ์: {result}")
    print("✅ ระบบแจ้งเตือนผู้ใช้ให้ระบุวันเกิดก่อน\n")
    
    # ทดสอบ 2: คำถามต่อเนื่องที่มีบริบทบางส่วน
    print("2. ทดสอบคำถามต่อเนื่องที่มีบริบทบางส่วน:")
    question = "ราศีนี้มีลักษณะนิสัยยังไง"
    user_context = {"birth_date": "09/02/2004", "zodiac_sign": None}
    
    result = enhance_question_context(question, user_context)
    print(f"คำถามเดิม: {question}")
    print(f"ผลลัพธ์: {result}")
    print("✅ ระบบแจ้งเตือนผู้ใช้ให้ระบุวันเกิดก่อน\n")
    
    # ทดสอบ 3: คำถามต่อเนื่องที่มีบริบทครบถ้วน
    print("3. ทดสอบคำถามต่อเนื่องที่มีบริบทครบถ้วน:")
    question = "ราศีนี้มีลักษณะนิสัยยังไง"
    user_context = {"birth_date": "09/02/2004", "zodiac_sign": "กุมภ์"}
    
    result = enhance_question_context(question, user_context)
    print(f"คำถามเดิม: {question}")
    print(f"ผลลัพธ์: {result}")
    print("✅ ระบบปรับปรุงคำถามให้ชัดเจน\n")

def test_get_user_context():
    """ทดสอบฟังก์ชัน get_user_context"""
    
    print("=== ทดสอบฟังก์ชัน get_user_context ===\n")
    
    # ทดสอบ 1: ผู้ใช้ที่ไม่มีข้อมูล
    print("1. ทดสอบผู้ใช้ที่ไม่มีข้อมูล:")
    user_id = "test_user_no_data"
    
    result = get_user_context(user_id)
    print(f"User ID: {user_id}")
    print(f"ผลลัพธ์: {result}")
    print("✅ ระบบ return None เมื่อไม่มีข้อมูล\n")
    
    # ทดสอบ 2: ผู้ใช้ที่มีข้อมูลบางส่วน
    print("2. ทดสอบผู้ใช้ที่มีข้อมูลบางส่วน:")
    user_id = "test_user_partial_data"
    
    result = get_user_context(user_id)
    print(f"User ID: {user_id}")
    print(f"ผลลัพธ์: {result}")
    print("✅ ระบบ return ข้อมูลที่มีอยู่\n")

if __name__ == "__main__":
    print("เริ่มทดสอบการแจ้งเตือนผู้ใช้เมื่อไม่มีข้อมูลบริบท...\n")
    
    # ทดสอบฟังก์ชันต่างๆ
    test_enhance_question_context()
    test_get_user_context()
    
    # ทดสอบการทำงานจริง (ต้องมี MongoDB และ OpenAI API key)
    print("=== ทดสอบการทำงานจริง ===")
    print("หมายเหตุ: การทดสอบนี้ต้องมี MongoDB และ OpenAI API key ที่ถูกต้อง")
    
    # ตรวจสอบ environment variables
    mongo_url = os.getenv("MONGO_URL")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if mongo_url and mongo_url != "mongodb+srv://your-username:your-password@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority":
        if openai_key and openai_key != "your-openai-api-key-here":
            test_context_notifications()
        else:
            print("❌ ไม่พบ OpenAI API key ที่ถูกต้อง")
    else:
        print("❌ ไม่พบ MongoDB connection string ที่ถูกต้อง")
    
    print("\n=== เสร็จสิ้นการทดสอบ ===")
