#!/usr/bin/env python3
"""
สคริปต์ทดสอบการบันทึกข้อมูลลง Google Sheets
"""

import os
import sys
from dotenv import load_dotenv

# โหลด environment variables
load_dotenv()

print("="*60)
print("🧪 ทดสอบการบันทึกข้อมูลลง Google Sheets")
print("="*60)

# ตรวจสอบ environment variables
googlesheets_enabled = os.getenv("GOOGLE_SHEETS_ENABLED", "false")
credentials_path = os.getenv("GOOGLE_SHEETS_CREDENTIALS_PATH")
spreadsheet_id = os.getenv("GOOGLE_SHEETS_ID")

print(f"\n📋 Environment Variables:")
print(f"  GOOGLE_SHEETS_ENABLED: {googlesheets_enabled}")
print(f"  GOOGLE_SHEETS_CREDENTIALS_PATH: {credentials_path}")
print(f"  GOOGLE_SHEETS_ID: {spreadsheet_id}")

if googlesheets_enabled.lower() != "true":
    print("\n❌ GOOGLE_SHEETS_ENABLED ไม่ได้ตั้งค่าเป็น 'true'")
    sys.exit(1)

if not credentials_path or not os.path.exists(credentials_path):
    print(f"\n❌ ไม่พบไฟล์ credentials: {credentials_path}")
    sys.exit(1)

if not spreadsheet_id:
    print("\n❌ ไม่พบ GOOGLE_SHEETS_ID")
    sys.exit(1)

# แยก Spreadsheet ID จาก URL (ถ้ามี)
if "/d/" in spreadsheet_id:
    parts = spreadsheet_id.split("/d/")
    if len(parts) > 1:
        spreadsheet_id = parts[1].split("/")[0].split("?")[0].split("#")[0]
        print(f"\n📊 แยก Spreadsheet ID: {spreadsheet_id}")

# ทดสอบบันทึกข้อมูล
print("\n" + "="*60)
print("🔄 กำลังทดสอบการบันทึกข้อมูล...")
print("="*60)

try:
    import gspread
    from google.oauth2.service_account import Credentials
    
    # โหลด credentials
    print(f"\n📁 กำลังโหลด credentials จาก: {credentials_path}")
    creds = Credentials.from_service_account_file(
        credentials_path,
        scopes=['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
    )
    client = gspread.authorize(creds)
    print("✅ เชื่อมต่อ Google Sheets API สำเร็จ")
    
    # เปิด spreadsheet
    print(f"\n📊 กำลังเปิด Spreadsheet: {spreadsheet_id}")
    spreadsheet = client.open_by_key(spreadsheet_id)
    print(f"✅ เปิด Spreadsheet สำเร็จ: {spreadsheet.title}")
    
    # ตรวจสอบหรือสร้าง worksheet "Dataset"
    worksheet_name = "Dataset"
    try:
        worksheet = spreadsheet.worksheet(worksheet_name)
        print(f"✅ พบ worksheet: {worksheet_name}")
        worksheet.clear()
        print("🗑️ ล้างข้อมูลเก่า")
    except gspread.exceptions.WorksheetNotFound:
        worksheet = spreadsheet.add_worksheet(title=worksheet_name, rows=1000, cols=10)
        print(f"✅ สร้าง worksheet ใหม่: {worksheet_name}")
    
    # เตรียมข้อมูลทดสอบ
    headers = [
        "คำถาม",
        "คำตอบ (Ground Truth)",
        "หน้า",
        "Similarity",
        "ประเภท",
        "แหล่งที่มา",
        "Contexts"
    ]
    
    test_data = [
        ["ทดสอบคำถาม 1", "ทดสอบคำตอบ 1", "1", 0.85, "text", "test", "context 1"],
        ["ทดสอบคำถาม 2", "ทดสอบคำตอบ 2", "2", 0.90, "text", "test", "context 2"],
    ]
    
    # บันทึก headers
    print(f"\n📝 กำลังบันทึก headers...")
    worksheet.update(values=[headers], range_name='A1:G1')
    print("✅ บันทึก headers สำเร็จ")
    
    # บันทึกข้อมูล
    print(f"\n📝 กำลังบันทึกข้อมูลทดสอบ ({len(test_data)} rows)...")
    worksheet.update(values=test_data, range_name=f'A2:G{len(test_data)+1}')
    print("✅ บันทึกข้อมูลสำเร็จ")
    
    # Format header row
    print(f"\n🎨 กำลัง format header row...")
    worksheet.format('A1:G1', {
        'backgroundColor': {'red': 0.2, 'green': 0.4, 'blue': 0.8},
        'textFormat': {'bold': True, 'foregroundColor': {'red': 1.0, 'green': 1.0, 'blue': 1.0}}
    })
    print("✅ Format header สำเร็จ")
    
    print("\n" + "="*60)
    print("✅ การทดสอบสำเร็จ!")
    print("="*60)
    print(f"\n📊 ตรวจสอบได้ที่: {spreadsheet.url}#gid={worksheet.id}")
    print(f"📋 Worksheet: {worksheet_name}")
    
except ImportError:
    print("\n❌ ไม่พบ gspread library")
    print("   ติดตั้งด้วย: pip install gspread google-auth")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
