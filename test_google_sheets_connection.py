#!/usr/bin/env python3
"""
สคริปต์ทดสอบการเชื่อมต่อ Google Sheets
"""

import os
import sys
from dotenv import load_dotenv

# โหลด environment variables
load_dotenv()

print("="*60)
print("🔍 ตรวจสอบการตั้งค่า Google Sheets")
print("="*60)

# 1. ตรวจสอบ environment variables
print("\n📋 Environment Variables:")
googlesheets_enabled = os.getenv("GOOGLE_SHEETS_ENABLED", "false")
credentials_path = os.getenv("GOOGLE_SHEETS_CREDENTIALS_PATH")
credentials_json = os.getenv("GOOGLE_SHEETS_CREDENTIALS")
spreadsheet_id = os.getenv("GOOGLE_SHEETS_ID")
worksheet_name = os.getenv("GOOGLE_SHEETS_WORKSHEET_NAME", "RAGAS Evaluation")

print(f"  GOOGLE_SHEETS_ENABLED: {googlesheets_enabled}")
print(f"  GOOGLE_SHEETS_CREDENTIALS_PATH: {credentials_path or 'ไม่พบ'}")
print(f"  GOOGLE_SHEETS_CREDENTIALS: {'ตั้งค่าแล้ว' if credentials_json else 'ไม่พบ'}")
print(f"  GOOGLE_SHEETS_ID: {spreadsheet_id or 'ไม่พบ'}")
print(f"  GOOGLE_SHEETS_WORKSHEET_NAME: {worksheet_name}")

# 2. ตรวจสอบว่าเปิดใช้งานหรือไม่
if googlesheets_enabled.lower() != "true":
    print("\n⚠️  GOOGLE_SHEETS_ENABLED ไม่ได้ตั้งค่าเป็น 'true'")
    print("   ตั้งค่า GOOGLE_SHEETS_ENABLED=true ใน .env")
    sys.exit(1)

# 3. ตรวจสอบ credentials path
if credentials_path:
    print(f"\n📁 ตรวจสอบไฟล์ credentials: {credentials_path}")
    if os.path.exists(credentials_path):
        print("  ✅ พบไฟล์ credentials")
        try:
            import json
            with open(credentials_path, 'r') as f:
                creds_data = json.load(f)
                print(f"  ✅ ไฟล์ JSON ถูกต้อง")
                print(f"  📧 Service Account Email: {creds_data.get('client_email', 'ไม่พบ')}")
                print(f"  📋 Project ID: {creds_data.get('project_id', 'ไม่พบ')}")
        except json.JSONDecodeError:
            print("  ❌ ไฟล์ JSON ไม่ถูกต้อง")
            sys.exit(1)
        except Exception as e:
            print(f"  ❌ Error อ่านไฟล์: {e}")
            sys.exit(1)
    else:
        print(f"  ❌ ไม่พบไฟล์ credentials ที่ path: {credentials_path}")
        sys.exit(1)
elif credentials_json:
    print("\n📁 ใช้ credentials จาก environment variable")
    try:
        import json
        creds_data = json.loads(credentials_json)
        print(f"  ✅ JSON string ถูกต้อง")
        print(f"  📧 Service Account Email: {creds_data.get('client_email', 'ไม่พบ')}")
    except json.JSONDecodeError:
        print("  ❌ JSON string ไม่ถูกต้อง")
        sys.exit(1)
else:
    print("\n❌ ไม่พบ credentials (GOOGLE_SHEETS_CREDENTIALS_PATH หรือ GOOGLE_SHEETS_CREDENTIALS)")
    sys.exit(1)

# 4. ตรวจสอบ spreadsheet ID
if not spreadsheet_id or spreadsheet_id == "your_spreadsheet_id_here":
    print("\n❌ ไม่พบ GOOGLE_SHEETS_ID หรือยังไม่ได้ตั้งค่า")
    print("   ตั้งค่า GOOGLE_SHEETS_ID ใน .env")
    sys.exit(1)

# แยก Spreadsheet ID จาก URL (ถ้ามี)
if "/d/" in spreadsheet_id:
    # แยก ID จาก URL เช่น: https://docs.google.com/spreadsheets/d/ID/edit
    parts = spreadsheet_id.split("/d/")
    if len(parts) > 1:
        spreadsheet_id = parts[1].split("/")[0].split("?")[0].split("#")[0]
        print(f"\n📊 แยก Spreadsheet ID จาก URL: {spreadsheet_id}")
    else:
        print(f"\n📊 Spreadsheet ID: {spreadsheet_id}")
else:
    print(f"\n📊 Spreadsheet ID: {spreadsheet_id}")

# 5. ทดสอบการเชื่อมต่อ
print("\n" + "="*60)
print("🔄 กำลังทดสอบการเชื่อมต่อ Google Sheets...")
print("="*60)

try:
    import gspread
    from google.oauth2.service_account import Credentials
    
    # โหลด credentials
    if credentials_path:
        creds = Credentials.from_service_account_file(
            credentials_path,
            scopes=['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
        )
    else:
        creds = Credentials.from_service_account_info(
            json.loads(credentials_json),
            scopes=['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
        )
    
    # เชื่อมต่อ
    client = gspread.authorize(creds)
    print("✅ เชื่อมต่อ Google Sheets API สำเร็จ")
    
    # เปิด spreadsheet
    try:
        print(f"   กำลังเปิด Spreadsheet ID: {spreadsheet_id}...")
        spreadsheet = client.open_by_key(spreadsheet_id)
        print(f"✅ เปิด Spreadsheet สำเร็จ: {spreadsheet.title}")
        print(f"   URL: {spreadsheet.url}")
        
        # ตรวจสอบ worksheet
        try:
            worksheet = spreadsheet.worksheet(worksheet_name)
            print(f"✅ พบ Worksheet: {worksheet_name}")
            print(f"   จำนวน rows: {worksheet.row_count}")
            print(f"   จำนวน cols: {worksheet.col_count}")
        except gspread.exceptions.WorksheetNotFound:
            print(f"⚠️  ไม่พบ Worksheet: {worksheet_name}")
            print(f"   Worksheet จะถูกสร้างอัตโนมัติเมื่อรันการประเมิน")
        
        # ทดสอบเขียนข้อมูล
        print("\n🧪 ทดสอบเขียนข้อมูล...")
        test_worksheet_name = "Test Connection"
        try:
            test_worksheet = spreadsheet.worksheet(test_worksheet_name)
            test_worksheet.clear()
        except gspread.exceptions.WorksheetNotFound:
            test_worksheet = spreadsheet.add_worksheet(title=test_worksheet_name, rows=10, cols=5)
        
        test_data = [
            ["Timestamp", "Status", "Message"],
            ["2024-01-01 00:00:00", "Success", "การเชื่อมต่อ Google Sheets ทำงานได้ปกติ"]
        ]
        test_worksheet.update(values=test_data, range_name='A1:C2')
        print("✅ ทดสอบเขียนข้อมูลสำเร็จ")
        print(f"   ตรวจสอบได้ที่: {spreadsheet.url}#gid={test_worksheet.id}")
        
        print("\n" + "="*60)
        print("✅ การเชื่อมต่อ Google Sheets ทำงานได้ปกติ!")
        print("="*60)
        
    except gspread.exceptions.SpreadsheetNotFound:
        print(f"❌ ไม่พบ Spreadsheet ที่ ID: {spreadsheet_id}")
        print("   ตรวจสอบว่า:")
        print("   1. Spreadsheet ID ถูกต้อง")
        print("   2. Service Account มีสิทธิ์เข้าถึง Spreadsheet")
        print("   3. แชร์ Spreadsheet กับ Service Account Email แล้ว")
        sys.exit(1)
    except PermissionError as e:
        error_msg = str(e)
        print(f"\n❌ Permission Error")
        
        # พยายามดึง error message จาก exception
        import traceback
        full_error = traceback.format_exc()
        
        # ตรวจสอบ error message เพื่อให้คำแนะนำที่เหมาะสม
        if "API has not been used" in full_error or "is disabled" in full_error or "403" in full_error:
            # ดึง project number จาก error message
            import re
            project_match = re.search(r'project (\d+)', full_error)
            project_number = project_match.group(1) if project_match else "727945824572"
            
            print("\n📋 Google Sheets API ยังไม่ได้เปิดใช้งาน")
            print(f"   Project Number: {project_number}")
            print("\n   วิธีแก้ไข:")
            print("   ⚡ วิธีที่ 1: เปิดใช้งานผ่านลิงก์โดยตรง (แนะนำ)")
            print(f"      https://console.developers.google.com/apis/api/sheets.googleapis.com/overview?project={project_number}")
            print("      → คลิก 'Enable'")
            print("\n   ⚡ วิธีที่ 2: เปิดใช้งานผ่าน API Library")
            print("      https://console.cloud.google.com/apis/library/sheets.googleapis.com")
            print("      → เลือกโปรเจกต์ที่ถูกต้อง")
            print("      → คลิก 'Enable'")
            print("\n   ⏱️  รอ 1-2 นาที หลังจากเปิดใช้งาน แล้วลองอีกครั้ง")
            print("\n   📋 และตรวจสอบว่า Spreadsheet แชร์กับ Service Account แล้ว:")
            print(f"      1. เปิด: https://docs.google.com/spreadsheets/d/{spreadsheet_id}/edit")
            print("      2. คลิก 'Share' (มุมขวาบน)")
            print(f"      3. เพิ่ม email: ragas-evaluation@ragas-480809.iam.gserviceaccount.com")
            print("      4. ตั้งสิทธิ์เป็น 'Editor'")
            print("      5. คลิก 'Send'")
        else:
            print(f"\n   Error Details: {error_msg}")
            print("\n📋 ตรวจสอบทั้ง 2 อย่าง:")
            print("   1. Google Sheets API เปิดใช้งานแล้ว")
            print("      https://console.cloud.google.com/apis/library/sheets.googleapis.com?project=ragas-480809")
            print("   2. Spreadsheet แชร์กับ Service Account แล้ว")
            print(f"      Email: ragas-evaluation@ragas-480809.iam.gserviceaccount.com")
            print(f"      Spreadsheet: https://docs.google.com/spreadsheets/d/{spreadsheet_id}/edit")
        
        print("\n📝 Full Error:")
        print(full_error)
        sys.exit(1)
    except Exception as e:
        error_msg = str(e)
        print(f"\n❌ Error: {error_msg}")
        
        if "API has not been used" in error_msg or "is disabled" in error_msg or "403" in error_msg:
            print("\n📋 Google Sheets API ยังไม่ได้เปิดใช้งาน หรือไม่มีสิทธิ์")
            print("   วิธีแก้ไข:")
            print("   1. เปิดใช้งาน Google Sheets API:")
            print("      https://console.cloud.google.com/apis/library/sheets.googleapis.com?project=ragas-480809")
            print("   2. แชร์ Spreadsheet กับ Service Account:")
            print(f"      https://docs.google.com/spreadsheets/d/{spreadsheet_id}/edit")
            print(f"      Email: ragas-evaluation@ragas-480809.iam.gserviceaccount.com")
            print("   3. รอ 1-2 นาที แล้วลองอีกครั้ง")
        else:
            print("\n📋 Error Details:")
            import traceback
            traceback.print_exc()
        sys.exit(1)
        
except ImportError:
    print("❌ ไม่พบ gspread library")
    print("   ติดตั้งด้วย: pip install gspread google-auth")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error เชื่อมต่อ Google Sheets: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
