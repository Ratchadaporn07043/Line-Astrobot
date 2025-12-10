#!/bin/bash

echo "============================================================"
echo "🔍 ตรวจสอบการตั้งค่า Google Sheets"
echo "============================================================"
echo ""

# ตรวจสอบ environment variables
echo "📋 Environment Variables:"
if [ -f .env ]; then
    echo "  ✅ พบไฟล์ .env"
    grep "GOOGLE_SHEETS" .env | sed 's/^/  /'
else
    echo "  ❌ ไม่พบไฟล์ .env"
fi

echo ""
echo "============================================================"
echo "📋 ขั้นตอนที่ต้องทำ:"
echo "============================================================"
echo ""
echo "1️⃣  เปิดใช้งาน Google Sheets API:"
echo "   ⚡ คลิกลิงก์นี้แล้วกด Enable:"
echo "   https://console.developers.google.com/apis/api/sheets.googleapis.com/overview?project=727945824572"
echo ""
echo "2️⃣  แชร์ Spreadsheet กับ Service Account:"
echo "   📊 เปิด Spreadsheet:"
echo "   https://docs.google.com/spreadsheets/d/1xDMHBIfgLVK-ORdVCSrK4K2MoBXXT0vBhTOYRqXhCFE/edit"
echo ""
echo "   📧 เพิ่ม email นี้:"
echo "   ragas-evaluation@ragas-480809.iam.gserviceaccount.com"
echo ""
echo "   🔑 ตั้งสิทธิ์เป็น: Editor"
echo ""
echo "3️⃣  รอ 1-2 นาที แล้วรันทดสอบอีกครั้ง:"
echo "   python3 test_google_sheets_connection.py"
echo ""
echo "============================================================"
