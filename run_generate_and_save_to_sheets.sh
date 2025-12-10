#!/bin/bash

echo "============================================================"
echo "🚀 รัน generate_ragas_dataset_from_mongo.py"
echo "============================================================"
echo ""

# ตรวจสอบ environment variables
echo "📋 ตรวจสอบ Environment Variables:"
python3 -c "
from dotenv import load_dotenv
import os
load_dotenv()
print(f'  GOOGLE_SHEETS_ENABLED: {os.getenv(\"GOOGLE_SHEETS_ENABLED\", \"false\")}')
print(f'  GOOGLE_SHEETS_CREDENTIALS_PATH: {os.getenv(\"GOOGLE_SHEETS_CREDENTIALS_PATH\", \"ไม่พบ\")}')
print(f'  GOOGLE_SHEETS_ID: {os.getenv(\"GOOGLE_SHEETS_ID\", \"ไม่พบ\")}')
"

echo ""
echo "============================================================"
echo "🔄 กำลังรัน generate_ragas_dataset_from_mongo.py..."
echo "============================================================"
echo ""

# รันสคริปต์
python3 generate_ragas_dataset_from_mongo.py 2>&1 | tee generate_log.txt

echo ""
echo "============================================================"
echo "✅ เสร็จสิ้น"
echo "============================================================"
echo ""
echo "📋 ตรวจสอบ log:"
echo "   tail -50 generate_log.txt"
echo ""
echo "📊 ตรวจสอบ Google Sheets:"
echo "   https://docs.google.com/spreadsheets/d/1xDMHBIfgLVK-ORdVCSrK4K2MoBXXT0vBhTOYRqXhCFE"
