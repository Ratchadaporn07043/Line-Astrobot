#!/bin/bash

echo "============================================================"
echo "🔄 สร้าง Dataset ใหม่ (กรองคำตอบที่บอกว่าไม่มีข้อมูล)"
echo "============================================================"
echo ""

# 1. สำรอง dataset เก่า
if [ -f "dataset_from_mongo.json" ]; then
    backup_file="dataset_from_mongo_backup_$(date +%Y%m%d_%H%M%S).json"
    cp dataset_from_mongo.json "$backup_file"
    echo "✅ สำรอง dataset เก่าเป็น: $backup_file"
fi

# 2. รันสคริปต์สร้าง dataset ใหม่
echo ""
echo "🔄 กำลังสร้าง dataset ใหม่..."
echo ""

python3 generate_ragas_dataset_from_mongo.py 2>&1 | tee generate_log_$(date +%Y%m%d_%H%M%S).txt

echo ""
echo "============================================================"
echo "✅ เสร็จสิ้น"
echo "============================================================"
echo ""
echo "📋 ตรวจสอบผลลัพธ์:"
echo "   python3 check_dataset_quality.py"
echo ""
echo "📊 ตรวจสอบ Google Sheets:"
echo "   https://docs.google.com/spreadsheets/d/1xDMHBIfgLVK-ORdVCSrK4K2MoBXXT0vBhTOYRqXhCFE"
