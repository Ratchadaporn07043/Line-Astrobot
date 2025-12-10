#!/bin/bash

# สคริปต์สำหรับรันการประเมิน Ragas

echo "🚀 เริ่มการประเมิน Ragas สำหรับแชทบอทโหราศาสตร์"
echo "=========================================="

# ตรวจสอบว่ามีไฟล์ dataset หรือไม่ (ลอง dataset_from_mongo.json ก่อน)
if [ ! -f "dataset_from_mongo.json" ] && [ ! -f "test_dataset.json" ]; then
    echo "❌ ไม่พบไฟล์ dataset"
    echo "กรุณาสร้างไฟล์ dataset_from_mongo.json หรือ test_dataset.json ก่อน"
    echo "💡 ใช้คำสั่ง: ./run_generate_dataset.sh เพื่อสร้าง dataset จาก MongoDB"
    exit 1
fi

# แสดงไฟล์ dataset ที่จะใช้
if [ -f "dataset_from_mongo.json" ]; then
    echo "📁 ใช้ dataset: dataset_from_mongo.json"
elif [ -f "test_dataset.json" ]; then
    echo "📁 ใช้ dataset: test_dataset.json"
fi

# ตรวจสอบว่ามีไฟล์ .env หรือไม่
if [ ! -f ".env" ]; then
    echo "⚠️ ไม่พบไฟล์ .env"
    echo "กรุณาสร้างไฟล์ .env และตั้งค่า MONGO_URL และ OPENAI_API_KEY"
fi

# ตรวจสอบว่ามี python3 หรือ python
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "❌ ไม่พบ python หรือ python3"
    echo "กรุณาติดตั้ง Python ก่อน"
    exit 1
fi

# รันการประเมิน
$PYTHON_CMD evaluate_ragas.py

# ตรวจสอบว่ามีรายงานหรือไม่
if [ -f "ragas_evaluation_report.json" ]; then
    echo ""
    echo "✅ การประเมินเสร็จสิ้น!"
    echo "📊 ดูรายงานได้ที่: ragas_evaluation_report.json"
else
    echo ""
    echo "⚠️ ไม่พบรายงานการประเมิน"
fi

