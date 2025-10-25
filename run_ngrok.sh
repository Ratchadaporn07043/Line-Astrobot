#!/bin/bash

# ไฟล์สำหรับรัน ngrok เพื่อ expose local server
# ใช้สำหรับ LINE Bot webhook

echo "🚀 กำลังเริ่มต้น ngrok..."

# ตรวจสอบว่า ngrok ติดตั้งแล้วหรือไม่
if ! command -v ngrok &> /dev/null; then
    echo "❌ ไม่พบ ngrok กรุณาติดตั้งก่อน"
    echo "📥 ดาวน์โหลดได้จาก: https://ngrok.com/download"
    echo "🔧 หรือติดตั้งผ่าน Homebrew: brew install ngrok"
    exit 1
fi

# ตรวจสอบว่า ngrok authenticate แล้วหรือไม่
if [ ! -f ~/.ngrok2/ngrok.yml ]; then
    echo "⚠️  กรุณา authenticate ngrok ก่อน"
    echo "🔑 รันคำสั่ง: ngrok authtoken YOUR_AUTH_TOKEN"
    echo "📝 ดึง token ได้จาก: https://dashboard.ngrok.com/get-started/your-authtoken"
    exit 1
fi

# รัน ngrok สำหรับ port 8000 (FastAPI default)
echo "🌐 กำลังสร้าง tunnel สำหรับ port 8000..."
echo "📱 ใช้ URL นี้สำหรับ LINE Bot webhook"
echo ""

# รัน ngrok และแสดง URL
ngrok http 8000

echo ""
echo "✅ ngrok ทำงานเสร็จแล้ว"
echo "📋 คัดลอก URL จากด้านบนไปใช้ใน LINE Bot webhook"
