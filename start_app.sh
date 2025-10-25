#!/bin/bash

# ไฟล์สำหรับรันแอปพลิเคชันและ ngrok พร้อมกัน

echo "🤖 กำลังเริ่มต้น AstroBot..."

# ตรวจสอบว่า Python ติดตั้งแล้วหรือไม่
if ! command -v python3 &> /dev/null; then
    echo "❌ ไม่พบ Python3 กรุณาติดตั้งก่อน"
    exit 1
fi

# ตรวจสอบว่า ngrok ติดตั้งแล้วหรือไม่
if ! command -v ngrok &> /dev/null; then
    echo "❌ ไม่พบ ngrok กรุณาติดตั้งก่อน"
    echo "📥 ดาวน์โหลดได้จาก: https://ngrok.com/download"
    echo "🔧 หรือติดตั้งผ่าน Homebrew: brew install ngrok"
    exit 1
fi

# ตรวจสอบไฟล์ .env
if [ ! -f .env ]; then
    echo "⚠️  ไม่พบไฟล์ .env"
    echo "📝 สร้างไฟล์ .env และใส่ environment variables ที่จำเป็น"
    echo "🔑 ตัวอย่าง:"
    echo "LINE_CHANNEL_ACCESS_TOKEN=your_access_token"
    echo "LINE_CHANNEL_SECRET=your_channel_secret"
    echo "OPENAI_API_KEY=your_openai_api_key"
    echo ""
    echo "สร้างไฟล์ .env ต่อ? (y/n)"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        echo "สร้างไฟล์ .env..."
        cat > .env << EOF
# LINE Bot Configuration
LINE_CHANNEL_ACCESS_TOKEN=your_access_token_here
LINE_CHANNEL_SECRET=your_channel_secret_here

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# MongoDB Configuration (ถ้าใช้)
MONGODB_URI=your_mongodb_uri_here

# Other Configuration
ENVIRONMENT=development
EOF
        echo "✅ สร้างไฟล์ .env เรียบร้อยแล้ว"
        echo "📝 กรุณาแก้ไขค่าในไฟล์ .env ให้ถูกต้อง"
    else
        echo "❌ ไม่สามารถดำเนินการต่อได้โดยไม่มีไฟล์ .env"
        exit 1
    fi
fi

echo "🚀 กำลังเริ่มต้น FastAPI server..."
echo "🌐 กำลังเริ่มต้น ngrok tunnel..."
echo ""

# รัน FastAPI server ใน background
echo "📡 เริ่มต้น FastAPI server..."
python3 run_app.py &
FASTAPI_PID=$!

# รอให้ server เริ่มต้น
sleep 3

# รัน ngrok ใน background
echo "🌐 เริ่มต้น ngrok tunnel..."
ngrok http 8000 > ngrok.log 2>&1 &
NGROK_PID=$!

# รอให้ ngrok เริ่มต้น
sleep 5

# แสดง URL ของ ngrok
echo ""
echo "🔗 Ngrok URL:"
echo "================================"
if command -v curl &> /dev/null; then
    NGROK_URL=$(curl -s http://localhost:4040/api/tunnels | grep -o '"public_url":"[^"]*"' | cut -d'"' -f4)
    if [ ! -z "$NGROK_URL" ]; then
        echo "🌐 Public URL: $NGROK_URL"
        echo "📱 Webhook URL: $NGROK_URL/callback"
        echo ""
        echo "📋 คัดลอก URL ด้านบนไปใช้ใน LINE Bot webhook"
    else
        echo "❌ ไม่สามารถดึง URL ได้"
    fi
else
    echo "📱 เปิด http://localhost:4040 เพื่อดู ngrok URL"
fi
echo "================================"

echo ""
echo "✅ แอปพลิเคชันทำงานแล้ว!"
echo "🛑 กด Ctrl+C เพื่อหยุด"

# ฟังก์ชัน cleanup เมื่อหยุด
cleanup() {
    echo ""
    echo "🛑 กำลังหยุดแอปพลิเคชัน..."
    kill $FASTAPI_PID 2>/dev/null
    kill $NGROK_PID 2>/dev/null
    rm -f ngrok.log
    echo "✅ หยุดเรียบร้อยแล้ว"
    exit 0
}

# จับ signal เพื่อ cleanup
trap cleanup SIGINT SIGTERM

# รอให้ผู้ใช้กด Ctrl+C
wait
