# 🤖 AstroBot Setup Guide

คู่มือการติดตั้งและใช้งาน AstroBot LINE Bot

## 📋 Prerequisites

### 1. ติดตั้ง ngrok
```bash
# วิธีที่ 1: ดาวน์โหลดจากเว็บไซต์
# ไปที่ https://ngrok.com/download และดาวน์โหลด

# วิธีที่ 2: ติดตั้งผ่าน Homebrew (macOS)
brew install ngrok
```

### 2. Authenticate ngrok
```bash
# ดึง authtoken จาก https://dashboard.ngrok.com/get-started/your-authtoken
ngrok authtoken YOUR_AUTH_TOKEN
```

## 🚀 การใช้งาน

### วิธีที่ 1: รันแอปพลิเคชันและ ngrok พร้อมกัน (แนะนำ)
```bash
# ให้สิทธิ์การรันไฟล์
chmod +x start_app.sh

# รันแอปพลิเคชัน
./start_app.sh
```

### วิธีที่ 2: รันแยกกัน
```bash
# Terminal 1: รัน FastAPI server
python3 -m app.main

# Terminal 2: รัน ngrok
./run_ngrok.sh
```

## ⚙️ การตั้งค่า

### 1. สร้างไฟล์ .env
```bash
# ไฟล์จะถูกสร้างอัตโนมัติเมื่อรัน start_app.sh
# หรือสร้างเอง:

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
```

### 2. ตั้งค่า LINE Bot Webhook
1. เปิด LINE Developers Console
2. ไปที่ Messaging API settings
3. ใส่ Webhook URL: `https://your-ngrok-url.ngrok.io/callback`
4. เปิดใช้งาน "Use webhook"

## 📱 การทดสอบ

### 1. ทดสอบ FastAPI server
```bash
# ทดสอบ endpoint /ask
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "test_user", "question": "สวัสดี"}'
```

### 2. ทดสอบ ngrok tunnel
```bash
# เปิด http://localhost:4040 เพื่อดู ngrok dashboard
# หรือใช้ curl เพื่อดึง URL
curl -s http://localhost:4040/api/tunnels | grep -o '"public_url":"[^"]*"'
```

## 🛠️ การแก้ไขปัญหา

### ปัญหาที่พบบ่อย

1. **ngrok ไม่ทำงาน**
   ```bash
   # ตรวจสอบว่า authenticate แล้ว
   ngrok authtoken YOUR_TOKEN
   ```

2. **FastAPI ไม่เริ่มต้น**
   ```bash
   # ตรวจสอบ environment variables
   cat .env
   
   # ตรวจสอบ Python packages
   pip3 list | grep fastapi
   ```

3. **LINE Bot ไม่ตอบกลับ**
   - ตรวจสอบ Webhook URL ใน LINE Developers Console
   - ตรวจสอบ Channel Access Token และ Channel Secret
   - ดู logs ใน terminal

## 📁 โครงสร้างไฟล์

```
astrobot/
├── app/                    # โค้ดหลัก
│   ├── main.py            # FastAPI server
│   ├── response_message.py
│   ├── quick_reply.py
│   └── retrieval_utils.py
├── data/                   # ข้อมูล
├── requirements.txt        # Python packages
├── start_app.sh           # Script รันแอปพลิเคชัน
├── run_ngrok.sh           # Script รัน ngrok
└── .env                   # Environment variables
```

## 🔧 คำสั่งที่มีประโยชน์

```bash
# ดู ngrok logs
tail -f ngrok.log

# ดู Python logs
python3 -m app.main --log-level debug

# ตรวจสอบ port ที่ใช้งาน
lsof -i :8000
lsof -i :4040

# หยุด process ทั้งหมด
pkill -f "python3 -m app.main"
pkill -f "ngrok"
```

## 📞 การสนับสนุน

หากมีปัญหาหรือต้องการความช่วยเหลือ:
1. ตรวจสอบ logs ใน terminal
2. ดู error messages
3. ตรวจสอบการตั้งค่าในไฟล์ .env
4. ตรวจสอบ LINE Bot configuration
