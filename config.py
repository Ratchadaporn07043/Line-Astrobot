# Configuration file for AstroBot
import os

# MongoDB Atlas Connection
# กรุณาใส่ MongoDB connection string ของคุณที่นี่
MONGO_URL = "mongodb+srv://your-username:your-password@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority"

# OpenAI API Key
# กรุณาใส่ OpenAI API key ของคุณที่นี่
OPENAI_API_KEY = "your-openai-api-key-here"

# Database Configuration
SUMMARY_DB_NAME = "astrobot_summary"  # สำหรับเก็บข้อมูลที่ summary และ summary embedding แล้ว
ORIGINAL_DB_NAME = "astrobot_original"  # สำหรับเก็บไฟล์ต้นฉบับที่ extract แล้ว

# Collection Names
# สำหรับข้อมูลต้นฉบับ (ORIGINAL_DB_NAME)
ORIGINAL_TEXT_COLLECTION = "original_text_chunks"
ORIGINAL_IMAGE_COLLECTION = "original_image_chunks"
ORIGINAL_TABLE_COLLECTION = "original_table_chunks"

# สำหรับข้อมูลที่ประมวลผลแล้ว (SUMMARY_DB_NAME)
PROCESSED_TEXT_COLLECTION = "processed_text_chunks"
PROCESSED_IMAGE_COLLECTION = "processed_image_chunks"
PROCESSED_TABLE_COLLECTION = "processed_table_chunks"

# PDF Paths
PDF_PATH = "data/attention.pdf"

