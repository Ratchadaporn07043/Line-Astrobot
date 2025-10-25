import base64
from PIL import Image
import io
import os
from pymongo import MongoClient
from dotenv import load_dotenv

# โหลด .env
load_dotenv()

# ใช้ MONGO_URL จาก .env
MONGO_URL = os.getenv("MONGO_URL")

# Connect MongoDB
client = MongoClient(MONGO_URL)
db = client["astrobot"]
collection = db["image_chunks"]

# ดึงข้อมูลจาก document
data = collection.find_one({"semantic_topic": "Image 1"})  # หรือใช้ _id แทน

if data and "image_base64" in data:
    # แปลง base64 เป็นรูปภาพ
    image_data = base64.b64decode(data["image_base64"])
    image = Image.open(io.BytesIO(image_data))
    image.show()
else:
    print("❗ ไม่พบข้อมูลภาพใน MongoDB หรือ image_base64 หายไป")
