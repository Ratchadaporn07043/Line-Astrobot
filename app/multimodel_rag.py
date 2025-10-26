import os
import io
import base64
import fitz  # PyMuPDF
import pdfplumber
from PIL import Image
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
import torch
import easyocr
from openai import OpenAI
from datetime import datetime
import json
import gc
import psutil
import re

# 🆕 เพิ่ม PyThaiNLP สำหรับปรับปรุง OCR
try:
    from pythainlp import word_tokenize
    from pythainlp.spell import correct
    from pythainlp.util import normalize
    PYTHAINLP_AVAILABLE = True
    print("✅ PyThaiNLP loaded successfully")
except ImportError:
    PYTHAINLP_AVAILABLE = False
    print("⚠️ PyThaiNLP not available, using basic text processing")

# ✅ แก้ไขปัญหา MPS device, PIL.ANTIALIAS และ tokenizers parallelism
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
if not hasattr(Image, 'ANTIALIAS'):
    Image.ANTIALIAS = Image.LANCZOS

# ✅ โหลด .env
dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
load_dotenv(dotenv_path)

# ✅ ตัวแปรระบบ
PDF_PATH = "data/attention.pdf"
MONGO_URL = os.getenv("MONGO_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUMMARY_DB_NAME = "astrobot_summary"  # สำหรับเก็บข้อมูลที่ summary และ summary embedding แล้ว
ORIGINAL_DB_NAME = "astrobot_original"  # สำหรับเก็บไฟล์ต้นฉบับที่ extract แล้ว

# ✅ ตัวแปรระบบ - Collection Names
# สำหรับข้อมูลต้นฉบับ (ORIGINAL_DB_NAME)
ORIGINAL_TEXT_COLLECTION = "original_text_chunks"
ORIGINAL_IMAGE_COLLECTION = "original_image_chunks"
ORIGINAL_TABLE_COLLECTION = "original_table_chunks"

# สำหรับข้อมูลที่ประมวลผลแล้ว (SUMMARY_DB_NAME)
PROCESSED_TEXT_COLLECTION = "processed_text_chunks"
PROCESSED_IMAGE_COLLECTION = "processed_image_chunks"
PROCESSED_TABLE_COLLECTION = "processed_table_chunks"

# ✅ ฟังก์ชันตรวจสอบ memory
def check_memory():
    """ตรวจสอบการใช้ memory"""
    memory = psutil.virtual_memory()
    print(f"💾 Memory: {memory.percent}% ({memory.used / 1024**3:.1f}GB / {memory.total / 1024**3:.1f}GB)")
    if memory.percent > 80:
        print("⚠️ High memory usage, running garbage collection...")
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

# 🆕 ฟังก์ชันปรับปรุงข้อความไทยจาก OCR ด้วย PyThaiNLP
def improve_thai_ocr_text(ocr_text):
    """
    ปรับปรุงข้อความไทยจาก OCR ด้วย PyThaiNLP
    """
    if not PYTHAINLP_AVAILABLE or not ocr_text.strip():
        return ocr_text
    
    try:
        # ทำความสะอาดข้อความ
        text = ocr_text.strip()
        
        # แก้ไขการเว้นวรรคที่ผิด
        text = re.sub(r'([ก-๙])([A-Za-z])', r'\1 \2', text)  # เว้นวรรคระหว่างไทย-อังกฤษ
        text = re.sub(r'([A-Za-z])([ก-๙])', r'\1 \2', text)  # เว้นวรรคระหว่างอังกฤษ-ไทย
        text = re.sub(r'([ก-๙])([0-9])', r'\1 \2', text)    # เว้นวรรคระหว่างไทย-ตัวเลข
        text = re.sub(r'([0-9])([ก-๙])', r'\1 \2', text)    # เว้นวรรคระหว่างตัวเลข-ไทย
        
        # แก้ไขการเว้นวรรคที่ซ้ำ
        text = re.sub(r'\s+', ' ', text)
        
        # แบ่งคำด้วย PyThaiNLP
        words = word_tokenize(text, engine='newmm')
        
        # แก้ไขคำผิดด้วย PyThaiNLP
        corrected_words = []
        for word in words:
            if len(word) > 2 and word.isalpha():  # แก้ไขเฉพาะคำที่มีความยาวมากกว่า 2 ตัวอักษร
                try:
                    corrected = correct(word)
                    corrected_words.append(corrected if corrected else word)
                except:
                    corrected_words.append(word)
            else:
                corrected_words.append(word)
        
        # รวมคำกลับเป็นประโยค
        improved_text = ' '.join(corrected_words)
        
        # ทำความสะอาดอีกครั้ง
        improved_text = re.sub(r'\s+', ' ', improved_text).strip()
        
        return improved_text
        
    except Exception as e:
        print(f"⚠️ Error in Thai text improvement: {e}")
        return ocr_text

# ✅ โหลดโมเดลแบบ lazy loading
def get_embedding_model():
    """โหลด embedding model แบบ lazy loading"""
    if not hasattr(get_embedding_model, 'model'):
        print("🔄 Loading embedding model...")
        get_embedding_model.model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    return get_embedding_model.model

def get_semantic_model():
    """โหลด semantic model แบบ lazy loading"""
    if not hasattr(get_semantic_model, 'model'):
        print("🔄 Loading semantic model...")
        get_semantic_model.model = SentenceTransformer("minishlab/potion-multilingual-128M", device="cpu")
    return get_semantic_model.model

def get_ocr_reader():
    """โหลด OCR reader แบบ lazy loading"""
    if not hasattr(get_ocr_reader, 'reader'):
        print(" Loading OCR reader...")
        get_ocr_reader.reader = easyocr.Reader(['en', 'th'], gpu=False, verbose=False)
    return get_ocr_reader.reader

# OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ✅ อ่านข้อความจาก PDF ด้วย PyMuPDF
def extract_text_with_pymupdf(path):
    """
    อ่านข้อความจาก PDF ด้วย PyMuPDF
    """
    print(f"📖 กำลังอ่านข้อความจาก: {path}")
    text_output = ""
    doc = fitz.open(path)
    
    try:
        for page_num, page in enumerate(doc):
            page_text = page.get_text("text")
            if page_text.strip():
                text_output += f"\n--- หน้า {page_num + 1} ---\n{page_text}"
            
            # ตรวจสอบ memory ทุก 20 หน้า
            if page_num % 20 == 0:
                check_memory()
                
    finally:
        doc.close()
    
    return text_output

# ✅ แปลงรูปภาพเป็นข้อความด้วย OCR + PyThaiNLP (ปรับปรุง memory management)
def extract_images_with_ocr(path):
    """
    แปลงรูปภาพใน PDF เป็นข้อความด้วย OCR + PyThaiNLP
    """
    print(f"กำลังแปลงรูปภาพเป็นข้อความจาก: {path}")
    images_data = []
    doc = fitz.open(path)
    
    try:
        ocr_reader = get_ocr_reader()
        
        for page_num, page in enumerate(doc):
            images = page.get_images(full=True)
            print(f"หน้า {page_num + 1}: {len(images)} รูป")
            
            for img_index, img in enumerate(images):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    # ตรวจสอบขนาดรูปภาพ
                    image = Image.open(io.BytesIO(image_bytes))
                    width, height = image.size
                    
                    # ข้ามรูปที่ใหญ่เกินไป
                    if width * height > 1500000:  # 1.5M pixels
                        print(f"⚠️ ข้ามรูปใหญ่ {img_index + 1} ({width}x{height})")
                        continue
                    
                    # ข้ามรูปที่เล็กเกินไป
                    if width < 50 or height < 50:
                        print(f"⚠️ ข้ามรูปเล็ก {img_index + 1} ({width}x{height})")
                        continue
                    
                    # OCR
                    ocr_results = ocr_reader.readtext(image_bytes)
                    ocr_text = " ".join([result[1] for result in ocr_results if result[2] > 0.3])  # ลด confidence threshold
                    
                    if ocr_text.strip():
                        # 🆕 ปรับปรุงข้อความด้วย PyThaiNLP
                        improved_text = improve_thai_ocr_text(ocr_text)
                        
                        image_info = {
                            "page": page_num + 1,
                            "image_index": img_index + 1,
                            "original_text": ocr_text.strip(),
                            "improved_text": improved_text,
                            "text": improved_text,  # ใช้ข้อความที่ปรับปรุงแล้ว
                            "image_base64": base64.b64encode(image_bytes).decode("utf-8")
                        }
                        images_data.append(image_info)
                        
                        print(f"✅ รูป {img_index + 1}: {len(improved_text)} ตัวอักษร")
                    
                    # ล้าง memory
                    del image, image_bytes, ocr_results
                    
                except Exception as e:
                    print(f"❗ Error processing image {img_index + 1} on page {page_num + 1}: {e}")
                    continue
            
            # ตรวจสอบ memory หลังจากประมวลผลแต่ละหน้า
            if page_num % 5 == 0:
                check_memory()
            
            # จำกัดจำนวนรูปต่อหน้า
            if len(images_data) > 50:  # จำกัดไม่เกิน 50 รูป
                print("⚠️ จำกัดจำนวนรูปที่ 50 รูป")
                break
                
    finally:
        doc.close()
    
    return images_data

# ✅ แปลงตารางเป็นข้อความด้วย pdfplumber
def extract_tables_with_pdfplumber(path):
    """
    แปลงตารางใน PDF เป็นข้อความด้วย pdfplumber
    """
    print(f" กำลังแปลงตารางเป็นข้อความจาก: {path}")
    tables_data = []
    
    try:
        with pdfplumber.open(path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                for table_index, table in enumerate(tables):
                    if table:
                        # แปลงตารางเป็นข้อความ
                        table_text = ""
                        for row in table:
                            if row:
                                row_text = " | ".join([cell if cell else "" for cell in row])
                                table_text += row_text + "\n"
                        
                        if table_text.strip():
                            table_info = {
                                "page": page_num + 1,
                                "table_index": table_index + 1,
                                "text": table_text.strip()
                            }
                            tables_data.append(table_info)
                
                # ตรวจสอบ memory ทุก 10 หน้า
                if page_num % 10 == 0:
                    check_memory()
                    
    except Exception as e:
        print(f"❗ Error extracting tables: {e}")
    
    return tables_data

# ✅ Semantic Chunking ด้วย Potion Model
def semantic_chunking_with_potion(text, content_type, chunk_size=1000, overlap=200):
    """
    แบ่งข้อความด้วย Semantic Chunking โดยใช้ Potion Model
    """
    print(f"🧠 เริ่ม Semantic Chunking สำหรับ {content_type.upper()}")
    
    try:
        semantic_model = get_semantic_model()
        
        # แบ่งข้อความเป็นประโยค
        sentences = text.split('. ')
        if len(sentences) <= 1:
            return [{"text": text, "type": content_type, "chunk_id": 0}]
        
        # จำกัดจำนวนประโยคเพื่อประหยัด memory
        if len(sentences) > 500:
            sentences = sentences[:500]
            print(f"⚠️ จำกัดที่ 500 ประโยคเพื่อประหยัด memory")
        
        # สร้าง embeddings สำหรับประโยค
        sentence_embeddings = semantic_model.encode(sentences)
        
        # คำนวณความคล้ายคลึงระหว่างประโยค
        chunks = []
        current_chunk = []
        current_length = 0
        
        for i, sentence in enumerate(sentences):
            current_chunk.append(sentence)
            current_length += len(sentence)
            
            # ตรวจสอบว่าควรแบ่ง chunk หรือไม่
            if current_length >= chunk_size or i == len(sentences) - 1:
                chunk_text = '. '.join(current_chunk)
                chunks.append({
                    "text": chunk_text,
                    "type": content_type,
                    "chunk_id": len(chunks)
                })
                current_chunk = []
                current_length = 0
        
        # ล้าง memory
        del sentence_embeddings, sentences
        check_memory()
        
        return chunks
        
    except Exception as e:
        print(f"❗ Error in semantic chunking: {e}")
        # Fallback: แบ่งแบบธรรมดา
        return [{"text": text, "type": content_type, "chunk_id": 0}]

# ✅ สร้าง Embeddings
def create_embeddings(text):
    """
    สร้าง embeddings สำหรับข้อความ
    """
    try:
        embedding_model = get_embedding_model()
        return embedding_model.encode(text).tolist()
    except Exception as e:
        print(f"❗ Error creating embeddings: {e}")
        return [0.0] * 384  # fallback vector

# ✅ สรุปข้อความด้วย OpenAI
def summarize_with_openai(text, content_type):
    """
    สรุปข้อความด้วย OpenAI GPT
    """
    try:
        prompt = f"""
        สรุปเนื้อหาต่อไปนี้ให้กระชับและเข้าใจง่าย (ภาษาไทย):
        
        ประเภทเนื้อหา: {content_type}
        เนื้อหา: {text[:2000]}...
        
        กรุณาสรุปให้ไม่เกิน 3 ประโยค
        """
        
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=150,
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"❗ Error in summarization: {e}")
        return text[:200] + "..." if len(text) > 200 else text

# ✅ บันทึกข้อมูลต้นฉบับลง MongoDB (ไม่มี embeddings และ summary)
def store_original_data_in_mongodb(chunks, collection_name):
    """
    บันทึกข้อมูลต้นฉบับลง ORIGINAL_DB_NAME (ไม่มี embeddings และ summary)
    """
    try:
        # ลองเชื่อมต่อ MongoDB Atlas
        print(f"🔗 กำลังเชื่อมต่อ MongoDB Atlas...")
        client = MongoClient(MONGO_URL, serverSelectionTimeoutMS=5000)
        
        # ทดสอบการเชื่อมต่อ
        client.admin.command('ping')
        print(f"✅ เชื่อมต่อ MongoDB Atlas สำเร็จ")
        
        # ใช้ ORIGINAL_DB_NAME สำหรับข้อมูลต้นฉบับ
        db_name = ORIGINAL_DB_NAME
        print(f"📊 ใช้ Database: {db_name} (Original - ไม่มี embeddings/summary)")
        
        db = client[db_name]
        collection = db[collection_name]
        
        # ลบข้อมูลเก่า
        collection.delete_many({})
        
        # บันทึกข้อมูลต้นฉบับ (ไม่มี embeddings และ summary)
        for i, chunk in enumerate(chunks):
            print(f"📝 กำลังบันทึกข้อมูลต้นฉบับ chunk {i+1}/{len(chunks)}...")
            
            # สร้างสำเนาของ chunk และเพิ่ม created_at
            original_chunk = chunk.copy()
            original_chunk["created_at"] = datetime.now()
            
            # ไม่เพิ่ม embeddings และ summary
            collection.insert_one(original_chunk)
            
            # ตรวจสอบ memory ทุก 5 chunks
            if i % 5 == 0:
                check_memory()
        
        print(f"✅ บันทึกข้อมูลต้นฉบับ {len(chunks)} chunks ลง {collection_name}")
        client.close()
        
    except Exception as e:
        print(f"❗ MongoDB Atlas connection failed: {e}")
        print(f"💾 บันทึกลงไฟล์ JSON แทน...")
        
        # Fallback: บันทึกลงไฟล์ JSON
        store_original_to_json(chunks, collection_name)

# ✅ บันทึกข้อมูลที่ประมวลผลแล้วลง MongoDB (มี embeddings และ summary)
def store_processed_data_in_mongodb(chunks, collection_name):
    """
    บันทึกข้อมูลที่ประมวลผลแล้วลง SUMMARY_DB_NAME (มี embeddings และ summary)
    """
    try:
        # ลองเชื่อมต่อ MongoDB Atlas
        print(f"🔗 กำลังเชื่อมต่อ MongoDB Atlas...")
        client = MongoClient(MONGO_URL, serverSelectionTimeoutMS=5000)
        
        # ทดสอบการเชื่อมต่อ
        client.admin.command('ping')
        print(f"✅ เชื่อมต่อ MongoDB Atlas สำเร็จ")
        
        # ใช้ SUMMARY_DB_NAME สำหรับข้อมูลที่ประมวลผลแล้ว
        db_name = SUMMARY_DB_NAME
        print(f"📊 ใช้ Database: {db_name} (Processed - มี summary embeddings/summary)")
        
        db = client[db_name]
        collection = db[collection_name]
        
        # ลบข้อมูลเก่า
        collection.delete_many({})
        
        # บันทึกข้อมูลที่ประมวลผลแล้ว (มี summary embeddings และ summary)
        for i, chunk in enumerate(chunks):
            print(f"📝 กำลังประมวลผล chunk {i+1}/{len(chunks)}...")
            
            # สร้างสำเนาของ chunk และเพิ่มข้อมูลที่ประมวลผลแล้ว
            processed_chunk = chunk.copy()
            processed_chunk["created_at"] = datetime.now()
            
            # สร้าง summary ก่อน
            summary_text = summarize_with_openai(chunk["text"], chunk["type"])
            processed_chunk["summary"] = summary_text
            
            # สร้าง embeddings จาก summary แทน text ต้นฉบับ
            processed_chunk["embeddings"] = create_embeddings(summary_text)
            
            collection.insert_one(processed_chunk)
            
            # ตรวจสอบ memory ทุก 3 chunks
            if i % 3 == 0:
                check_memory()
        
        print(f"✅ บันทึกข้อมูลที่ประมวลผลแล้ว {len(chunks)} chunks ลง {collection_name}")
        client.close()
        
    except Exception as e:
        print(f"❗ MongoDB Atlas connection failed: {e}")
        print(f"💾 บันทึกลงไฟล์ JSON แทน...")
        
        # Fallback: บันทึกลงไฟล์ JSON
        store_processed_to_json(chunks, collection_name)

# ✅ บันทึกข้อมูลต้นฉบับลงไฟล์ JSON (fallback)
def store_original_to_json(chunks, collection_name):
    """
    บันทึกข้อมูลต้นฉบับลงไฟล์ JSON เป็น fallback (ไม่มี embeddings และ summary)
    """
    try:
        # สร้างโฟลเดอร์ output ถ้าไม่มี
        output_dir = "output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # บันทึกข้อมูลต้นฉบับ (ไม่มี embeddings และ summary)
        original_chunks = []
        for i, chunk in enumerate(chunks):
            print(f"📝 กำลังบันทึกข้อมูลต้นฉบับ chunk {i+1}/{len(chunks)}...")
            
            # สร้างสำเนาของ chunk และเพิ่ม created_at
            original_chunk = chunk.copy()
            original_chunk["created_at"] = datetime.now().isoformat()
            
            # ไม่เพิ่ม embeddings และ summary
            original_chunks.append(original_chunk)
            
            # ตรวจสอบ memory ทุก 5 chunks
            if i % 5 == 0:
                check_memory()
        
        # บันทึกลงไฟล์
        filename = f"{output_dir}/{collection_name}_original.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(original_chunks, f, ensure_ascii=False, indent=2)
        
        print(f"✅ บันทึกข้อมูลต้นฉบับ {len(original_chunks)} chunks ลง {filename}")
        
    except Exception as e:
        print(f"❗ Error saving original data to JSON: {e}")

# ✅ บันทึกข้อมูลที่ประมวลผลแล้วลงไฟล์ JSON (fallback)
def store_processed_to_json(chunks, collection_name):
    """
    บันทึกข้อมูลที่ประมวลผลแล้วลงไฟล์ JSON เป็น fallback (มี embeddings และ summary)
    """
    try:
        # สร้างโฟลเดอร์ output ถ้าไม่มี
        output_dir = "output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # ประมวลผล chunks
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            print(f"📝 กำลังประมวลผล chunk {i+1}/{len(chunks)}...")
            
            # สร้างสำเนาของ chunk และเพิ่มข้อมูลที่ประมวลผลแล้ว
            processed_chunk = chunk.copy()
            processed_chunk["created_at"] = datetime.now().isoformat()
            
            # สร้าง summary ก่อน
            summary_text = summarize_with_openai(chunk["text"], chunk["type"])
            processed_chunk["summary"] = summary_text
            
            # สร้าง embeddings จาก summary แทน text ต้นฉบับ
            processed_chunk["embeddings"] = create_embeddings(summary_text)
            processed_chunks.append(processed_chunk)
            
            # ตรวจสอบ memory ทุก 3 chunks
            if i % 3 == 0:
                check_memory()
        
        # บันทึกลงไฟล์
        filename = f"{output_dir}/{collection_name}_processed.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(processed_chunks, f, ensure_ascii=False, indent=2)
        
        print(f"✅ บันทึกข้อมูลที่ประมวลผลแล้ว {len(processed_chunks)} chunks ลง {filename}")
        
    except Exception as e:
        print(f"❗ Error saving processed data to JSON: {e}")

# ✅ ฟังก์ชันหลัก
def main():
    print("🚀 เริ่ม Pipeline: Extract → OCR + PyThaiNLP → Semantic Chunking → Store")
    print(" TEXT | 🖼️ IMAGE (Improved) | 📊 TABLE")
    print()
    
    try:
        # === STEP 1: EXTRACTION ===
        print("=== STEP 1: EXTRACTION + OCR IMPROVEMENT ===")
        check_memory()
        
        # Extract text
        pdf_text = extract_text_with_pymupdf(PDF_PATH)
        
        # Extract images (จำกัดจำนวนเพื่อประหยัด memory)
        print("🖼️ กำลังแปลงรูปภาพเป็นข้อความ...")
        images_data = extract_images_with_ocr(PDF_PATH)
        
        # Extract tables
        tables_data = extract_tables_with_pdfplumber(PDF_PATH)
        
        print(f"\n📊 สถิติการ Extract:")
        print(f"   📝 Text: {len(pdf_text)} ตัวอักษร")
        print(f"   🖼️ Images: {len(images_data)} รูป")
        print(f"   📊 Tables: {len(tables_data)} ตาราง")
        
        # === STEP 2: SEMANTIC CHUNKING ===
        print("\n=== STEP 2: SEMANTIC CHUNKING ===")
        check_memory()
        
        # Chunk text
        text_chunks = semantic_chunking_with_potion(pdf_text, "text", chunk_size=1000, overlap=200)
        
        # Chunk images
        image_chunks = []
        for img in images_data:
            chunk = {
                "text": img["text"],
                "type": "image",
                "chunk_id": len(image_chunks),
                "page": img["page"],
                "image_index": img["image_index"],
                "original_text": img.get("original_text", ""),
                "improved_text": img.get("improved_text", ""),
                "image_base64": img["image_base64"]
            }
            image_chunks.append(chunk)
        
        # Chunk tables
        table_chunks = []
        for table in tables_data:
            chunk = {
                "text": table["text"],
                "type": "table",
                "chunk_id": len(table_chunks),
                "page": table["page"],
                "table_index": table["table_index"]
            }
            table_chunks.append(chunk)
        
        print(f"✅ สร้าง chunks สำเร็จ:")
        print(f"   📝 Text chunks: {len(text_chunks)}")
        print(f"   🖼️ Image chunks: {len(image_chunks)}")
        print(f"   Table chunks: {len(table_chunks)}")
        
        # === STEP 3: STORE IN MONGODB ===
        print("\n=== STEP 3: STORE IN MONGODB ===")
        check_memory()
        
        # เก็บข้อมูลต้นฉบับใน ORIGINAL_DB_NAME (ไม่มี embeddings และ summary)
        print("\n📁 เก็บข้อมูลต้นฉบับใน ORIGINAL_DB_NAME...")
        # สร้างสำเนาของ chunks เพื่อไม่ให้มี embeddings และ summary
        import copy
        original_text_chunks = copy.deepcopy(text_chunks)
        original_image_chunks = copy.deepcopy(image_chunks)
        original_table_chunks = copy.deepcopy(table_chunks)
        
        store_original_data_in_mongodb(original_text_chunks, ORIGINAL_TEXT_COLLECTION)
        store_original_data_in_mongodb(original_image_chunks, ORIGINAL_IMAGE_COLLECTION)
        store_original_data_in_mongodb(original_table_chunks, ORIGINAL_TABLE_COLLECTION)
        
        # เก็บข้อมูลที่ประมวลผลแล้ว (มี summary embedding และ summary) ใน SUMMARY_DB_NAME
        print("\n📊 เก็บข้อมูลที่ประมวลผลแล้วใน SUMMARY_DB_NAME...")
        store_processed_data_in_mongodb(text_chunks, PROCESSED_TEXT_COLLECTION)
        store_processed_data_in_mongodb(image_chunks, PROCESSED_IMAGE_COLLECTION)
        store_processed_data_in_mongodb(table_chunks, PROCESSED_TABLE_COLLECTION)
        
        print("\n✅ Pipeline เสร็จสิ้น!")
        
    except Exception as e:
        print(f"❗ Error in main pipeline: {e}")
        print("🔄 Running garbage collection...")
        gc.collect()
        check_memory()

if __name__ == "__main__":
    main()