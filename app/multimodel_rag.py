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

# ✅ ฟังก์ชันแปลง bbox เป็น format ที่ MongoDB สามารถ encode ได้
def convert_bbox_to_mongodb_format(bbox):
    """
    แปลง bbox (pymupdf.Rect, tuple, หรือ None) เป็น format ที่ MongoDB สามารถ encode ได้
    
    Args:
        bbox: pymupdf.Rect, tuple (x0, y0, x1, y1), หรือ None
        
    Returns:
        tuple หรือ None: (x0, y0, x1, y1) หรือ None
    """
    if bbox is None:
        return None
    
    try:
        # ถ้าเป็น pymupdf.Rect object
        if hasattr(bbox, 'x0') and hasattr(bbox, 'y0') and hasattr(bbox, 'x1') and hasattr(bbox, 'y1'):
            return (float(bbox.x0), float(bbox.y0), float(bbox.x1), float(bbox.y1))
        # ถ้าเป็น tuple หรือ list
        elif isinstance(bbox, (tuple, list)) and len(bbox) >= 4:
            return (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        else:
            return None
    except Exception as e:
        print(f"   ⚠️ Error converting bbox: {e}")
        return None

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
        get_embedding_model.model = SentenceTransformer("minishlab/potion-multilingual-128M", device="cpu")
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

# 🆕 โหลด Image Embedding Model แบบ lazy loading (ใช้โมเดลเดียวกับ text)
def get_image_embedding_model():
    """โหลด embedding model สำหรับสร้าง image embeddings แบบ lazy loading (ใช้โมเดลเดียวกับ text)"""
    if not hasattr(get_image_embedding_model, 'model'):
        try:
            print("🔄 Loading image embedding model...")
            # ใช้โมเดลเดียวกับ text embedding
            get_image_embedding_model.model = SentenceTransformer("minishlab/potion-multilingual-128M", device="cpu")
            print("✅ Image embedding model loaded successfully")
        except Exception as e:
            print(f"⚠️ Failed to load image embedding model: {e}")
            print("⚠️ Image embeddings will be disabled")
            get_image_embedding_model.model = None
    return get_image_embedding_model.model

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

# 🆕 สร้าง Image Embeddings (ใช้ text จาก OCR)
def create_image_embeddings(text):
    """
    สร้าง embeddings สำหรับรูปภาพโดยใช้ข้อความจาก OCR (ใช้โมเดลเดียวกับ text embedding)
    
    Args:
        text: ข้อความจาก OCR ของรูปภาพ
        
    Returns:
        list: image embedding vector หรือ None ถ้าไม่สามารถสร้างได้
    """
    try:
        image_model = get_image_embedding_model()
        if image_model is None:
            print("   ⚠️ Image embedding model not available, skipping...")
            return None
        
        # สร้าง embedding จากข้อความ OCR ด้วยโมเดลเดียวกับ text
        embedding = image_model.encode(text)
        return embedding.tolist()
        
    except Exception as e:
        print(f"   ⚠️ Error creating image embeddings: {e}")
        return None

# ✅ สรุปข้อความด้วย OpenAI
def summarize_with_openai(text, content_type, timeout=30, max_retries=3):
    """
    สรุปข้อความด้วย OpenAI GPT
    
    Args:
        text: ข้อความที่ต้องการสรุป
        content_type: ประเภทเนื้อหา (text/image/table)
        timeout: ระยะเวลารอสูงสุด (วินาที)
        max_retries: จำนวนครั้งที่ลองใหม่ถ้าเกิด error
    """
    for attempt in range(max_retries):
        try:
            prompt = f"""
            สรุปเนื้อหาต่อไปนี้ให้กระชับและเข้าใจง่าย (ภาษาไทย):
            
            ประเภทเนื้อหา: {content_type}
            เนื้อหา: {text[:2000]}...
            
            กรุณาสรุปให้ไม่เกิน 3 ประโยค
            """
            
            # เพิ่ม timeout และ error handling
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=150,
                temperature=0.7,
                timeout=timeout  # เพิ่ม timeout
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            error_msg = str(e)
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2  # Exponential backoff: 2s, 4s, 6s
                print(f"   ⚠️ Error in summarization (attempt {attempt + 1}/{max_retries}): {error_msg[:100]}")
                print(f"   ⏳ รอ {wait_time} วินาที แล้วลองใหม่...")
                import time
                time.sleep(wait_time)
            else:
                print(f"   ❗ Error in summarization after {max_retries} attempts: {error_msg[:100]}")
                # Fallback: ใช้ข้อความต้นฉบับที่ตัดแล้ว
                return text[:200] + "..." if len(text) > 200 else text
    
    # Fallback ถ้า retry ทั้งหมดล้มเหลว
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

# ✅ ฟังก์ชันประมวลผลหน้าเดียว (ตาม flow ที่ออกแบบ - เจออะไรก่อนทำอันนั้น)
def process_single_page(page_num, pymupdf_page, pdfplumber_pdf, ocr_reader, doc_id_counter):
    """
    ประมวลผลหน้าเดียว: Extract → Summary → Embedding → Store
    🆕 แก้ไขให้ทำงานตามลำดับที่เจอในหน้า (เจออะไรก่อนทำอันนั้นก่อน) - เรียงตาม y-coordinate
    
    Args:
        page_num: หมายเลขหน้าที่กำลังประมวลผล (0-based)
        pymupdf_page: หน้า PDF จาก PyMuPDF
        pdfplumber_pdf: PDF object จาก pdfplumber
        ocr_reader: OCR reader สำหรับประมวลผลรูปภาพ
        doc_id_counter: counter สำหรับสร้าง doc_id
        
    Returns:
        dict: {
            'has_content': bool,  # มีเนื้อหาหรือไม่ (สำหรับตรวจสอบหน้าเปล่า)
            'text_chunks': list,
            'image_chunks': list,
            'table_chunks': list,
            'text_processed_chunks': list,
            'image_processed_chunks': list,
            'table_processed_chunks': list
        }
    """
    page_results = {
        'has_content': False,
        'text_chunks': [],
        'image_chunks': [],
        'table_chunks': [],
        'text_processed_chunks': [],
        'image_processed_chunks': [],
        'table_processed_chunks': []
    }
    
    try:
        print(f"\n{'='*50}")
        print(f"📄 กำลังประมวลผลหน้า {page_num + 1} (ตามลำดับที่เจอ)")
        print(f"{'='*50}")
        
        # === STEP 1: รวบรวม elements ทั้งหมดพร้อมตำแหน่ง ===
        elements = []  # เก็บ elements ทั้งหมดพร้อมตำแหน่ง y-coordinate
        
        # 1.1 ดึง Text Blocks พร้อมตำแหน่ง
        text_blocks = pymupdf_page.get_text("blocks")  # Returns: [(x0, y0, x1, y1, text, block_no, block_type), ...]
        for block in text_blocks:
            if block[6] == 0:  # block_type = 0 คือ text block
                x0, y0, x1, y1, text, block_no, block_type = block
                if text.strip():
                    elements.append({
                        'type': 'text',
                        'y_pos': y0,  # ใช้ y0 (ตำแหน่งบนสุด) สำหรับเรียงลำดับ
                        'data': {
                            'text': text.strip(),
                            'bbox': (x0, y0, x1, y1),
                            'block_no': block_no
                        }
                    })
        
        # 1.2 ดึง Images พร้อมตำแหน่ง
        images = pymupdf_page.get_images(full=True)
        if images:
            print(f"   🖼️ พบ {len(images)} รูปภาพในหน้านี้")
        
        for img_index, img in enumerate(images):
            xref = img[0]
            try:
                # พยายามหา bbox ของรูปภาพจาก get_image_rects
                y_pos = 0  # ค่าเริ่มต้น
                bbox = None
                try:
                    from pymupdf.utils import get_image_rects
                    image_rects = get_image_rects(pymupdf_page, xref)
                    if image_rects:
                        bbox = image_rects[0]  # ใช้ rect แรก
                        if hasattr(bbox, 'y0'):
                            y_pos = bbox.y0
                        elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                            y_pos = bbox[1]  # y0
                except Exception as rect_error:
                    # ถ้าไม่สามารถดึงตำแหน่งได้ ให้ประมาณจาก image list position
                    # (รูปแรกจะอยู่ตำแหน่งบนสุดกว่า)
                    y_pos = img_index * 100  # ประมาณตำแหน่ง
                
                elements.append({
                    'type': 'image',
                    'y_pos': y_pos,
                    'data': {
                        'xref': xref,
                        'image_index': img_index,
                        'bbox': bbox
                    }
                })
            except Exception as e:
                print(f"⚠️ ไม่สามารถดึงตำแหน่งรูป {img_index + 1} ได้: {e}")
                # ถ้าไม่สามารถดึงตำแหน่งได้ ให้ใส่ตำแหน่ง 0 (จะอยู่แรกสุด)
                elements.append({
                    'type': 'image',
                    'y_pos': img_index * 100,  # ประมาณตำแหน่ง
                    'data': {
                        'xref': xref,
                        'image_index': img_index,
                        'bbox': None
                    }
                })
        
        # 1.3 ดึง Tables พร้อมตำแหน่ง (จาก pdfplumber)
        if page_num < len(pdfplumber_pdf.pages):
            pdfplumber_page = pdfplumber_pdf.pages[page_num]
            
            # พยายามหา bbox ของตาราง
            try:
                # ใช้ find_tables เพื่อได้ bbox (ถ้ามี)
                if hasattr(pdfplumber_page, 'find_tables'):
                    table_objects = pdfplumber_page.find_tables()
                    
                    for table_index, table_obj in enumerate(table_objects):
                        if table_obj and hasattr(table_obj, 'bbox'):
                            bbox = table_obj.bbox
                            y_pos = bbox[1] if isinstance(bbox, (list, tuple)) else getattr(bbox, 'y0', bbox[1])
                            
                            # แปลงตารางเป็นข้อความ
                            table = table_obj.extract() if hasattr(table_obj, 'extract') else None
                            table_text = ""
                            if table:
                                for row in table:
                                    if row:
                                        row_text = " | ".join([cell if cell else "" for cell in row])
                                        table_text += row_text + "\n"
                            
                            if table_text.strip():
                                elements.append({
                                    'type': 'table',
                                    'y_pos': y_pos,
                                    'data': {
                                        'table_index': table_index,
                                        'text': table_text.strip(),
                                        'bbox': bbox
                                    }
                                })
                else:
                    raise AttributeError("find_tables not available")
            except Exception as e:
                # Fallback: ถ้า find_tables ไม่ได้ ให้ใช้ extract_tables แบบเดิม
                print(f"⚠️ ไม่สามารถใช้ find_tables ได้: {e}, ใช้ extract_tables แทน")
                tables = pdfplumber_page.extract_tables()
                
                # ประมาณตำแหน่งตารางจากตำแหน่งของ text และ image elements ที่มีอยู่
                existing_y_positions = [e['y_pos'] for e in elements]
                base_y_pos = max(existing_y_positions) if existing_y_positions else 500  # เริ่มที่ 500 ถ้าไม่มี elements อื่น
                
                for table_index, table in enumerate(tables):
                    if table:
                        table_text = ""
                        for row in table:
                            if row:
                                row_text = " | ".join([cell if cell else "" for cell in row])
                                table_text += row_text + "\n"
                        
                        if table_text.strip():
                            # ประมาณตำแหน่งตาราง (ถัดจาก elements อื่นๆ)
                            table_y_pos = base_y_pos + (table_index * 150)
                            elements.append({
                                'type': 'table',
                                'y_pos': table_y_pos,
                                'data': {
                                    'table_index': table_index,
                                    'text': table_text.strip(),
                                    'bbox': None
                                }
                            })
        
        # === STEP 2: เรียงลำดับ elements ตาม y-coordinate (จากบนลงล่าง) ===
        elements.sort(key=lambda x: x['y_pos'])
        
        print(f"📊 พบ {len(elements)} elements: {len([e for e in elements if e['type']=='text'])} text, "
              f"{len([e for e in elements if e['type']=='image'])} images, "
              f"{len([e for e in elements if e['type']=='table'])} tables")
        
        # === STEP 3: ประมวลผลตามลำดับที่เรียงแล้ว (เจออะไรก่อนทำอันนั้นก่อน) ===
        text_chunk_counter = 0
        image_chunk_counter = 0
        table_chunk_counter = 0
        
        for element_index, element in enumerate(elements):
            element_type = element['type']
            data = element['data']
            
            print(f"\n📌 Element {element_index + 1}/{len(elements)}: {element_type.upper()} "
                  f"(y={element['y_pos']:.1f})")
            
            if element_type == 'text':
                # ประมวลผล Text Block
                page_results['has_content'] = True
                text_content = data['text']
                print(f"   📝 Text: {len(text_content)} ตัวอักษร")
                
                text_chunk = {
                    "text": text_content,
                    "type": "text",
                    "chunk_id": text_chunk_counter,
                    "page": page_num + 1,
                    "doc_id": f"doc_{doc_id_counter}_{page_num + 1}_text_{text_chunk_counter}",
                    "bbox": convert_bbox_to_mongodb_format(data['bbox'])
                }
                # ✅ Original chunk: ไม่มี embeddings (เก็บต้นฉบับเท่านั้น)
                page_results['text_chunks'].append(text_chunk)
                text_chunk_counter += 1
                
                # สร้าง summary และ embeddings สำหรับ processed chunk
                print(f"   🔄 กำลังสร้าง summary...")
                summary_text = summarize_with_openai(text_chunk["text"], "text")
                print(f"   🔄 กำลังสร้าง embeddings...")
                text_processed_chunk = text_chunk.copy()
                text_processed_chunk["summary"] = summary_text
                text_processed_chunk["embeddings"] = create_embeddings(summary_text)  # embeddings จาก summary
                text_processed_chunk["created_at"] = datetime.now()
                page_results['text_processed_chunks'].append(text_processed_chunk)
                print(f"   ✅ สร้าง summary และ embeddings แล้ว")
            
            elif element_type == 'image':
                # ประมวลผล Image
                xref = data['xref']
                img_index = data['image_index']
                
                try:
                    print(f"   🖼️ กำลังประมวลผลรูปภาพ {img_index + 1}...")
                    base_image = pymupdf_page.parent.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    # ตรวจสอบขนาดรูปภาพ
                    image = Image.open(io.BytesIO(image_bytes))
                    width, height = image.size
                    print(f"   📏 ขนาดรูปภาพ: {width}x{height} pixels")
                    
                    # ข้ามรูปที่ใหญ่เกินไป
                    if width * height > 1500000:
                        print(f"   ⚠️ ข้ามรูปใหญ่ ({width}x{height}, {width*height:,} pixels > 1,500,000)")
                        del image, image_bytes
                        continue
                    
                    # ข้ามรูปที่เล็กเกินไป
                    if width < 50 or height < 50:
                        print(f"   ⚠️ ข้ามรูปเล็ก ({width}x{height} < 50x50)")
                        del image, image_bytes
                        continue
                    
                    # OCR
                    print(f"   🔍 กำลังทำ OCR...")
                    ocr_results = ocr_reader.readtext(image_bytes)
                    ocr_text = " ".join([result[1] for result in ocr_results if result[2] > 0.3])
                    
                    if ocr_text.strip():
                        page_results['has_content'] = True
                        # ปรับปรุงข้อความด้วย PyThaiNLP
                        improved_text = improve_thai_ocr_text(ocr_text)
                        
                        print(f"   🖼️ Image {img_index + 1}: {len(improved_text)} ตัวอักษร (OCR: {len(ocr_text)} ตัวอักษร)")
                        
                        # 🆕 สร้าง image embedding จากข้อความ OCR (ใช้โมเดลเดียวกับ text)
                        print(f"   🔄 กำลังสร้าง image embedding...")
                        image_embedding = create_image_embeddings(improved_text)
                        
                        # Create image chunk
                        image_chunk = {
                            "text": improved_text,
                            "type": "image",
                            "chunk_id": image_chunk_counter,
                            "page": page_num + 1,
                            "image_index": img_index + 1,
                            "original_text": ocr_text.strip(),
                            "improved_text": improved_text,
                            "image_base64": base64.b64encode(image_bytes).decode("utf-8"),
                            "doc_id": f"doc_{doc_id_counter}_{page_num + 1}_img_{img_index + 1}",
                            "bbox": convert_bbox_to_mongodb_format(data['bbox'])
                        }
                        # ✅ Original chunk: ไม่มี embeddings (เก็บต้นฉบับเท่านั้น)
                        page_results['image_chunks'].append(image_chunk)
                        image_chunk_counter += 1
                        
                        # สร้าง summary และ embeddings สำหรับ processed chunk
                        print(f"   🔄 กำลังสร้าง summary...")
                        summary_text = summarize_with_openai(image_chunk["text"], "image")
                        print(f"   🔄 กำลังสร้าง embeddings...")
                        text_embedding = create_embeddings(summary_text)
                        
                        # สร้าง image embedding สำหรับ processed chunk
                        if image_embedding is not None:
                            print(f"   ✅ สร้าง image embedding สำเร็จ ({len(image_embedding)} dimensions)")
                        
                        image_processed_chunk = image_chunk.copy()
                        image_processed_chunk["summary"] = summary_text
                        image_processed_chunk["embeddings"] = text_embedding  # text embedding จาก summary
                        image_processed_chunk["created_at"] = datetime.now()
                        
                        # เพิ่ม image embedding ใน processed chunk ด้วย
                        if image_embedding is not None:
                            image_processed_chunk["image_embeddings"] = image_embedding
                        
                        page_results['image_processed_chunks'].append(image_processed_chunk)
                        print(f"   ✅ สร้าง summary, text embeddings และ image embeddings แล้ว")
                    else:
                        print(f"   ⚠️ ไม่พบข้อความในรูปภาพ {img_index + 1} (OCR ไม่เจอข้อความ) - ข้าม")
                    
                    # ล้าง memory
                    del image, image_bytes, ocr_results
                    
                except Exception as e:
                    print(f"   ❗ Error processing image {img_index + 1}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            elif element_type == 'table':
                # ประมวลผล Table
                table_text = data['text']
                table_index = data['table_index']
                
                if table_text.strip():
                    page_results['has_content'] = True
                    print(f"   📊 Table {table_index + 1}: {len(table_text)} ตัวอักษร")
                    
                    # Create table chunk
                    table_chunk = {
                        "text": table_text,
                        "type": "table",
                        "chunk_id": table_chunk_counter,
                        "page": page_num + 1,
                        "table_index": table_index + 1,
                        "doc_id": f"doc_{doc_id_counter}_{page_num + 1}_table_{table_index + 1}",
                        "bbox": convert_bbox_to_mongodb_format(data['bbox'])
                    }
                    # ✅ Original chunk: ไม่มี embeddings (เก็บต้นฉบับเท่านั้น)
                    page_results['table_chunks'].append(table_chunk)
                    table_chunk_counter += 1
                    
                    # สร้าง summary และ embeddings สำหรับ processed chunk
                    print(f"   🔄 กำลังสร้าง summary...")
                    summary_text = summarize_with_openai(table_chunk["text"], "table")
                    print(f"   🔄 กำลังสร้าง embeddings...")
                    table_processed_chunk = table_chunk.copy()
                    table_processed_chunk["summary"] = summary_text
                    table_processed_chunk["embeddings"] = create_embeddings(summary_text)  # embeddings จาก summary
                    table_processed_chunk["created_at"] = datetime.now()
                    page_results['table_processed_chunks'].append(table_processed_chunk)
                    print(f"   ✅ สร้าง summary และ embeddings แล้ว")
        
        # สรุปผลการประมวลผลหน้า
        if not page_results['has_content']:
            print(f"⚠️ หน้า {page_num + 1} เป็นหน้าเปล่า (ไม่มี text, images, หรือ tables)")
        else:
            total_chunks = (len(page_results['text_chunks']) + 
                          len(page_results['image_chunks']) + 
                          len(page_results['table_chunks']))
            print(f"\n✅ ประมวลผลหน้า {page_num + 1} เสร็จ: {total_chunks} chunks")
            print(f"   📝 Text: {len(page_results['text_chunks'])} chunks")
            print(f"   🖼️ Image: {len(page_results['image_chunks'])} chunks")
            print(f"   📊 Table: {len(page_results['table_chunks'])} chunks")
        
        return page_results
        
    except Exception as e:
        print(f"❗ Error processing page {page_num + 1}: {e}")
        import traceback
        traceback.print_exc()
        return page_results

# ✅ ฟังก์ชันช่วยบันทึกข้อมูลทีละหน้า
def store_page_results_to_mongodb(page_results, client, is_first_page=False):
    """
    บันทึกผลลัพธ์จากหนึ่งหน้าลง MongoDB ทันที
    
    Args:
        page_results: ผลลัพธ์จาก process_single_page()
        client: MongoDB client (เปิดไว้แล้ว)
        is_first_page: เป็นหน้าแรกหรือไม่ (ถ้าใช่จะลบข้อมูลเก่าก่อน)
    """
    try:
        # เตรียม databases และ collections
        db_original = client[ORIGINAL_DB_NAME]
        db_summary = client[SUMMARY_DB_NAME]
        
        orig_text_col = db_original[ORIGINAL_TEXT_COLLECTION]
        orig_image_col = db_original[ORIGINAL_IMAGE_COLLECTION]
        orig_table_col = db_original[ORIGINAL_TABLE_COLLECTION]
        
        proc_text_col = db_summary[PROCESSED_TEXT_COLLECTION]
        proc_image_col = db_summary[PROCESSED_IMAGE_COLLECTION]
        proc_table_col = db_summary[PROCESSED_TABLE_COLLECTION]
        
        # ลบข้อมูลเก่าครั้งเดียวตอนหน้าแรก
        if is_first_page:
            print("🗑️ ลบข้อมูลเก่าใน MongoDB...")
            orig_text_col.delete_many({})
            orig_image_col.delete_many({})
            orig_table_col.delete_many({})
            proc_text_col.delete_many({})
            proc_image_col.delete_many({})
            proc_table_col.delete_many({})
            print("✅ ลบข้อมูลเก่าเสร็จสิ้น")
        
        # เพิ่ม created_at ให้ทุก chunk
        now = datetime.now()
        
        # บันทึก Original Data
        if page_results['text_chunks']:
            for chunk in page_results['text_chunks']:
                chunk['created_at'] = now
            orig_text_col.insert_many(page_results['text_chunks'])
            print(f"   ✅ บันทึก {len(page_results['text_chunks'])} text chunks (original)")
        
        if page_results['image_chunks']:
            for chunk in page_results['image_chunks']:
                chunk['created_at'] = now
            orig_image_col.insert_many(page_results['image_chunks'])
            print(f"   ✅ บันทึก {len(page_results['image_chunks'])} image chunks (original)")
        
        if page_results['table_chunks']:
            for chunk in page_results['table_chunks']:
                chunk['created_at'] = now
            orig_table_col.insert_many(page_results['table_chunks'])
            print(f"   ✅ บันทึก {len(page_results['table_chunks'])} table chunks (original)")
        
        # บันทึก Processed Data (มี summary และ embeddings แล้ว)
        if page_results['text_processed_chunks']:
            for chunk in page_results['text_processed_chunks']:
                if 'created_at' not in chunk:
                    chunk['created_at'] = now
            proc_text_col.insert_many(page_results['text_processed_chunks'])
            print(f"   ✅ บันทึก {len(page_results['text_processed_chunks'])} text chunks (processed)")
        
        if page_results['image_processed_chunks']:
            for chunk in page_results['image_processed_chunks']:
                if 'created_at' not in chunk:
                    chunk['created_at'] = now
            proc_image_col.insert_many(page_results['image_processed_chunks'])
            print(f"   ✅ บันทึก {len(page_results['image_processed_chunks'])} image chunks (processed)")
        
        if page_results['table_processed_chunks']:
            for chunk in page_results['table_processed_chunks']:
                if 'created_at' not in chunk:
                    chunk['created_at'] = now
            proc_table_col.insert_many(page_results['table_processed_chunks'])
            print(f"   ✅ บันทึก {len(page_results['table_processed_chunks'])} table chunks (processed)")
        
        return True
        
    except Exception as e:
        print(f"❗ Error storing page results to MongoDB: {e}")
        import traceback
        traceback.print_exc()
        return False

# ✅ ฟังก์ชันหลัก (ประมวลผลหนึ่งหน้า → บันทึก → loop ต่อ)
def main():
    print("🚀 เริ่ม Pipeline: Extract → OCR + PyThaiNLP → Summary → Embedding → Store")
    print("📄 ประมวลผลหนึ่งหน้า → บันทึก MongoDB → loop ต่อ")
    print()
    
    client = None
    pymupdf_doc = None
    pdfplumber_pdf = None
    
    try:
        # === INITIALIZATION ===
        print("=== INITIALIZATION ===")
        check_memory()
        
        # เปิดไฟล์ PDF ทั้ง PyMuPDF และ pdfplumber
        pymupdf_doc = fitz.open(PDF_PATH)
        pdfplumber_pdf = pdfplumber.open(PDF_PATH)
        ocr_reader = get_ocr_reader()
        
        total_pages = len(pymupdf_doc)
        print(f"📚 จำนวนหน้าทั้งหมด: {total_pages} หน้า")
        
        # เปิด MongoDB connection ครั้งเดียว (ใช้ตลอดทั้ง pipeline)
        print(f"🔗 กำลังเชื่อมต่อ MongoDB Atlas...")
        client = MongoClient(MONGO_URL, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        print(f"✅ เชื่อมต่อ MongoDB Atlas สำเร็จ")
        
        # ตัวแปรสำหรับนับจำนวน chunks ทั้งหมด
        total_text_chunks = 0
        total_image_chunks = 0
        total_table_chunks = 0
        total_text_processed = 0
        total_image_processed = 0
        total_table_processed = 0
        
        doc_id_counter = 1  # สำหรับสร้าง doc_id
        
        # === LOOP: More Pages (ประมวลผลและบันทึกทีละหน้า) ===
        print("\n=== STEP 1: PAGE-BY-PAGE PROCESSING & STORING ===")
        for page_num in range(total_pages):
            print(f"\n{'='*60}")
            print(f"📄 กำลังประมวลผลหน้า {page_num + 1}/{total_pages}")
            print(f"{'='*60}")
            
            # ประมวลผลหน้าเดียว (Extract → Summary → Embedding)
            page_results = process_single_page(
                page_num=page_num,
                pymupdf_page=pymupdf_doc[page_num],
                pdfplumber_pdf=pdfplumber_pdf,
                ocr_reader=ocr_reader,
                doc_id_counter=doc_id_counter
            )
            
            # บันทึกลง MongoDB ทันที (หน้าแรกจะลบข้อมูลเก่าก่อน)
            is_first_page = (page_num == 0)
            print(f"\n💾 บันทึกผลลัพธ์จากหน้า {page_num + 1} ลง MongoDB...")
            
            success = store_page_results_to_mongodb(page_results, client, is_first_page=is_first_page)
            
            if success:
                # นับจำนวน chunks
                total_text_chunks += len(page_results['text_chunks'])
                total_image_chunks += len(page_results['image_chunks'])
                total_table_chunks += len(page_results['table_chunks'])
                total_text_processed += len(page_results['text_processed_chunks'])
                total_image_processed += len(page_results['image_processed_chunks'])
                total_table_processed += len(page_results['table_processed_chunks'])
                
                print(f"✅ บันทึกหน้า {page_num + 1} เสร็จสิ้น")
            else:
                print(f"⚠️ มีปัญหาในการบันทึกหน้า {page_num + 1} แต่จะดำเนินการต่อ...")
            
            # ตรวจสอบ memory ทุก 5 หน้า
            if (page_num + 1) % 5 == 0:
                check_memory()
            
            # ตรวจสอบว่ามีหน้าอื่นอีกไหม (More Pages Decision)
            if page_num < total_pages - 1:
                print(f"➡️ มีหน้าอื่นอีก {total_pages - page_num - 1} หน้า")
            else:
                print(f"✅ ประมวลผลและบันทึกครบทุกหน้าแล้ว ({total_pages} หน้า)")
        
        # ปิดไฟล์ PDF
        pymupdf_doc.close()
        pdfplumber_pdf.close()
        pymupdf_doc = None
        pdfplumber_pdf = None
        
        # === สรุปผลการประมวลผล ===
        print("\n" + "="*60)
        print("📊 สรุปผลการประมวลผลทั้งหมด")
        print("="*60)
        print(f"   📝 Text chunks (original): {total_text_chunks}")
        print(f"   🖼️ Image chunks (original): {total_image_chunks}")
        print(f"   📊 Table chunks (original): {total_table_chunks}")
        print(f"   📝 Text chunks (processed): {total_text_processed}")
        print(f"   🖼️ Image chunks (processed): {total_image_processed}")
        print(f"   📊 Table chunks (processed): {total_table_processed}")
        print(f"   📊 Total processed chunks: {total_text_processed + total_image_processed + total_table_processed}")
        
        print("\n✅ Pipeline เสร็จสิ้น!")
        print(f"✅ ข้อมูลทั้งหมดถูกบันทึกใน MongoDB:")
        print(f"   - Original: {ORIGINAL_DB_NAME}")
        print(f"   - Summary: {SUMMARY_DB_NAME}")
        
    except Exception as e:
        print(f"❗ Error in main pipeline: {e}")
        import traceback
        traceback.print_exc()
        print("🔄 Running garbage collection...")
        gc.collect()
        check_memory()
        
        # แสดงข้อมูลที่บันทึกไปแล้ว (ถ้ามี)
        if client:
            try:
                db_original = client[ORIGINAL_DB_NAME]
                db_summary = client[SUMMARY_DB_NAME]
                
                orig_text_count = db_original[ORIGINAL_TEXT_COLLECTION].count_documents({})
                proc_text_count = db_summary[PROCESSED_TEXT_COLLECTION].count_documents({})
                
                print(f"\n⚠️ ข้อมูลที่บันทึกไปแล้ว:")
                print(f"   - Original text chunks: {orig_text_count}")
                print(f"   - Processed text chunks: {proc_text_count}")
            except:
                pass
        
    finally:
        # ปิด MongoDB connection
        if client:
            try:
                client.close()
                print("🔌 ปิด MongoDB connection")
            except:
                pass
        
        # ปิดไฟล์ PDF (ถ้ายังไม่ได้ปิด)
        if pymupdf_doc:
            try:
                pymupdf_doc.close()
            except:
                pass
        if pdfplumber_pdf:
            try:
                pdfplumber_pdf.close()
            except:
                pass

if __name__ == "__main__":
    main()