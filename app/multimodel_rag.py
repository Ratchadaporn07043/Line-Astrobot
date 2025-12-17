import os
import io
import base64
import tempfile
import fitz  # PyMuPDF
import pdfplumber
from PIL import Image
from dotenv import load_dotenv
from pymongo import MongoClient
from datetime import datetime
import json
import gc
import psutil
import re
from sentence_transformers import SentenceTransformer

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
ORIGINAL_DB_NAME = "astrobot_original"  # สำหรับเก็บไฟล์ต้นฉบับที่ extract แล้ว

# ✅ ตัวแปรระบบ - Collection Names
# สำหรับข้อมูลต้นฉบับ (ORIGINAL_DB_NAME)
ORIGINAL_TEXT_COLLECTION = "original_text_chunks"
ORIGINAL_IMAGE_COLLECTION = "original_image_chunks"
ORIGINAL_TABLE_COLLECTION = "original_table_chunks"

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

def get_ocr_reader():
    """โหลด OCR reader แบบ lazy loading (ใช้ Typhoon OCR)"""
    if not hasattr(get_ocr_reader, 'reader'):
        print("🔄 Loading Typhoon OCR...")
        try:
            from typhoon_ocr import ocr_document
            # ตรวจสอบ API key
            api_key = os.getenv("TYPHOON_OCR_API_KEY")
            if not api_key:
                print("⚠️ TYPHOON_OCR_API_KEY not found in environment variables")
                print("   Falling back to EasyOCR. Set TYPHOON_OCR_API_KEY to use Typhoon OCR")
                raise ValueError("API key not found")
            
            # ยืนยันว่า API key ถูกโหลดแล้ว (ไม่แสดงค่าเพื่อความปลอดภัย)
            print(f"✅ TYPHOON_OCR_API_KEY loaded (length: {len(api_key)} characters)")
            
            get_ocr_reader.ocr_document = ocr_document
            get_ocr_reader.reader = "typhoon_ocr"  # ใช้เป็น flag
            print("✅ Typhoon OCR loaded successfully")
        except (ImportError, ValueError) as e:
            print(f"⚠️ Typhoon OCR not available ({e}), falling back to EasyOCR")
            import easyocr
            get_ocr_reader.reader = easyocr.Reader(['en', 'th'], gpu=False, verbose=False)
            get_ocr_reader.ocr_document = None
    return get_ocr_reader.reader

def perform_ocr_on_image_bytes(image_bytes):
    """
    ทำ OCR บน image bytes โดยใช้ Typhoon OCR หรือ EasyOCR (fallback)
    และปรับปรุงข้อความด้วย PyThaiNLP
    
    Args:
        image_bytes: bytes ของรูปภาพ
        
    Returns:
        tuple: (original_text, improved_text) 
               - original_text: ข้อความดิบจาก OCR
               - improved_text: ข้อความที่ปรับปรุงด้วย PyThaiNLP แล้ว
    """
    reader = get_ocr_reader()
    ocr_text = ""
    
    # ตรวจสอบว่าใช้ Typhoon OCR หรือ EasyOCR
    if reader == "typhoon_ocr" and hasattr(get_ocr_reader, 'ocr_document'):
        try:
            # สร้างไฟล์ชั่วคราว
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                tmp_file.write(image_bytes)
                tmp_path = tmp_file.name
            
            try:
                # เรียกใช้ Typhoon OCR
                ocr_document = get_ocr_reader.ocr_document
                markdown_text = ocr_document(pdf_or_image_path=tmp_path)
                
                # แปลง markdown เป็น plain text (ลบ markdown syntax)
                # ลบ markdown headers, bold, italic, etc.
                text = re.sub(r'#+\s*', '', markdown_text)  # ลบ headers
                text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # ลบ bold
                text = re.sub(r'\*([^*]+)\*', r'\1', text)  # ลบ italic
                text = re.sub(r'`([^`]+)`', r'\1', text)  # ลบ code
                text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)  # ลบ links
                text = re.sub(r'\n+', ' ', text)  # แทนที่ newlines ด้วย space
                text = re.sub(r'\s+', ' ', text).strip()  # ลบ spaces ซ้ำ
                
                ocr_text = text
            finally:
                # ลบไฟล์ชั่วคราว
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        except Exception as e:
            print(f"⚠️ Error using Typhoon OCR: {e}, falling back to EasyOCR")
            # Fallback to EasyOCR
            import easyocr
            easyocr_reader = easyocr.Reader(['en', 'th'], gpu=False, verbose=False)
            ocr_results = easyocr_reader.readtext(image_bytes)
            ocr_text = " ".join([result[1] for result in ocr_results if result[2] > 0.3])
    else:
        # ใช้ EasyOCR (fallback)
        ocr_results = reader.readtext(image_bytes)
        ocr_text = " ".join([result[1] for result in ocr_results if result[2] > 0.3])
    
    # 🆕 ปรับปรุงข้อความด้วย PyThaiNLP
    original_text = ocr_text.strip() if ocr_text else ""
    if original_text:
        improved_text = improve_thai_ocr_text(original_text)
        return (original_text, improved_text)
    
    return (original_text, original_text)

# ✅ ฟังก์ชันโหลด embedding model แบบ lazy loading
def get_embedding_model():
    """โหลด embedding model แบบ lazy loading"""
    if not hasattr(get_embedding_model, 'model'):
        print("🔄 Loading embedding model...")
        get_embedding_model.model = SentenceTransformer("minishlab/potion-multilingual-128M", device="cpu")
        print("✅ Embedding model loaded successfully")
    return get_embedding_model.model

# ✅ ฟังก์ชันสร้าง embedding สำหรับข้อความ
def create_text_embedding(text):
    """
    สร้าง embedding สำหรับข้อความ
    
    Args:
        text: ข้อความที่ต้องการสร้าง embedding
        
    Returns:
        list: embedding vector (list of floats) หรือ None ถ้าเกิดข้อผิดพลาด
    """
    if not text or not text.strip():
        return None
    
    try:
        model = get_embedding_model()
        embedding = model.encode(text, convert_to_numpy=True).tolist()
        return embedding
    except Exception as e:
        print(f"⚠️ Error creating embedding: {e}")
        return None

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
    print(f"🖼️ กำลังแปลงรูปภาพเป็นข้อความจาก: {path}")
    images_data = []
    doc = fitz.open(path)
    
    try:
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
                    
                    # OCR (ใช้ Typhoon OCR) และปรับปรุงข้อความด้วย PyThaiNLP
                    original_text, improved_text = perform_ocr_on_image_bytes(image_bytes)
                    
                    if improved_text.strip():
                        image_info = {
                            "page": page_num + 1,
                            "image_index": img_index + 1,
                            "original_text": original_text,
                            "improved_text": improved_text,
                            "text": improved_text,  # ใช้ข้อความที่ปรับปรุงแล้ว
                            "image_base64": base64.b64encode(image_bytes).decode("utf-8")
                        }
                        images_data.append(image_info)
                        
                        print(f"✅ รูป {img_index + 1}: {len(improved_text)} ตัวอักษร (OCR: {len(original_text)} ตัวอักษร)")
                    
                    # ล้าง memory
                    del image, image_bytes
                    
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
    print(f"📊 กำลังแปลงตารางเป็นข้อความจาก: {path}")
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

# ✅ บันทึกข้อมูลต้นฉบับลง MongoDB
def store_original_data_in_mongodb(chunks, collection_name):
    """
    บันทึกข้อมูลต้นฉบับลง ORIGINAL_DB_NAME
    🆕 เพิ่มการสร้าง embeddings ก่อนบันทึก
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
        print(f"📊 ใช้ Database: {db_name} (Original)")
        
        db = client[db_name]
        collection = db[collection_name]
        
        # ลบข้อมูลเก่า
        collection.delete_many({})
        
        # บันทึกข้อมูลต้นฉบับ
        print(f"🔄 กำลังสร้าง embeddings สำหรับ {len(chunks)} chunks...")
        for i, chunk in enumerate(chunks):
            print(f"📝 กำลังบันทึกข้อมูลต้นฉบับ chunk {i+1}/{len(chunks)}...")
            
            # สร้างสำเนาของ chunk และเพิ่ม created_at
            original_chunk = chunk.copy()
            original_chunk["created_at"] = datetime.now()
            
            # 🆕 สร้าง embedding จาก text
            text_content = original_chunk.get('text', '')
            if text_content:
                embedding = create_text_embedding(text_content)
                if embedding:
                    original_chunk['embeddings'] = embedding
                else:
                    print(f"   ⚠️ ไม่สามารถสร้าง embedding สำหรับ chunk {i+1} ได้")
            
            collection.insert_one(original_chunk)
            
            # ตรวจสอบ memory ทุก 5 chunks
            if i % 5 == 0:
                check_memory()
        
        print(f"✅ บันทึกข้อมูลต้นฉบับ {len(chunks)} chunks ลง {collection_name} (พร้อม embeddings)")
        client.close()
        
    except Exception as e:
        print(f"❗ MongoDB Atlas connection failed: {e}")
        print(f"💾 บันทึกลงไฟล์ JSON แทน...")
        
        # Fallback: บันทึกลงไฟล์ JSON
        store_original_to_json(chunks, collection_name)

# ✅ บันทึกข้อมูลต้นฉบับลงไฟล์ JSON (fallback)
def store_original_to_json(chunks, collection_name):
    """
    บันทึกข้อมูลต้นฉบับลงไฟล์ JSON เป็น fallback
    """
    try:
        # สร้างโฟลเดอร์ output ถ้าไม่มี
        output_dir = "output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # บันทึกข้อมูลต้นฉบับ
        original_chunks = []
        for i, chunk in enumerate(chunks):
            print(f"📝 กำลังบันทึกข้อมูลต้นฉบับ chunk {i+1}/{len(chunks)}...")
            
            # สร้างสำเนาของ chunk และเพิ่ม created_at
            original_chunk = chunk.copy()
            original_chunk["created_at"] = datetime.now().isoformat()
            
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

# ✅ ฟังก์ชันประมวลผลหน้าเดียว (ตาม flow ที่ออกแบบ - เจออะไรก่อนทำอันนั้น)
def process_single_page(page_num, pymupdf_page, pdfplumber_pdf, doc_id_counter):
    """
    ประมวลผลหน้าเดียว: Extract → Store
    🆕 แก้ไขให้ทำงานตามลำดับที่เจอในหน้า (เจออะไรก่อนทำอันนั้นก่อน) - เรียงตาม y-coordinate
    
    Args:
        page_num: หมายเลขหน้าที่กำลังประมวลผล (0-based)
        pymupdf_page: หน้า PDF จาก PyMuPDF
        pdfplumber_pdf: PDF object จาก pdfplumber
        doc_id_counter: counter สำหรับสร้าง doc_id
        
    Returns:
        dict: {
            'has_content': bool,  # มีเนื้อหาหรือไม่ (สำหรับตรวจสอบหน้าเปล่า)
            'text_chunks': list,
            'image_chunks': list,
            'table_chunks': list
        }
    """
    page_results = {
        'has_content': False,
        'text_chunks': [],
        'image_chunks': [],
        'table_chunks': []
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
        
        # === STEP 2.5: รวม text blocks ที่อยู่ใกล้กัน (ในบรรทัดเดียวกันหรือใกล้กัน) ===
        # 🆕 เพื่อแก้ปัญหาที่ text blocks ถูกแบ่งเป็น chunks เล็กเกินไป
        text_elements = [e for e in elements if e['type'] == 'text']
        if text_elements:
            # 🆕 กลยุทธ์การรวม: รวม text blocks ที่อยู่ใกล้กันมากขึ้น
            # ใช้ threshold ที่ใหญ่ขึ้น (50 pixels) และรวม chunks ที่สั้นมาก (< 100 ตัวอักษร) เข้าด้วยกัน
            merged_text_chunks = []
            current_chunk_texts = []
            current_chunk_y_pos = None
            current_chunk_bbox = None
            Y_POS_THRESHOLD = 50  # 🆕 เพิ่มจาก 20 เป็น 50 pixels เพื่อรวม chunks ที่อยู่ห่างกันมากขึ้น
            MAX_CHUNK_LENGTH = 2000  # 🆕 กำหนดขนาดสูงสุดของ chunk (2000 ตัวอักษร) เพื่อป้องกัน chunks ใหญ่เกินไป
            
            for text_elem in text_elements:
                y_pos = text_elem['y_pos']
                text_content = text_elem['data']['text']
                bbox = text_elem['data'].get('bbox')
                text_length = len(text_content) if text_content else 0
                
                # 🆕 ถ้า chunk สั้นมาก (< 100 ตัวอักษร) ให้รวมกับ chunk ก่อนหน้าเสมอ (ถ้ามี)
                # หรือถ้า y_pos ใกล้กับ block ก่อนหน้า ให้รวมกัน
                should_merge = False
                if current_chunk_y_pos is None:
                    should_merge = True  # block แรก
                elif abs(y_pos - current_chunk_y_pos) <= Y_POS_THRESHOLD:
                    should_merge = True  # y_pos ใกล้กัน
                elif text_length < 100:
                    # 🆕 ถ้า chunk สั้นมาก ให้รวมกับ chunk ก่อนหน้า (แม้ y_pos จะห่างกัน)
                    # แต่ต้องไม่ห่างเกินไป (ภายใน 100 pixels)
                    if abs(y_pos - current_chunk_y_pos) <= 100:
                        should_merge = True
                
                if should_merge:
                    # 🆕 ตรวจสอบว่าถ้ารวม text นี้เข้าไปแล้ว chunk จะใหญ่เกินไปหรือไม่
                    potential_text = " ".join(current_chunk_texts + [text_content])
                    if len(potential_text) > MAX_CHUNK_LENGTH:
                        # ถ้า chunk จะใหญ่เกินไป ให้บันทึก chunk ปัจจุบันก่อน แล้วเริ่ม chunk ใหม่
                        if current_chunk_texts:
                            merged_text = " ".join(current_chunk_texts)
                            merged_text_chunks.append({
                                'text': merged_text,
                                'y_pos': current_chunk_y_pos,
                                'bbox': current_chunk_bbox
                            })
                        # เริ่ม chunk ใหม่ด้วย text ปัจจุบัน
                        current_chunk_texts = [text_content]
                        current_chunk_y_pos = y_pos
                        current_chunk_bbox = bbox
                    else:
                        # ถ้า chunk ยังไม่ใหญ่เกินไป ให้รวม text นี้เข้าไป
                        current_chunk_texts.append(text_content)
                        if current_chunk_bbox is None:
                            current_chunk_bbox = bbox
                        current_chunk_y_pos = y_pos  # อัพเดท y_pos เป็นของ block ล่าสุด
                else:
                    # ถ้า y_pos ต่างกันมาก แสดงว่าเป็นย่อหน้าใหม่ - สร้าง chunk ใหม่
                    if current_chunk_texts:
                        merged_text = " ".join(current_chunk_texts)
                        merged_text_chunks.append({
                            'text': merged_text,
                            'y_pos': current_chunk_y_pos,
                            'bbox': current_chunk_bbox
                        })
                    # เริ่ม chunk ใหม่
                    current_chunk_texts = [text_content]
                    current_chunk_y_pos = y_pos
                    current_chunk_bbox = bbox
            
            # เพิ่ม chunk สุดท้าย
            if current_chunk_texts:
                merged_text = " ".join(current_chunk_texts)
                merged_text_chunks.append({
                    'text': merged_text,
                    'y_pos': current_chunk_y_pos,
                    'bbox': current_chunk_bbox
                })
            
            # 🆕 ถ้ายังมี chunks ที่สั้นมาก (< 100 ตัวอักษร) ให้รวมกับ chunks ที่อยู่ใกล้กัน
            # รอบที่ 2: รวม chunks ที่สั้นมากกับ chunks ที่อยู่ใกล้กัน
            final_merged_chunks = []
            for i, chunk in enumerate(merged_text_chunks):
                chunk_text = chunk['text']
                chunk_length = len(chunk_text) if chunk_text else 0
                chunk_y_pos = chunk['y_pos']
                
                # ถ้า chunk สั้นมาก (< 100 ตัวอักษร) และมี chunk ถัดไป ให้รวมกัน
                if chunk_length < 100 and i < len(merged_text_chunks) - 1:
                    next_chunk = merged_text_chunks[i + 1]
                    next_y_pos = next_chunk['y_pos']
                    next_text = next_chunk['text']
                    next_length = len(next_text) if next_text else 0
                    # ถ้า y_pos ใกล้กัน (ภายใน 100 pixels) ให้รวมกัน
                    if abs(next_y_pos - chunk_y_pos) <= 100:
                        # 🆕 ตรวจสอบว่าถ้ารวมแล้ว chunk จะใหญ่เกินไปหรือไม่
                        combined_text = chunk_text + " " + next_text
                        if len(combined_text) <= MAX_CHUNK_LENGTH:
                            # รวมกับ chunk ถัดไป
                            final_merged_chunks.append({
                                'text': combined_text,
                                'y_pos': chunk_y_pos,
                                'bbox': chunk.get('bbox')
                            })
                            # ข้าม chunk ถัดไป (เพราะรวมแล้ว)
                            merged_text_chunks[i + 1] = None  # mark as merged
                        else:
                            # ถ้า chunk จะใหญ่เกินไป ให้เก็บ chunk ปัจจุบันไว้
                            final_merged_chunks.append(chunk)
                    else:
                        final_merged_chunks.append(chunk)
                else:
                    # ถ้า chunk นี้ถูก mark เป็น None (ถูกรวมไปแล้ว) ให้ข้าม
                    if chunk is not None:
                        final_merged_chunks.append(chunk)
            
            # กรอง None ออก
            final_merged_chunks = [c for c in final_merged_chunks if c is not None]
            
            # แทนที่ text elements เดิมด้วย merged chunks
            # ลบ text elements เดิมออกจาก elements list
            elements = [e for e in elements if e['type'] != 'text']
            # เพิ่ม merged text chunks กลับเข้าไป
            for merged_chunk in final_merged_chunks:
                elements.append({
                    'type': 'text_merged',
                    'y_pos': merged_chunk['y_pos'],
                    'data': {
                        'text': merged_chunk['text'],
                        'bbox': merged_chunk.get('bbox')
                    }
                })
            
            # เรียงลำดับใหม่หลังจาก merge
            elements.sort(key=lambda x: x['y_pos'])
            print(f"🔄 รวม text blocks เป็น {len(final_merged_chunks)} chunks (จาก {len(text_elements)} blocks เดิม)")
            
            # 🆕 ตรวจสอบและแสดงสถิติของ chunks
            chunk_lengths = [len(chunk['text']) for chunk in final_merged_chunks if chunk.get('text')]
            if chunk_lengths:
                avg_length = sum(chunk_lengths) / len(chunk_lengths)
                max_length = max(chunk_lengths)
                min_length = min(chunk_lengths)
                chunks_over_limit = sum(1 for length in chunk_lengths if length > MAX_CHUNK_LENGTH)
                
                print(f"   📊 สถิติ chunks:")
                print(f"      - จำนวน chunks: {len(final_merged_chunks)}")
                print(f"      - ขนาดเฉลี่ย: {avg_length:.0f} ตัวอักษร")
                print(f"      - ขนาดสูงสุด: {max_length} ตัวอักษร")
                print(f"      - ขนาดต่ำสุด: {min_length} ตัวอักษร")
                if chunks_over_limit > 0:
                    print(f"      ⚠️ พบ {chunks_over_limit} chunks ที่ใหญ่เกิน {MAX_CHUNK_LENGTH} ตัวอักษร")
                else:
                    print(f"      ✅ ทุก chunks มีขนาดไม่เกิน {MAX_CHUNK_LENGTH} ตัวอักษร")
            
            for i, chunk in enumerate(final_merged_chunks[:5], 1):  # แสดง 5 อันดับแรก
                chunk_length = len(chunk.get('text', ''))
                size_indicator = " ⚠️ ใหญ่เกินไป" if chunk_length > MAX_CHUNK_LENGTH else ""
                print(f"   📝 Merged chunk {i}: {chunk_length} ตัวอักษร{size_indicator}")
        
        # === STEP 3: ประมวลผลตามลำดับที่เรียงแล้ว (เจออะไรก่อนทำอันนั้นก่อน) ===
        text_chunk_counter = 0
        image_chunk_counter = 0
        table_chunk_counter = 0
        
        for element_index, element in enumerate(elements):
            element_type = element['type']
            data = element['data']
            
            print(f"\n📌 Element {element_index + 1}/{len(elements)}: {element_type.upper()} "
                  f"(y={element['y_pos']:.1f})")
            
            if element_type == 'text' or element_type == 'text_merged':
                # ประมวลผล Text Block (ทั้งแบบเดิมและแบบ merged)
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
                page_results['text_chunks'].append(text_chunk)
                text_chunk_counter += 1
            
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
                    
                    # OCR (ใช้ Typhoon OCR) และปรับปรุงข้อความด้วย PyThaiNLP
                    print(f"   🔍 กำลังทำ OCR...")
                    original_text, improved_text = perform_ocr_on_image_bytes(image_bytes)
                    
                    if improved_text.strip():
                        page_results['has_content'] = True
                        
                        print(f"   🖼️ Image {img_index + 1}: {len(improved_text)} ตัวอักษร (OCR: {len(original_text)} ตัวอักษร)")
                        
                        # Create image chunk
                        image_chunk = {
                            "text": improved_text,
                            "type": "image",
                            "chunk_id": image_chunk_counter,
                            "page": page_num + 1,
                            "image_index": img_index + 1,
                            "original_text": original_text,
                            "improved_text": improved_text,
                            "image_base64": base64.b64encode(image_bytes).decode("utf-8"),
                            "doc_id": f"doc_{doc_id_counter}_{page_num + 1}_img_{img_index + 1}",
                            "bbox": convert_bbox_to_mongodb_format(data['bbox'])
                        }
                        page_results['image_chunks'].append(image_chunk)
                        image_chunk_counter += 1
                    else:
                        print(f"   ⚠️ ไม่พบข้อความในรูปภาพ {img_index + 1} (OCR ไม่เจอข้อความ) - ข้าม")
                    
                    # ล้าง memory
                    del image, image_bytes
                    
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
                    page_results['table_chunks'].append(table_chunk)
                    table_chunk_counter += 1
        
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
    🆕 เพิ่มการสร้าง embeddings ก่อนบันทึก
    
    Args:
        page_results: ผลลัพธ์จาก process_single_page()
        client: MongoDB client (เปิดไว้แล้ว)
        is_first_page: เป็นหน้าแรกหรือไม่ (ถ้าใช่จะลบข้อมูลเก่าก่อน)
    """
    try:
        # เตรียม database และ collections
        db_original = client[ORIGINAL_DB_NAME]
        
        orig_text_col = db_original[ORIGINAL_TEXT_COLLECTION]
        orig_image_col = db_original[ORIGINAL_IMAGE_COLLECTION]
        orig_table_col = db_original[ORIGINAL_TABLE_COLLECTION]
        
        # ลบข้อมูลเก่าครั้งเดียวตอนหน้าแรก
        if is_first_page:
            print("🗑️ ลบข้อมูลเก่าใน MongoDB...")
            orig_text_col.delete_many({})
            orig_image_col.delete_many({})
            orig_table_col.delete_many({})
            print("✅ ลบข้อมูลเก่าเสร็จสิ้น")
        
        # เพิ่ม created_at และ embeddings ให้ทุก chunk
        now = datetime.now()
        
        # บันทึก Original Data - Text Chunks
        if page_results['text_chunks']:
            print(f"   🔄 กำลังสร้าง embeddings สำหรับ {len(page_results['text_chunks'])} text chunks...")
            for chunk in page_results['text_chunks']:
                chunk['created_at'] = now
                # 🆕 สร้าง embedding จาก text
                text_content = chunk.get('text', '')
                if text_content:
                    embedding = create_text_embedding(text_content)
                    if embedding:
                        chunk['embeddings'] = embedding
                    else:
                        print(f"   ⚠️ ไม่สามารถสร้าง embedding สำหรับ text chunk {chunk.get('chunk_id', 'unknown')} ได้")
            orig_text_col.insert_many(page_results['text_chunks'])
            print(f"   ✅ บันทึก {len(page_results['text_chunks'])} text chunks (พร้อม embeddings)")
        
        # บันทึก Original Data - Image Chunks
        if page_results['image_chunks']:
            print(f"   🔄 กำลังสร้าง embeddings สำหรับ {len(page_results['image_chunks'])} image chunks...")
            for chunk in page_results['image_chunks']:
                chunk['created_at'] = now
                # 🆕 สร้าง embedding จาก text (ข้อความที่ได้จาก OCR)
                text_content = chunk.get('text', '')
                if text_content:
                    embedding = create_text_embedding(text_content)
                    if embedding:
                        chunk['embeddings'] = embedding
                    else:
                        print(f"   ⚠️ ไม่สามารถสร้าง embedding สำหรับ image chunk {chunk.get('chunk_id', 'unknown')} ได้")
            orig_image_col.insert_many(page_results['image_chunks'])
            print(f"   ✅ บันทึก {len(page_results['image_chunks'])} image chunks (พร้อม embeddings)")
        
        # บันทึก Original Data - Table Chunks
        if page_results['table_chunks']:
            print(f"   🔄 กำลังสร้าง embeddings สำหรับ {len(page_results['table_chunks'])} table chunks...")
            for chunk in page_results['table_chunks']:
                chunk['created_at'] = now
                # 🆕 สร้าง embedding จาก text
                text_content = chunk.get('text', '')
                if text_content:
                    embedding = create_text_embedding(text_content)
                    if embedding:
                        chunk['embeddings'] = embedding
                    else:
                        print(f"   ⚠️ ไม่สามารถสร้าง embedding สำหรับ table chunk {chunk.get('chunk_id', 'unknown')} ได้")
            orig_table_col.insert_many(page_results['table_chunks'])
            print(f"   ✅ บันทึก {len(page_results['table_chunks'])} table chunks (พร้อม embeddings)")
        
        return True
        
    except Exception as e:
        print(f"❗ Error storing page results to MongoDB: {e}")
        import traceback
        traceback.print_exc()
        return False

# ✅ ฟังก์ชันหลัก (ประมวลผลหนึ่งหน้า → บันทึก → loop ต่อ)
def main():
    print("🚀 เริ่ม Pipeline: Extract → OCR + PyThaiNLP → Store")
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
        
        doc_id_counter = 1  # สำหรับสร้าง doc_id
        
        # === LOOP: More Pages (ประมวลผลและบันทึกทีละหน้า) ===
        print("\n=== STEP 1: PAGE-BY-PAGE PROCESSING & STORING ===")
        for page_num in range(total_pages):
            print(f"\n{'='*60}")
            print(f"📄 กำลังประมวลผลหน้า {page_num + 1}/{total_pages}")
            print(f"{'='*60}")
            
            # ประมวลผลหน้าเดียว (Extract)
            page_results = process_single_page(
                page_num=page_num,
                pymupdf_page=pymupdf_doc[page_num],
                pdfplumber_pdf=pdfplumber_pdf,
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
        print(f"   📝 Text chunks: {total_text_chunks}")
        print(f"   🖼️ Image chunks: {total_image_chunks}")
        print(f"   📊 Table chunks: {total_table_chunks}")
        print(f"   📊 Total chunks: {total_text_chunks + total_image_chunks + total_table_chunks}")
        
        print("\n✅ Pipeline เสร็จสิ้น!")
        print(f"✅ ข้อมูลทั้งหมดถูกบันทึกใน MongoDB:")
        print(f"   - Database: {ORIGINAL_DB_NAME}")
        
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
                
                orig_text_count = db_original[ORIGINAL_TEXT_COLLECTION].count_documents({})
                orig_image_count = db_original[ORIGINAL_IMAGE_COLLECTION].count_documents({})
                orig_table_count = db_original[ORIGINAL_TABLE_COLLECTION].count_documents({})
                
                print(f"\n⚠️ ข้อมูลที่บันทึกไปแล้ว:")
                print(f"   - Original text chunks: {orig_text_count}")
                print(f"   - Original image chunks: {orig_image_count}")
                print(f"   - Original table chunks: {orig_table_count}")
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
