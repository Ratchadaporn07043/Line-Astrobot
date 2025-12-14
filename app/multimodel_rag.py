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
from langchain_text_splitters import RecursiveCharacterTextSplitter


# 🆕 เพิ่ม camelot สำหรับ extract ตาราง
try:
    import camelot
    CAMELOT_AVAILABLE = True
    print("✅ โหลด Camelot สำเร็จ")
except ImportError:
    CAMELOT_AVAILABLE = False
    print("⚠️ Camelot ไม่พร้อมใช้งาน ใช้ pdfplumber สำหรับการดึงตารางแทน")

# 🆕 เพิ่ม PyThaiNLP สำหรับปรับปรุง OCR
try:
    from pythainlp import word_tokenize
    from pythainlp.spell import correct
    from pythainlp.util import normalize
    PYTHAINLP_AVAILABLE = True
    print("✅ โหลด PyThaiNLP สำเร็จ")
except ImportError:
    PYTHAINLP_AVAILABLE = False
    print("⚠️ PyThaiNLP ไม่พร้อมใช้งาน ใช้การประมวลผลข้อความพื้นฐานแทน")

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
        print(f"   ⚠️ เกิดข้อผิดพลาดในการแปลง bbox: {e}")
        return None

# ✅ ฟังก์ชันสำหรับตัดข้อความเป็น Chunks
def chunk_text_content(text):
    """
    ตัดข้อความเป็น Chunks โดยใช้ RecursiveCharacterTextSplitter
    - รองรับภาษาไทย (ใช้ separators ที่เหมาะสม)
    - ขนาด chunk 1000 characters, overlap 200
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_text(text)

# ✅ ฟังก์ชันตรวจสอบ memory
def check_memory():
    """ตรวจสอบการใช้ memory"""
    memory = psutil.virtual_memory()
    print(f"💾 Memory: {memory.percent}% ({memory.used / 1024**3:.1f}GB / {memory.total / 1024**3:.1f}GB)")
    if memory.percent > 80:
        print("⚠️ High memory usage, running garbage collection...")
        gc.collect()

# 🆕 Dictionary สำหรับคำที่พบบ่อยใน OCR ที่ถูกแยกผิด
COMMON_OCR_CORRECTIONS = {
    'สญ ลก ษณ์': 'สัญลักษณ์',
    'สญลกษณ์': 'สัญลักษณ์',
    'สญ ลก ษณ': 'สัญลักษณ์',
    'สญลกษณ': 'สัญลักษณ์',
    'ตลุ': 'ตุลย์',
    'พิจิกิ': 'พิจิก',
    'พิ จิ กิ': 'พิจิก',
    'มถิ นุ': 'มิถุน',
    'มถินุ': 'มิถุน',
    'กรก ฏ': 'กรกฎ',
    'กรกฏ': 'กรกฎ',
    'มกร': 'มกร',
    'มิถุน': 'มิถุน',
    'พฤษภ': 'พฤษภ',
    'กรกฎ': 'กรกฎ',
    'ธนู': 'ธนู',
    'เมษ': 'เมษ',
}

# 🆕 ฟังก์ชันปรับปรุงข้อความไทยจาก OCR ด้วย PyThaiNLP (ปรับปรุงให้ดีขึ้น)
def improve_thai_ocr_text(ocr_text):
    """
    ปรับปรุงข้อความไทยจาก OCR ด้วย PyThaiNLP (เวอร์ชันปรับปรุง)
    - Normalize ข้อความ (แก้ไขตัวอักษรพิเศษ, วรรณยุกต์)
    - แก้ไขคำที่ถูกแยกผิดด้วย dictionary (COMMON_OCR_CORRECTIONS)
    - รวมคำที่ถูกแยกด้วยช่องว่าง (pattern matching)
    - แก้ไขการเว้นวรรคระหว่างภาษาและตัวเลข
    - แบ่งคำด้วย word tokenizer (newmm engine)
    - แก้ไขคำผิดด้วย spell checker (เฉพาะคำไทย)
    - ทำความสะอาดข้อความขั้นสุดท้าย
    """
    if not PYTHAINLP_AVAILABLE or not ocr_text.strip():
        return ocr_text
    
    try:
        # ทำความสะอาดข้อความเบื้องต้น
        text = ocr_text.strip()
        
        # 🆕 แก้ไขคำที่พบบ่อยใน OCR ที่ถูกแยกผิด (ทำก่อน normalize)
        for wrong, correct in COMMON_OCR_CORRECTIONS.items():
            # แทนที่ทั้งแบบมีช่องว่างและไม่มีช่องว่าง
            text = text.replace(wrong, correct)
            # แทนที่แบบไม่มีช่องว่าง (กรณีที่ถูกแยกเป็น "สญลกษณ์")
            text = text.replace(wrong.replace(' ', ''), correct)
        
        # 🆕 รวมคำที่ถูกแยกด้วยช่องว่าง (เช่น "สญ ลก ษณ์" -> "สัญลักษณ์")
        # หาคำที่ถูกแยกด้วยช่องว่าง (คำไทยที่สั้นๆ หลายคำติดกัน)
        
        # Pattern 1: คำไทย 1-3 ตัว + ช่องว่าง + คำไทย 1-3 ตัว + ช่องว่าง + คำไทย 1-3 ตัว (3 คำ)
        thai_word_pattern_3 = r'([ก-๙]{1,3})\s+([ก-๙]{1,3})\s+([ก-๙]{1,3})'
        matches_3 = list(re.finditer(thai_word_pattern_3, text))
        matches_3.sort(key=lambda m: m.end() - m.start(), reverse=True)
        
        # Pattern 2: คำไทย 1-3 ตัว + ช่องว่าง + คำไทย 1-3 ตัว (2 คำ)
        thai_word_pattern_2 = r'([ก-๙]{1,3})\s+([ก-๙]{1,3})'
        matches_2 = list(re.finditer(thai_word_pattern_2, text))
        matches_2.sort(key=lambda m: m.end() - m.start(), reverse=True)
        
        replacements = []
        
        # ประมวลผลคำที่ถูกแยกเป็น 3 คำก่อน
        for match in matches_3:
            combined = match.group(1) + match.group(2) + match.group(3)
            if combined in COMMON_OCR_CORRECTIONS:
                replacements.append((match.group(0), COMMON_OCR_CORRECTIONS[combined]))
            elif combined in COMMON_OCR_CORRECTIONS.values():
                replacements.append((match.group(0), combined))
            elif len(combined) >= 4:
                replacements.append((match.group(0), combined))
        
        # ประมวลผลคำที่ถูกแยกเป็น 2 คำ
        for match in matches_2:
            combined = match.group(1) + match.group(2)
            if combined in COMMON_OCR_CORRECTIONS:
                replacements.append((match.group(0), COMMON_OCR_CORRECTIONS[combined]))
            elif combined in COMMON_OCR_CORRECTIONS.values():
                replacements.append((match.group(0), combined))
            elif len(combined) >= 3:
                replacements.append((match.group(0), combined))
        
        # แทนที่คำที่ถูกแยก (จากหลังไปหน้าเพื่อไม่ให้ตำแหน่งเปลี่ยน)
        for old, new in reversed(replacements):
            text = text.replace(old, new, 1)  # แทนที่ครั้งเดียว
        
        # 🆕 แก้ไขคำที่พบบ่อยอีกครั้ง (หลังจากรวมคำแล้ว)
        for wrong, correct in COMMON_OCR_CORRECTIONS.items():
            text = text.replace(wrong, correct)
        
        # 🆕 Normalize ข้อความด้วย PyThaiNLP (แก้ไขตัวอักษรพิเศษ, วรรณยุกต์)
        try:
            text = normalize(text)
        except:
            pass  # ถ้า normalize ไม่ได้ ให้ข้าม
        
        # แก้ไขการเว้นวรรคที่ผิด
        text = re.sub(r'([ก-๙])([A-Za-z])', r'\1 \2', text)  # เว้นวรรคระหว่างไทย-อังกฤษ
        text = re.sub(r'([A-Za-z])([ก-๙])', r'\1 \2', text)  # เว้นวรรคระหว่างอังกฤษ-ไทย
        text = re.sub(r'([ก-๙])([0-9])', r'\1 \2', text)    # เว้นวรรคระหว่างไทย-ตัวเลข
        text = re.sub(r'([0-9])([ก-๙])', r'\1 \2', text)    # เว้นวรรคระหว่างตัวเลข-ไทย
        
        # แก้ไขการเว้นวรรคที่ซ้ำ
        text = re.sub(r'\s+', ' ', text)
        
        # 🆕 แบ่งคำด้วย PyThaiNLP (ใช้ newmm engine สำหรับความแม่นยำ)
        try:
            words = word_tokenize(text, engine='newmm')
        except Exception as e:
            # Fallback: แบ่งคำแบบง่ายๆ ถ้า word_tokenize ไม่ได้
            print(f"   ⚠️ word_tokenize ล้มเหลว: {e}, ใช้ simple split แทน")
            words = text.split()
        
        # 🆕 แก้ไขคำผิดด้วย PyThaiNLP spell checker (ปรับปรุงให้ดีขึ้น)
        corrected_words = []
        for word in words:
            # ตรวจสอบว่าเป็นคำไทยหรือไม่ (มีตัวอักษรไทย)
            has_thai = bool(re.search(r'[ก-๙]', word))
            
            if has_thai and len(word) > 2:
                # แก้ไขเฉพาะคำไทยที่มีความยาวมากกว่า 2 ตัวอักษร
                # ตรวจสอบว่าเป็นคำที่ประกอบด้วยตัวอักษรเท่านั้น (ไม่รวมตัวเลข/สัญลักษณ์)
                is_alpha_only = bool(re.match(r'^[ก-๙a-zA-Z]+$', word))
                
                if is_alpha_only:
                    try:
                        corrected = correct(word)
                        # ใช้คำที่แก้ไขแล้วถ้าไม่ใช่ None และไม่เหมือนเดิม
                        if corrected and corrected != word:
                            # ตรวจสอบว่าคำที่แก้ไขแล้วมีความยาวใกล้เคียงกับคำเดิม (อนุญาตให้ต่างได้ 2 ตัวอักษร)
                            length_diff = abs(len(corrected) - len(word))
                            # ตรวจสอบว่าคำที่แก้ไขแล้วมีตัวอักษรไทย (ไม่ใช่คำแปลกๆ)
                            has_thai_corrected = bool(re.search(r'[ก-๙]', corrected))
                            
                            if length_diff <= 2 and has_thai_corrected:
                                corrected_words.append(corrected)
                            else:
                                corrected_words.append(word)
                        else:
                            corrected_words.append(word)
                    except Exception as e:
                        # ถ้าแก้ไขไม่ได้ ให้ใช้คำเดิม
                        corrected_words.append(word)
                else:
                    # มีตัวเลขหรือสัญลักษณ์ ให้เก็บไว้ตามเดิม
                    corrected_words.append(word)
            else:
                # ไม่ใช่คำไทย หรือสั้นเกินไป หรือเป็นตัวเลข/สัญลักษณ์ ให้เก็บไว้ตามเดิม
                corrected_words.append(word)
        
        # รวมคำกลับเป็นประโยค
        improved_text = ' '.join(corrected_words)
        
        # 🆕 ทำความสะอาดขั้นสุดท้าย (ปรับปรุงให้ดีขึ้น)
        # ลบช่องว่างซ้ำ
        improved_text = re.sub(r'\s+', ' ', improved_text)
        # ลบช่องว่างหน้าและหลังเครื่องหมายวรรคตอน
        improved_text = re.sub(r'\s+([,\.;:!?])', r'\1', improved_text)
        improved_text = re.sub(r'([,\.;:!?])\s+', r'\1 ', improved_text)
        # ลบช่องว่างที่ต้นและท้าย
        improved_text = improved_text.strip()
        
        return improved_text
        
    except Exception as e:
        print(f"⚠️ เกิดข้อผิดพลาดในการปรับปรุงข้อความไทย: {e}")
        return ocr_text

# 🆕 ฟังก์ชันปรับปรุงข้อความในตารางด้วย PyThaiNLP (ปรับปรุงให้ดีขึ้น)
def improve_thai_table_text(table_text):
    """
    ปรับปรุงข้อความในตารางด้วย PyThaiNLP (เวอร์ชันปรับปรุง)
    - แยกแต่ละเซลล์ในตาราง (แยกด้วย " | ")
    - ปรับปรุงข้อความในแต่ละเซลล์ด้วย improve_thai_ocr_text() (normalize, spell check, word tokenize)
    - รักษาโครงสร้างตาราง (แถวและคอลัมน์)
    - ทำความสะอาดข้อความขั้นสุดท้าย
    """
    if not PYTHAINLP_AVAILABLE or not table_text.strip():
        return table_text
    
    try:
        # แยกตารางเป็นแถว
        rows = table_text.split('\n')
        improved_rows = []
        
        for row in rows:
            if not row.strip():
                improved_rows.append(row)
                continue
            
            # แยกเซลล์ด้วย " | " (รองรับทั้ง " | " และ "|")
            # ใช้ regex เพื่อแยกเซลล์ที่ถูกต้อง (ไม่แยกในกรณีที่มี " | " ในข้อความ)
            cells = [cell.strip() for cell in row.split(' | ')]
            improved_cells = []
            
            for cell in cells:
                if cell.strip():
                    # ✅ ปรับปรุงข้อความในแต่ละเซลล์ด้วย improve_thai_ocr_text()
                    # ซึ่งจะทำ normalize, spell check, word tokenize, และทำความสะอาด
                    improved_cell = improve_thai_ocr_text(cell.strip())
                    improved_cells.append(improved_cell)
                else:
                    improved_cells.append(cell)
            
            # รวมเซลล์กลับเป็นแถว (ใช้ " | " เป็นตัวคั่น)
            improved_row = ' | '.join(improved_cells)
            improved_rows.append(improved_row)
        
        # รวมแถวกลับเป็นตาราง
        improved_table = '\n'.join(improved_rows)
        
        # 🆕 ทำความสะอาดข้อความขั้นสุดท้าย
        # ลบช่องว่างซ้ำในแต่ละแถว
        improved_table = re.sub(r' +', ' ', improved_table)
        # ลบช่องว่างที่ต้นและท้ายแต่ละแถว
        improved_table = '\n'.join([row.strip() for row in improved_table.split('\n')])
        
        return improved_table
        
    except Exception as e:
        print(f"⚠️ เกิดข้อผิดพลาดในการปรับปรุงข้อความตารางไทย: {e}")
        return table_text

def get_ocr_reader():
    """โหลด OCR reader แบบ lazy loading (ใช้เฉพาะ Typhoon OCR)"""
    if not hasattr(get_ocr_reader, 'reader'):
        print("🔄 กำลังโหลด Typhoon OCR...")
        try:
            from typhoon_ocr import ocr_document
            # ตรวจสอบ API key
            api_key = os.getenv("TYPHOON_OCR_API_KEY")
            if not api_key:
                error_msg = "ไม่พบ TYPHOON_OCR_API_KEY ในตัวแปรสภาพแวดล้อม กรุณาตั้งค่า TYPHOON_OCR_API_KEY เพื่อใช้ Typhoon OCR"
                print(f"❌ {error_msg}")
                raise ValueError(error_msg)
            
            # ยืนยันว่า API key ถูกโหลดแล้ว (ไม่แสดงค่าเพื่อความปลอดภัย)
            print(f"✅ โหลด TYPHOON_OCR_API_KEY สำเร็จ (ความยาว: {len(api_key)} ตัวอักษร)")
            
            get_ocr_reader.ocr_document = ocr_document
            get_ocr_reader.reader = "typhoon_ocr"  # ใช้เป็น flag
            print("✅ โหลด Typhoon OCR สำเร็จ")
        except ImportError as e:
            error_msg = f"ไลบรารี Typhoon OCR ไม่พร้อมใช้งาน กรุณาติดตั้งแพ็กเกจ typhoon-ocr ข้อผิดพลาด: {e}"
            print(f"❌ {error_msg}")
            raise ImportError(error_msg)
        except ValueError as e:
            raise  # Re-raise ValueError from API key check
        except Exception as e:
            error_msg = f"ไม่สามารถโหลด Typhoon OCR ได้: {e}"
            print(f"❌ {error_msg}")
            raise RuntimeError(error_msg)
    return get_ocr_reader.reader

def perform_ocr_on_image_bytes(image_bytes):
    """
    ทำ OCR บน image bytes โดยใช้เฉพาะ Typhoon OCR
    
    Args:
        image_bytes: bytes ของรูปภาพ
        
    Returns:
        str: ข้อความที่ได้จาก OCR (ผ่านการปรับปรุงด้วย PyThaiNLP แล้ว)
        
    Raises:
        RuntimeError: ถ้าไม่สามารถใช้ Typhoon OCR ได้
    """
    # ตรวจสอบว่าใช้ Typhoon OCR
    reader = get_ocr_reader()
    if reader != "typhoon_ocr" or not hasattr(get_ocr_reader, 'ocr_document'):
        raise RuntimeError("Typhoon OCR ไม่พร้อมใช้งาน กรุณาตรวจสอบ TYPHOON_OCR_API_KEY และการติดตั้ง typhoon-ocr")
    
    # สร้างไฟล์ชั่วคราว
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            tmp_file.write(image_bytes)
            tmp_path = tmp_file.name
        
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
        
        # ✅ ปรับปรุงข้อความด้วย PyThaiNLP ทันทีหลังจากได้ผลลัพธ์จาก OCR
        if text.strip():
            text = improve_thai_ocr_text(text)
        
        return text
    except Exception as e:
        error_msg = f"เกิดข้อผิดพลาดในการใช้ Typhoon OCR: {e}"
        print(f"❌ {error_msg}")
        raise RuntimeError(error_msg)
    finally:
        # ลบไฟล์ชั่วคราว
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except:
                pass

# ✅ ฟังก์ชันโหลด embedding model แบบ lazy loading
def get_embedding_model():
    """โหลด embedding model แบบ lazy loading - ใช้โมเดล minishlab/potion-multilingual-128M"""
    if not hasattr(get_embedding_model, 'model'):
        model_name = "minishlab/potion-multilingual-128M"
        print(f"🔄 กำลังโหลด embedding model: {model_name}...")
        get_embedding_model.model = SentenceTransformer(model_name, device="cpu")
        print(f"✅ โหลด embedding model สำเร็จ: {model_name}")
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
        print(f"⚠️ เกิดข้อผิดพลาดในการสร้าง embedding: {e}")
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
def extract_images_from_page(page_num, pymupdf_page, doc):
    """
    แปลงรูปภาพใน PDF เป็นข้อความด้วย OCR + PyThaiNLP (สำหรับหน้าเดียว)
    """
    images_data = []
    
    try:
        images = pymupdf_page.get_images(full=True)
        if images:
            print(f"   🖼️ หน้า {page_num}: พบ {len(images)} รูป")
        
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
                    print(f"      ⚠️ ข้ามรูปใหญ่ {img_index + 1} ({width}x{height})")
                    continue
                
                # ข้ามรูปที่เล็กเกินไป
                if width < 50 or height < 50:
                    print(f"      ⚠️ ข้ามรูปเล็ก {img_index + 1} ({width}x{height})")
                    continue
                
                # OCR (ใช้ Typhoon OCR - ปรับปรุงข้อความด้วย PyThaiNLP แล้ว)
                improved_text = perform_ocr_on_image_bytes(image_bytes)
                
                if improved_text.strip():
                    image_info = {
                        "page": page_num,
                        "image_index": img_index + 1,
                        "text": improved_text,  # ข้อความที่ผ่านการปรับปรุงด้วย PyThaiNLP แล้ว
                        "improved_text": improved_text,
                        "image_base64": base64.b64encode(image_bytes).decode("utf-8"),
                         "type": "image", # Add type
                         "metadata": {
                            "source": "image_ocr",
                            "page": page_num
                        }
                    }
                    images_data.append(image_info)
                    
                    print(f"      ✅ รูป {img_index + 1}: {len(improved_text)} ตัวอักษร (OCR)")
                
                # ล้าง memory
                del image, image_bytes
                
            except Exception as e:
                print(f"      ❗ เกิดข้อผิดพลาดในการประมวลผลรูปภาพ {img_index + 1} ในหน้า {page_num}: {e}")
                continue
        
        # จำกัดจำนวนรูปต่อหน้า
        if len(images_data) > 20:  # จำกัดไม่เกิน 20 รูปต่อหน้า
            print("      ⚠️ จำกัดจำนวนรูปที่ 20 รูปต่อหน้า")
            images_data = images_data[:20]
            
    except Exception as e:
         print(f"   ❗ เกิดข้อผิดพลาดในการดึงรูปภาพหน้า {page_num}: {e}")

    return images_data

# ✅ แปลงตารางเป็นข้อความด้วย camelot + PyThaiNLP (ใช้เฉพาะ camelot)
def extract_tables_from_page(path, page_num):
    """
    แปลงตารางใน PDF เป็นข้อความด้วย camelot + PyThaiNLP (สำหรับหน้าเดียว)
    """
    tables_data = []
    
    if not CAMELOT_AVAILABLE:
        # Silently fail or simple print if not available, as handled in top level
        # But we already checked imports.
        return []
    
    try:
        # ใช้ camelot extract ตารางจาก PDF เฉพาะหน้านั้น
        # camelot pages argument accepts strings like '1', '1-5', etc.
        # page_num is 1-based index for camelot
        
        tables = []
        try:
            # Suppress stdout from camelot if possible or just let it print
            tables = camelot.read_pdf(path, pages=str(page_num), flavor='lattice')
        except Exception as e1:
            try:
                tables = camelot.read_pdf(path, pages=str(page_num), flavor='stream')
            except Exception as e2:
                # ถ้าไม่เจอตารางหรือไม่ error ก็แค่ return empty
                pass
        
        if tables and len(tables) > 0:
            print(f"   📊 หน้า {page_num}: พบ {len(tables)} ตาราง")

        for table_index, table in enumerate(tables):
            try:
                # แปลงตารางเป็น list of lists
                try:
                    table_data = table.df.values.tolist()
                except:
                    table_data = [[str(cell) for cell in row] for row in table.df.values] if hasattr(table.df, 'values') else []
                
                # ✅ ปรับปรุงข้อความในแต่ละเซลล์ด้วย PyThaiNLP
                table_text = ""
                for row in table_data:
                    if row:
                        improved_cells = []
                        for cell in row:
                            cell_str = str(cell).strip() if cell is not None and str(cell).strip() else ""
                            if cell_str:
                                if PYTHAINLP_AVAILABLE:
                                    improved_cell = improve_thai_ocr_text(cell_str)
                                else:
                                    improved_cell = cell_str
                                improved_cells.append(improved_cell)
                            else:
                                improved_cells.append("")
                        
                        row_text = " | ".join(improved_cells)
                        if row_text.strip():
                            table_text += row_text + "\n"
                
                if table_text.strip():
                    improved_table_text = improve_thai_table_text(table_text.strip())
                    
                    table_info = {
                        "page": page_num,
                        "table_index": table_index + 1,
                        "original_text": table_text.strip(),
                        "improved_text": improved_table_text,
                        "text": improved_table_text,
                        "bbox": table._bbox if hasattr(table, '_bbox') else None,
                        "type": "table", # Add type
                        "metadata": {
                            "source": "table_camelot",
                            "page": page_num
                        }
                    }
                    tables_data.append(table_info)
                    print(f"      ✅ ตาราง {table_index + 1}: {len(improved_table_text)} ตัวอักษร")
                
            except Exception as e:
                print(f"      ⚠️ เกิดข้อผิดพลาดในการประมวลผลตาราง {table_index + 1}: {e}")
                continue
                    
    except Exception as e:
        print(f"   ❗ เกิดข้อผิดพลาดในการดึงตารางหน้า {page_num}: {e}")
    
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
        print(f"❗ การเชื่อมต่อ MongoDB Atlas ล้มเหลว: {e}")
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
        print(f"❗ เกิดข้อผิดพลาดในการบันทึกข้อมูลต้นฉบับเป็น JSON: {e}")

        return text_chunks
        
    except Exception as e:
        print(f"❗ เกิดข้อผิดพลาดในการประมวลผลหน้า {page_num + 1}: {e}")
        import traceback
        traceback.print_exc()
        return []

# ✅ ฟังก์ชันประมวลผลหน้าเดียว (ตาม flow ที่ออกแบบ - เจออะไรก่อนทำอันนั้น)
def process_single_page(page_num, pymupdf_page, pdfplumber_pdf, doc_id_counter, pdf_path=None):
    """
    ประมวลผลหน้าเดียว: Extract → Clean → Chunk → Store
    🆕 ใช้ RecursiveCharacterTextSplitter สำหรับ chunking
    """
    page_results = {
        'has_content': False,
        'text_chunks': [],
        'image_chunks': [],
        'table_chunks': []
    }
    
    try:
        print(f"\n{'='*50}")
        print(f"📄 กำลังประมวลผลหน้า {page_num + 1}")
        print(f"{'='*50}")
        
        # ดึงข้อความทั้งหมดในหน้า
        text = pymupdf_page.get_text("text")
        if not text.strip():
            print(f"⚠️ หน้า {page_num + 1} ไม่มีข้อความ")
            return page_results

        # Clean text เบื้องต้น
        text = re.sub(r'\s+', ' ', text).strip()
        
        # ใช้ improved_ocr logic ถ้ามี (optional) แต่นี่คือ text จาก PDF โดยตรง
        # เราอาจจะข้าม improve_thai_ocr_text ถ้า PDF text อ่านได้ดีอยู่แล้ว
        # แต่ถ้า PDF text แย่ อาจจะต้องใช้ OCR หรือ improve logic
        
        # ตัดแบ่ง Chunk
        chunks = chunk_text_content(text)
        print(f"   📝 หน้า {page_num + 1}: แบ่งได้ {len(chunks)} chunks")
        
        for i, chunk_text in enumerate(chunks):
             chunk_info = {
                "doc_id": doc_id_counter + i,
                "page": page_num + 1,
                "chunk_index": i + 1,
                "text": chunk_text,
                "type": "text",
                "metadata": {
                    "source": "pdf_text",
                    "page": page_num + 1
                }
            }
             page_results['text_chunks'].append(chunk_info)
        
        page_results['has_content'] = True
        
        # Extract Images
        if pymupdf_page:
            image_chunks = extract_images_from_page(page_num + 1, pymupdf_page, pymupdf_page.parent)
            if image_chunks:
                for i, chunk in enumerate(image_chunks):
                    # Add remaining necessary fields
                    chunk["doc_id"] = doc_id_counter + len(page_results['text_chunks']) + i
                    page_results['image_chunks'].append(chunk)

        # Extract Tables
        if pdf_path and CAMELOT_AVAILABLE:
            table_chunks = extract_tables_from_page(pdf_path, page_num + 1)
            if table_chunks:
                for i, chunk in enumerate(table_chunks):
                    # Add remaining necessary fields
                    chunk["doc_id"] = doc_id_counter + len(page_results['text_chunks']) + len(page_results['image_chunks']) + i
                    page_results['table_chunks'].append(chunk)
        
        return page_results

    except Exception as e:
        print(f"❗ เกิดข้อผิดพลาดในการประมวลผลหน้า {page_num + 1}: {e}")
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
        print(f"❗ เกิดข้อผิดพลาดในการบันทึกผลลัพธ์หน้าไปยัง MongoDB: {e}")
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
                doc_id_counter=doc_id_counter,
                pdf_path=PDF_PATH  # 🆕 ส่ง pdf_path สำหรับใช้กับ camelot
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
        print(f"❗ เกิดข้อผิดพลาดใน main pipeline: {e}")
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
