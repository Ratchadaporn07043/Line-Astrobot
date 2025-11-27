# อธิบายการทำงานของระบบ AstroBot

## ภาพรวมระบบ

ระบบ AstroBot เป็นระบบ RAG (Retrieval-Augmented Generation) สำหรับทำนายดวงชะตาและตอบคำถามเกี่ยวกับโหราศาสตร์ โดยใช้เทคโนโลยี AI และฐานข้อมูล MongoDB

---

## ฝั่ง Admin (การจัดการเอกสาร)

### วัตถุประสงค์
ฝั่ง Admin รับผิดชอบการอัปโหลดและประมวลผลเอกสาร PDF เพื่อสร้างฐานข้อมูลสำหรับระบบ RAG

### เครื่องมือที่ใช้

#### 1. **PyMuPDF (fitz)**
- **หน้าที่**: อ่านและแยกข้อความจาก PDF
- **การทำงาน**: 
  - อ่านข้อความจากแต่ละหน้า
  - ดึงรูปภาพจาก PDF
  - ดึงตำแหน่ง (bbox) ของแต่ละ element

#### 2. **pdfplumber**
- **หน้าที่**: แยกตารางจาก PDF
- **การทำงาน**:
  - ตรวจจับและแยกตารางออกจากหน้า PDF
  - แปลงตารางเป็นข้อความแบบ row-wise (แถวต่อแถว)

#### 3. **EasyOCR**
- **หน้าที่**: แปลงรูปภาพเป็นข้อความ (OCR)
- **การทำงาน**:
  - อ่านข้อความจากรูปภาพใน PDF
  - รองรับภาษาไทยและอังกฤษ
  - ส่งข้อความที่อ่านได้ไปยัง PyThaiNLP เพื่อปรับปรุง

#### 4. **PyThaiNLP**
- **หน้าที่**: ปรับปรุงข้อความไทยจาก OCR
- **การทำงาน**:
  - แก้ไขการเว้นวรรคที่ผิด
  - แก้ไขคำผิดด้วย spell correction
  - แบ่งคำด้วย word tokenization

#### 5. **SentenceTransformer (all-MiniLM-L6-v2)**
- **หน้าที่**: สร้าง embeddings สำหรับข้อความ
- **การทำงาน**:
  - แปลงข้อความเป็น vector 384 dimensions
  - ใช้สำหรับการค้นหาแบบ semantic search

#### 6. **SentenceTransformer (clip-ViT-B-32)**
- **หน้าที่**: สร้าง embeddings สำหรับรูปภาพ
- **การทำงาน**:
  - แปลงรูปภาพเป็น vector
  - ใช้สำหรับการค้นหารูปภาพที่เกี่ยวข้อง

#### 7. **OpenAI GPT-4o-mini**
- **หน้าที่**: สรุปข้อความ (Summarization)
- **การทำงาน**:
  - สรุปข้อความยาวให้กระชับ (ไม่เกิน 3 ประโยค)
  - สร้าง summary สำหรับ text, image, และ table chunks

#### 8. **MongoDB**
- **หน้าที่**: เก็บข้อมูลที่ประมวลผลแล้ว
- **โครงสร้างฐานข้อมูล**:
  - **astrobot_original**: เก็บข้อมูลต้นฉบับ (ไม่มี embeddings และ summary)
    - `original_text_chunks`: ข้อความต้นฉบับ
    - `original_image_chunks`: รูปภาพต้นฉบับ
    - `original_table_chunks`: ตารางต้นฉบับ
  - **astrobot_summary**: เก็บข้อมูลที่ประมวลผลแล้ว (มี embeddings และ summary)
    - `processed_text_chunks`: ข้อความที่มี summary และ embeddings
    - `processed_image_chunks`: รูปภาพที่มี summary, text embeddings, และ image embeddings
    - `processed_table_chunks`: ตารางที่มี summary และ embeddings

### ขั้นตอนการทำงานฝั่ง Admin

1. **อัปโหลด PDF**
   - Admin อัปโหลดไฟล์ PDF ไปยัง `data/attention.pdf`

2. **เริ่มต้นระบบ (Initialization)**
   - เปิดไฟล์ PDF ด้วย PyMuPDF และ pdfplumber
   - โหลด OCR reader (EasyOCR)
   - เชื่อมต่อ MongoDB
   - นับจำนวนหน้าทั้งหมด (`total_pages`)

3. **Loop: ประมวลผลทีละหน้า (More Pages Decision)**
   
   ระบบจะวนลูปประมวลผลทีละหน้าจนกว่าจะครบทุกหน้า โดยมีการตรวจสอบ "More Pages" ในแต่ละรอบ:
   
   ```python
   for page_num in range(total_pages):
       # ประมวลผลหน้าเดียว
       # บันทึกลง MongoDB
       # ตรวจสอบว่ามีหน้าอื่นอีกไหม (More Pages Decision)
       if page_num < total_pages - 1:
           # มีหน้าอื่นอีก → วนลูปต่อ
       else:
           # ไม่มีหน้าอื่นแล้ว → จบการประมวลผล
   ```

   **สำหรับแต่ละหน้า:**
   
   a. **ดึง Elements ทั้งหมด**
      - **ดึง Text Blocks**: ใช้ PyMuPDF ดึงข้อความพร้อมตำแหน่ง (bbox)
      - **ดึง Images**: ใช้ PyMuPDF ดึงรูปภาพพร้อมตำแหน่ง
      - **ดึง Tables**: ใช้ pdfplumber ดึงตารางพร้อมตำแหน่ง
      - **เรียงลำดับ**: เรียง elements ตาม y-coordinate (จากบนลงล่าง) เพื่อประมวลผลตามลำดับที่ปรากฏในหน้า

   b. **ประมวลผลแต่ละ Element ตามลำดับ**
      - **Text**:
        - เก็บข้อความต้นฉบับ → `astrobot_original.original_text_chunks`
        - สร้าง summary ด้วย GPT-4o-mini
        - สร้าง embeddings จาก summary → `astrobot_summary.processed_text_chunks`
      
      - **Image**:
        - แปลงรูปภาพเป็นข้อความด้วย EasyOCR
        - ปรับปรุงข้อความด้วย PyThaiNLP
        - เก็บรูปภาพต้นฉบับ (base64) → `astrobot_original.original_image_chunks`
        - สร้าง summary ด้วย GPT-4o-mini
        - สร้าง text embeddings จาก summary
        - สร้าง image embeddings ด้วย CLIP → `astrobot_summary.processed_image_chunks`
      
      - **Table**:
        - แปลงตารางเป็นข้อความแบบ row-wise
        - เก็บตารางต้นฉบับ → `astrobot_original.original_table_chunks`
        - สร้าง summary ด้วย GPT-4o-mini
        - สร้าง embeddings จาก summary → `astrobot_summary.processed_table_chunks`

   c. **บันทึกลง MongoDB ทันที**
      - บันทึกข้อมูลต้นฉบับลง `astrobot_original` (หน้าแรกจะลบข้อมูลเก่าก่อน)
      - บันทึกข้อมูลที่ประมวลผลแล้วลง `astrobot_summary`
      - **สำคัญ**: บันทึกทีละหน้าเพื่อประหยัด memory และป้องกันข้อมูลสูญหาย

   d. **ตรวจสอบ More Pages**
      - ตรวจสอบว่า `page_num < total_pages - 1`
      - **ถ้ามีหน้าอื่น**: แสดงข้อความ "มีหน้าอื่นอีก X หน้า" และวนลูปต่อ
      - **ถ้าไม่มีหน้าอื่น**: แสดงข้อความ "ประมวลผลและบันทึกครบทุกหน้าแล้ว" และจบการประมวลผล

   e. **ตรวจสอบ Memory**
      - ทุก 5 หน้า ระบบจะตรวจสอบการใช้ memory
      - ถ้าใช้ memory มากกว่า 80% จะทำ garbage collection

4. **สรุปผลการประมวลผล**
   - แสดงจำนวน chunks ทั้งหมดที่ประมวลผล
   - แสดงจำนวน chunks แยกตามประเภท (text, image, table)
   - ปิดการเชื่อมต่อ MongoDB และไฟล์ PDF

---

## ฝั่ง User (การใช้งานระบบ)

### วัตถุประสงค์
ฝั่ง User รับผิดชอบการรับคำถามจากผู้ใช้และสร้างคำตอบที่เหมาะสม

### เครื่องมือที่ใช้

#### 1. **LINE Bot API**
- **หน้าที่**: รับและส่งข้อความผ่าน LINE
- **การทำงาน**:
  - รับข้อความจากผู้ใช้
  - ส่งคำตอบกลับไปยังผู้ใช้

#### 2. **Content Filter**
- **หน้าที่**: ตรวจสอบความปลอดภัยของเนื้อหา
- **การทำงาน**:
  - ตรวจสอบคำหยาบ (ภาษาไทยและอังกฤษ)
  - ตรวจสอบเนื้อหาความรุนแรง
  - ตรวจสอบเนื้อหายาเสพติด
  - ส่งข้อความแจ้งเตือนถ้าพบเนื้อหาไม่เหมาะสม

#### 3. **Birth Date Parser**
- **หน้าที่**: แยกข้อมูลวันเกิด เวลาเกิด และสถานที่เกิดจากข้อความ
- **การทำงาน**:
  - รองรับรูปแบบวันที่หลายแบบ (dd/mm/yyyy, dd-mm-yyyy, ฯลฯ)
  - รองรับชื่อเดือนไทยและอังกฤษ
  - คำนวณราศีจากวันเกิด
  - คำนวณ Ascendant (ลัคณา) จากเวลาเกิดและสถานที่เกิด
  - คำนวณบ้านทั้ง 12 บ้าน

#### 4. **Astronomical Calculator**
- **หน้าที่**: คำนวณข้อมูลดาราศาสตร์
- **การทำงาน**:
  - คำนวณ Ascendant (ราศีประจำลัคนา)
  - คำนวณบ้านทั้ง 12 บ้าน
  - ใช้พิกัดสถานที่เกิด (latitude, longitude)

#### 5. **SentenceTransformer (all-MiniLM-L6-v2)**
- **หน้าที่**: สร้าง embeddings สำหรับคำถาม
- **การทำงาน**:
  - แปลงคำถามเป็น vector
  - ใช้สำหรับการค้นหาแบบ semantic search

#### 6. **MongoDB**
- **หน้าที่**: เก็บข้อมูลผู้ใช้และประวัติการสนทนา
- **โครงสร้างฐานข้อมูล**:
  - **astrobot.user_profiles**: เก็บข้อมูลผู้ใช้ (วันเกิด, ราศี, ฯลฯ)
  - **astrobot.responses**: เก็บประวัติการสนทนา (คำถาม, คำตอบ, embeddings)

#### 7. **OpenAI GPT-4o-mini**
- **หน้าที่**: สร้างคำตอบจากข้อมูลที่ค้นหาได้
- **การทำงาน**:
  - รับคำถามและข้อมูลที่เกี่ยวข้อง
  - สร้างคำตอบที่เหมาะสมตามบริบท

### ขั้นตอนการทำงานฝั่ง User

1. **รับข้อความจากผู้ใช้**
   - ผ่าน LINE Bot หรือ API endpoint `/ask`
   - ตรวจสอบความปลอดภัยของเนื้อหาก่อน

2. **ตรวจสอบความปลอดภัย (Guardrails)**
   - ใช้ Content Filter ตรวจสอบคำหยาบ, ความรุนแรง, ยาเสพติด
   - ถ้าพบเนื้อหาไม่เหมาะสม → ส่งข้อความแจ้งเตือน

3. **แยกข้อมูลวันเกิด**
   - ใช้ Birth Date Parser แยกวันเกิด, เวลาเกิด, สถานที่เกิด
   - ถ้าพบวันเกิด → คำนวณราศี, Ascendant, บ้านทั้ง 12 บ้าน

4. **ตรวจสอบคำถามต่อเนื่อง (Follow-up Detection)**
   - ใช้ Semantic Similarity หรือ LLM ตรวจสอบว่าเป็นคำถามต่อเนื่องหรือไม่
   - เปรียบเทียบคำถามปัจจุบันกับคำถาม/คำตอบก่อนหน้า
   - ถ้าเป็นคำถามต่อเนื่อง → ใช้ข้อมูลราศีจากบริบท

5. **ค้นหาข้อมูล (Retrieval)**
   - สร้าง query embedding จากคำถาม
   - ค้นหาใน `astrobot_summary` ด้วย cosine similarity
   - ดึงข้อมูลที่เกี่ยวข้องจาก:
     - `processed_text_chunks`
     - `processed_image_chunks`
     - `processed_table_chunks`
   - ดึงข้อมูลต้นฉบับจาก `astrobot_original` ถ้าจำเป็น

6. **สร้างคำตอบ (Generation)**
   - ส่งคำถาม + ข้อมูลที่ค้นหาได้ + ข้อมูลดวงชะตา → GPT-4o-mini
   - GPT-4o-mini สร้างคำตอบตาม:
     - ข้อมูลจากฐานข้อมูล (RAG)
     - ข้อมูลดวงชะตา (ราศี, Ascendant, บ้าน)
     - บริบทการสนทนาก่อนหน้า
   - ตอบตามเจตนาของคำถาม (ลักษณะนิสัย, ความรัก, การงาน, การเงิน, สีมงคล)

7. **บันทึกการสนทนา**
   - บันทึกคำถามและคำตอบลง `astrobot.responses`
   - สร้าง embeddings สำหรับคำถามและคำตอบ
   - อัปเดตข้อมูลผู้ใช้ใน `astrobot.user_profiles`

8. **ส่งคำตอบกลับ**
   - ส่งคำตอบกลับไปยังผู้ใช้ผ่าน LINE Bot หรือ API

---

## สรุปความแตกต่างระหว่างฝั่ง Admin และ User

| ด้าน | ฝั่ง Admin | ฝั่ง User |
|------|-----------|----------|
| **วัตถุประสงค์** | สร้างฐานข้อมูล | ใช้งานระบบ |
| **Input** | PDF เอกสาร | คำถามจากผู้ใช้ |
| **Output** | ฐานข้อมูล MongoDB | คำตอบจาก GPT-4o-mini |
| **เครื่องมือหลัก** | PyMuPDF, pdfplumber, EasyOCR, PyThaiNLP | LINE Bot, Birth Date Parser, RAG System |
| **ฐานข้อมูล** | `astrobot_original`, `astrobot_summary` | `astrobot.user_profiles`, `astrobot.responses` |
| **ความถี่** | ครั้งเดียว (เมื่ออัปโหลดเอกสาร) | หลายครั้ง (ทุกครั้งที่ผู้ใช้ถาม) |

---

## Flow Diagram สรุป

### ฝั่ง Admin

```
PDF Upload
    ↓
Initialization (เปิด PDF, โหลด OCR, เชื่อมต่อ MongoDB)
    ↓
┌─────────────────────────────────────────────────┐
│ Loop: ประมวลผลทีละหน้า (More Pages Decision)   │
│                                                  │
│  สำหรับแต่ละหน้า (page_num = 0 ถึง total_pages-1):│
│                                                  │
│  1. ดึง Elements ทั้งหมด                        │
│     ├─ Text Blocks (PyMuPDF)                   │
│     ├─ Images (PyMuPDF)                         │
│     └─ Tables (pdfplumber)                      │
│                                                  │
│  2. เรียงลำดับตาม y-coordinate                  │
│                                                  │
│  3. ประมวลผลแต่ละ Element ตามลำดับ:            │
│     ├─ Text: Summary → Embeddings               │
│     ├─ Image: OCR → PyThaiNLP → Summary →       │
│     │           Text Embeddings + Image Embeddings│
│     └─ Table: Summary → Embeddings               │
│                                                  │
│  4. บันทึกลง MongoDB ทันที                      │
│     ├─ astrobot_original (ต้นฉบับ)              │
│     └─ astrobot_summary (processed)             │
│                                                  │
│  5. ตรวจสอบ More Pages:                         │
│     ├─ if page_num < total_pages - 1:          │
│     │   → มีหน้าอื่นอีก → วนลูปต่อ              │
│     └─ else:                                    │
│         → ไม่มีหน้าอื่นแล้ว → จบการประมวลผล    │
│                                                  │
│  6. ตรวจสอบ Memory (ทุก 5 หน้า)                 │
└─────────────────────────────────────────────────┘
    ↓
สรุปผลการประมวลผล
    ↓
ปิดการเชื่อมต่อและไฟล์
```

**รายละเอียด More Pages Decision:**

การตรวจสอบ "More Pages" ทำงานดังนี้:

1. **ในแต่ละรอบของ loop**: หลังจากประมวลผลและบันทึกหน้าแล้ว ระบบจะตรวจสอบว่า `page_num < total_pages - 1`

2. **ถ้ามีหน้าอื่น** (`page_num < total_pages - 1`):
   - แสดงข้อความ: "➡️ มีหน้าอื่นอีก X หน้า" (X = จำนวนหน้าที่เหลือ)
   - วนลูปต่อเพื่อประมวลผลหน้าถัดไป
   - ตัวอย่าง: ถ้ามี 10 หน้า และกำลังประมวลผลหน้า 5 → แสดง "มีหน้าอื่นอีก 5 หน้า"

3. **ถ้าไม่มีหน้าอื่น** (`page_num == total_pages - 1`):
   - แสดงข้อความ: "✅ ประมวลผลและบันทึกครบทุกหน้าแล้ว (X หน้า)"
   - ออกจาก loop
   - ไปยังขั้นตอนสรุปผลการประมวลผล

**ประโยชน์ของการตรวจสอบ More Pages:**
- **แสดงความคืบหน้า**: ผู้ใช้ทราบว่ามีหน้าอื่นอีกกี่หน้า
- **ป้องกันการประมวลผลซ้ำ**: ตรวจสอบให้แน่ใจว่าประมวลผลครบทุกหน้า
- **จัดการ Memory**: รู้ว่ายังมีหน้าอื่นอีก จึงไม่ต้องรีบทำ cleanup

### ฝั่ง User
```
User Question → Content Filter → Birth Date Parser → Follow-up Detection
            → RAG Retrieval (MongoDB) → GPT-4o-mini → Answer → User
```

---

## หมายเหตุสำคัญ

1. **Embeddings**: ระบบใช้ embeddings ที่สร้างจาก **summary** ไม่ใช่ text ต้นฉบับ เพื่อให้การค้นหาแม่นยำและเร็วขึ้น

2. **Memory Management**: ระบบประมวลผลทีละหน้าและบันทึกทันทีเพื่อประหยัด memory

3. **Context Management**: ระบบเก็บบริบทการสนทนาไว้ใน MongoDB เพื่อตอบคำถามต่อเนื่องได้อย่างถูกต้อง

4. **Guardrails**: ระบบมี Content Filter เพื่อป้องกันเนื้อหาไม่เหมาะสม

5. **Multi-modal**: ระบบรองรับทั้ง text, image, และ table chunks

