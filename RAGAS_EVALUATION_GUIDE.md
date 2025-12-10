# คู่มือการใช้งาน Ragas Evaluation สำหรับแชทบอทโหราศาสตร์

## ภาพรวม

Ragas (Retrieval-Augmented Generation Assessment) เป็น framework สำหรับประเมินประสิทธิภาพของระบบ RAG (Retrieval-Augmented Generation) โดยวัด metrics ต่างๆ ที่สำคัญ

## Metrics ที่ประเมิน

1. **Faithfulness** - คำตอบตรงกับบริบทที่ retrieve หรือไม่
2. **Answer Relevancy** - คำตอบเกี่ยวข้องกับคำถามหรือไม่
3. **Context Precision** - บริบทที่ retrieve มีความเกี่ยวข้องกับคำถามหรือไม่
4. **Context Recall** - บริบทที่ retrieve ครอบคลุมข้อมูลที่จำเป็นหรือไม่

## การติดตั้ง

### 1. ติดตั้ง Dependencies

```bash
pip install ragas datasets
```

หรือใช้ requirements.txt:

```bash
pip install -r requirements.txt
```

### 2. ตั้งค่า Environment Variables

สร้างไฟล์ `.env` และตั้งค่า:

```env
MONGO_URL=mongodb+srv://your-username:your-password@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority
OPENAI_API_KEY=sk-your-openai-api-key
```

## การใช้งาน

### ขั้นตอนที่ 1: สร้าง Dataset จาก MongoDB (แนะนำ)

ใช้ LLM สร้างคำถามและคำตอบจากข้อมูลใน MongoDB โดยเน้นคำถามที่เกี่ยวข้องกับระบบหลัก:

#### วิธีที่ 1: ใช้สคริปต์ Shell

```bash
./run_generate_dataset.sh
```

#### วิธีที่ 2: รันโดยตรง

```bash
python generate_ragas_dataset_from_mongo.py
```

#### พารามิเตอร์ (ตั้งค่าใน .env หรือ environment variables)

```env
# จำนวนเอกสารสูงสุดต่อ collection (default: 50)
MONGO_LIMIT_PER_COLLECTION=50

# จำนวนคำถามต่อ chunk (default: 2)
NUM_QUESTIONS_PER_CHUNK=2

# จำนวนคำถามสูงสุดทั้งหมด (default: 100)
MAX_TOTAL_QUESTIONS=100

# ไฟล์ผลลัพธ์ (default: dataset_from_mongo.json)
OUTPUT_DATASET_FILE=dataset_from_mongo.json
```

#### ตัวอย่างการใช้งาน

```bash
# ใช้ค่า default
python generate_ragas_dataset_from_mongo.py

# หรือตั้งค่าผ่าน environment variables
MONGO_LIMIT_PER_COLLECTION=100 NUM_QUESTIONS_PER_CHUNK=3 MAX_TOTAL_QUESTIONS=200 python generate_ragas_dataset_from_mongo.py
```

**ผลลัพธ์**: จะได้ไฟล์ `dataset_from_mongo.json` ที่มีคำถาม-คำตอบที่สร้างจากข้อมูลใน MongoDB

### ขั้นตอนที่ 2: รันการประเมิน RAGAS

#### วิธีที่ 1: ใช้สคริปต์ Shell

```bash
./run_ragas_evaluation.sh
```

#### วิธีที่ 2: รันโดยตรง

```bash
python evaluate_ragas.py
```

**หมายเหตุ**: `evaluate_ragas.py` จะใช้ `dataset_from_mongo.json` อัตโนมัติถ้ามีอยู่ ถ้าไม่มีจะใช้ `test_dataset.json` แทน

## โครงสร้างไฟล์

```
astrobot/
├── evaluate_ragas.py                    # สคริปต์หลักสำหรับประเมิน
├── generate_ragas_dataset_from_mongo.py # สคริปต์สำหรับสร้าง dataset จาก MongoDB
├── test_dataset.json                     # ข้อมูลทดสอบ (questions และ ground truth)
├── dataset_from_mongo.json               # Dataset ที่สร้างจาก MongoDB (จะถูกสร้างหลังรัน)
├── run_ragas_evaluation.sh               # สคริปต์ shell สำหรับรันการประเมิน
├── run_generate_dataset.sh               # สคริปต์ shell สำหรับสร้าง dataset จาก MongoDB
├── ragas_evaluation_report.json          # รายงานผลการประเมิน (จะถูกสร้างหลังรัน)
└── RAGAS_EVALUATION_GUIDE.md            # คู่มือนี้
```

## รูปแบบ Test Dataset

ไฟล์ `test_dataset.json` ควรมีรูปแบบดังนี้:

```json
[
  {
    "question": "คำถามตัวอย่าง",
    "ground_truth": "คำตอบที่ถูกต้อง",
    "contexts": [
      "บริบทที่เกี่ยวข้อง 1",
      "บริบทที่เกี่ยวข้อง 2"
    ]
  }
]
```

### ตัวอย่าง Test Case

```json
{
  "question": "ราศีเมษมีลักษณะนิสัยอย่างไร",
  "ground_truth": "ราศีเมษ (Aries) เป็นราศีแรกของจักรราศี มีลักษณะนิสัยเด่นชัดคือเป็นผู้นำ มีความกล้าหาญ มุ่งมั่น กระตือรือร้น และมีความเป็นอิสระสูง",
  "contexts": [
    "ราศีเมษเป็นราศีแรกของจักรราศี",
    "ลักษณะนิสัยของราศีเมษคือเป็นผู้นำ มีความกล้าหาญ"
  ]
}
```

## การอ่านรายงาน

หลังจากการประเมินเสร็จสิ้น จะได้ไฟล์ `ragas_evaluation_report.json` ที่มีโครงสร้างดังนี้:

```json
{
  "timestamp": "2024-01-01T00:00:00",
  "summary": {
    "faithfulness": 0.85,
    "answer_relevancy": 0.90,
    "context_precision": 0.75,
    "context_recall": 0.80
  },
  "detailed_results": [
    {
      "question": "...",
      "answer": "...",
      "faithfulness": 0.85,
      "answer_relevancy": 0.90,
      ...
    }
  ]
}
```

### การตีความผลลัพธ์

- **Score 0.8-1.0**: ดีมาก (Excellent)
- **Score 0.6-0.8**: ดี (Good)
- **Score 0.4-0.6**: พอใช้ (Fair)
- **Score 0.0-0.4**: ต้องปรับปรุง (Poor)

## การเพิ่ม Test Cases

### วิธีที่ 1: สร้างจาก MongoDB (แนะนำ)

ใช้สคริปต์ `generate_ragas_dataset_from_mongo.py` เพื่อให้ LLM สร้างคำถาม-คำตอบจากข้อมูลใน MongoDB อัตโนมัติ:

```bash
python generate_ragas_dataset_from_mongo.py
```

**ข้อดี**:
- คำถาม-คำตอบมาจากข้อมูลจริงในระบบ
- ครอบคลุมเนื้อหาที่มีใน MongoDB
- เน้นคำถามที่เกี่ยวข้องกับระบบหลัก (โหราศาสตร์, ราศี, ดวงชะตา, ลัคณา, บ้านทั้ง 12 บ้าน)

### วิธีที่ 2: เพิ่มด้วยตนเอง

1. เปิดไฟล์ `test_dataset.json` หรือ `dataset_from_mongo.json`
2. เพิ่ม test case ใหม่ในรูปแบบ JSON array
3. รันการประเมินใหม่

### ตัวอย่างการเพิ่ม Test Case

```json
{
  "question": "คำถามใหม่",
  "ground_truth": "คำตอบที่ถูกต้อง",
  "contexts": [
    "บริบทที่เกี่ยวข้อง"
  ]
}
```

## การแก้ไขปัญหา

### ปัญหา: ไม่พบ MONGO_URL

**วิธีแก้**: ตรวจสอบว่าไฟล์ `.env` มี `MONGO_URL` ที่ถูกต้อง

### ปัญหา: ไม่พบ OPENAI_API_KEY

**วิธีแก้**: ตรวจสอบว่าไฟล์ `.env` มี `OPENAI_API_KEY` ที่ถูกต้อง

### ปัญหา: ไม่สามารถดึงบริบทได้

**วิธีแก้**: 
1. ตรวจสอบว่า MongoDB มีข้อมูลใน collections `processed_text_chunks`, `processed_image_chunks`, `processed_table_chunks`
2. ตรวจสอบว่าเอกสารมี field `embeddings` และ `summary`
3. รัน `python app/multimodel_rag.py` เพื่อสร้าง embeddings

### ปัญหา: Ragas evaluation ล้มเหลว

**วิธีแก้**:
1. ตรวจสอบว่าได้ติดตั้ง `ragas` และ `datasets` แล้ว
2. ตรวจสอบว่า test dataset มีรูปแบบที่ถูกต้อง
3. ตรวจสอบว่า MongoDB และ OpenAI API ทำงานได้ปกติ

## การปรับปรุงระบบ

### 1. เพิ่ม Metrics

แก้ไขไฟล์ `evaluate_ragas.py` และเพิ่ม metrics ที่ต้องการ:

```python
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    # เพิ่ม metrics ใหม่ที่นี่
    answer_similarity,
    answer_correctness,
)
```

### 2. ปรับแต่ง Threshold

แก้ไข threshold สำหรับ similarity ในฟังก์ชัน `get_retrieved_contexts()`:

```python
# เปลี่ยนจาก top 5 เป็น top 10
top_docs = similarities[:10]
```

### 3. เพิ่ม Custom Metrics

สร้าง custom metrics ตามความต้องการ:

```python
from ragas.metrics.base import Metric

class CustomMetric(Metric):
    # Implement custom metric logic
    pass
```

## Best Practices

1. **ใช้ Test Dataset ที่หลากหลาย**: ครอบคลุมคำถามหลายประเภท
2. **อัปเดต Ground Truth**: ให้แน่ใจว่า ground truth ถูกต้องและเป็นปัจจุบัน
3. **รันการประเมินเป็นประจำ**: เพื่อติดตามประสิทธิภาพของระบบ
4. **วิเคราะห์ Detailed Results**: ดูว่า test case ไหนมีปัญหาและปรับปรุง

## เอกสารอ้างอิง

- [Ragas Documentation](https://docs.ragas.io/)
- [RAG Evaluation Guide](https://docs.ragas.io/concepts/metrics/)

## ติดต่อ

หากมีปัญหาหรือคำถาม กรุณาตรวจสอบ:
1. Logs ใน terminal
2. ไฟล์ `ragas_evaluation_report.json`
3. เอกสาร Ragas

