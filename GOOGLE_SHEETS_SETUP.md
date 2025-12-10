# คู่มือการตั้งค่า Google Sheets สำหรับ RAGAS Evaluation

คู่มือนี้จะช่วยคุณตั้งค่า Google Sheets เพื่อรับผลลัพธ์การประเมิน RAGAS อัตโนมัติ

## 📋 สารบัญ

1. [สร้าง Service Account](#1-สร้าง-service-account)
2. [สร้าง Google Spreadsheet](#2-สร้าง-google-spreadsheet)
3. [ตั้งค่า Environment Variables](#3-ตั้งค่า-environment-variables)
4. [ทดสอบการเชื่อมต่อ](#4-ทดสอบการเชื่อมต่อ)

---

## 1. สร้าง Service Account

### ขั้นตอนที่ 1: ไปที่ Google Cloud Console

1. ไปที่ [Google Cloud Console](https://console.cloud.google.com/)
2. สร้างโปรเจกต์ใหม่หรือเลือกโปรเจกต์ที่มีอยู่

### ขั้นตอนที่ 2: เปิดใช้งาน Google Sheets API

1. ไปที่ **APIs & Services** > **Library**
2. ค้นหา "Google Sheets API"
3. คลิก **Enable**

### ขั้นตอนที่ 3: สร้าง Service Account

1. ไปที่ **APIs & Services** > **Credentials**
2. คลิก **Create Credentials** > **Service Account**
3. ตั้งชื่อ Service Account (เช่น: `ragas-evaluation`)
4. คลิก **Create and Continue**
5. ข้ามขั้นตอน Grant access (ไม่จำเป็น)
6. คลิก **Done**

### ขั้นตอนที่ 4: สร้าง Key สำหรับ Service Account

1. คลิกที่ Service Account ที่สร้างไว้
2. ไปที่แท็บ **Keys**
3. คลิก **Add Key** > **Create new key**
4. เลือก **JSON**
5. คลิก **Create** (ไฟล์ JSON จะถูกดาวน์โหลด)

**⚠️ เก็บไฟล์ JSON นี้ไว้อย่างปลอดภัย!**

---

## 2. สร้าง Google Spreadsheet

### ขั้นตอนที่ 1: สร้าง Spreadsheet ใหม่

1. ไปที่ [Google Sheets](https://sheets.google.com/)
2. สร้าง Spreadsheet ใหม่
3. ตั้งชื่อ (เช่น: "RAGAS Evaluation Results")

### ขั้นตอนที่ 2: แชร์ Spreadsheet กับ Service Account

1. คลิกปุ่ม **Share** (มุมขวาบน)
2. คัดลอก **Email address** ของ Service Account (อยู่ในไฟล์ JSON ที่ดาวน์โหลด)
   - ดูที่ field `client_email` ในไฟล์ JSON
   - เช่น: `ragas-evaluation@your-project.iam.gserviceaccount.com`
3. วาง Email address ในช่อง "Add people and groups"
4. ตั้งสิทธิ์เป็น **Editor**
5. คลิก **Send**

### ขั้นตอนที่ 3: คัดลอก Spreadsheet ID

1. ดู URL ของ Spreadsheet
   - เช่น: `https://docs.google.com/spreadsheets/d/SPREADSHEET_ID/edit`
2. คัดลอก `SPREADSHEET_ID` (ส่วนที่อยู่ระหว่าง `/d/` และ `/edit`)

---

## 3. ตั้งค่า Environment Variables

### วิธีที่ 1: ใช้ไฟล์ Credentials (แนะนำ)

1. วางไฟล์ JSON ที่ดาวน์โหลดไว้ในโฟลเดอร์โปรเจกต์ (อย่า commit ลง git!)
2. เพิ่มในไฟล์ `.env`:

```env
# Google Sheets Configuration
GOOGLE_SHEETS_ENABLED=true
GOOGLE_SHEETS_CREDENTIALS_PATH=/Users/ratchadaporn/Desktop/ragas-connect.json
GOOGLE_SHEETS_ID=your_spreadsheet_id_here
GOOGLE_SHEETS_WORKSHEET_NAME=RAGAS Evaluation
GOOGLE_SHEETS_CLEAR_EXISTING=false
```

### วิธีที่ 2: ใช้ JSON String (สำหรับ Production)

1. อ่านเนื้อหาของไฟล์ JSON
2. แปลงเป็น JSON string (escape quotes)
3. เพิ่มในไฟล์ `.env`:

```env
# Google Sheets Configuration
GOOGLE_SHEETS_ENABLED=true
GOOGLE_SHEETS_CREDENTIALS={"type":"service_account","project_id":"...","private_key_id":"...","private_key":"...","client_email":"...","client_id":"...","auth_uri":"...","token_uri":"...","auth_provider_x509_cert_url":"...","client_x509_cert_url":"..."}
GOOGLE_SHEETS_ID=your_spreadsheet_id_here
GOOGLE_SHEETS_WORKSHEET_NAME=RAGAS Evaluation
GOOGLE_SHEETS_CLEAR_EXISTING=false
```

### Environment Variables

| Variable | คำอธิบาย | ตัวอย่าง |
|----------|----------|---------|
| `GOOGLE_SHEETS_ENABLED` | เปิด/ปิดการส่งข้อมูลไป Google Sheets | `true` หรือ `false` |
| `GOOGLE_SHEETS_CREDENTIALS_PATH` | Path ไปยังไฟล์ service account JSON | `./credentials.json` |
| `GOOGLE_SHEETS_CREDENTIALS` | Service account JSON string (ใช้แทนไฟล์) | `{"type":"service_account",...}` |
| `GOOGLE_SHEETS_ID` | Spreadsheet ID | `1a2b3c4d5e6f7g8h9i0j` |
| `GOOGLE_SHEETS_WORKSHEET_NAME` | ชื่อ worksheet (ถ้าไม่มีจะสร้างใหม่) | `RAGAS Evaluation` |
| `GOOGLE_SHEETS_CLEAR_EXISTING` | ล้างข้อมูลเก่าก่อนบันทึกใหม่ | `true` หรือ `false` |

---

## 4. ทดสอบการเชื่อมต่อ

### ติดตั้ง Dependencies

```bash
pip install gspread google-auth
```

### รันการประเมิน RAGAS

```bash
python evaluate_ragas.py
```

### ตรวจสอบผลลัพธ์

1. ไปที่ Google Spreadsheet ที่สร้างไว้
2. ตรวจสอบว่า worksheet "RAGAS Evaluation" มีข้อมูลหรือไม่
3. ข้อมูลควรมีคอลัมน์:
   - Timestamp
   - Question
   - Answer
   - Ground Truth
   - Faithfulness
   - Answer Relevancy
   - Context Precision
   - Context Recall

---

## 🔒 ความปลอดภัย

### ⚠️ สำคัญ: อย่า Commit Credentials!

1. เพิ่มไฟล์ credentials ใน `.gitignore`:
   ```
   *.json
   credentials/
   service-account*.json
   ```

2. ใช้ environment variables แทนการ hardcode credentials

3. สำหรับ Production ใช้ secrets management (เช่น AWS Secrets Manager, Google Secret Manager)

---

## 🐛 แก้ไขปัญหา

### ปัญหา: "Permission denied"

**สาเหตุ**: Service Account ไม่มีสิทธิ์เข้าถึง Spreadsheet

**แก้ไข**:
1. ตรวจสอบว่าแชร์ Spreadsheet กับ Service Account Email แล้ว
2. ตรวจสอบว่า Service Account มีสิทธิ์เป็น Editor

### ปัญหา: "Spreadsheet not found"

**สาเหตุ**: Spreadsheet ID ไม่ถูกต้อง

**แก้ไข**:
1. ตรวจสอบ `GOOGLE_SHEETS_ID` ใน `.env`
2. ตรวจสอบว่า Spreadsheet ID ถูกต้อง (อยู่ระหว่าง `/d/` และ `/edit` ใน URL)

### ปัญหา: "ModuleNotFoundError: No module named 'gspread'"

**แก้ไข**:
```bash
pip install gspread google-auth
```

---

## 📊 ตัวอย่างข้อมูลใน Google Sheets

| Timestamp | Question | Answer | Ground Truth | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|-----------|----------|--------|--------------|--------------|------------------|-------------------|----------------|
| 2024-01-15 10:30:00 | ราศีเมษมีลักษณะนิสัยอย่างไร? | ราศีเมษเป็นราศี... | ราศีเมษมีลักษณะ... | 0.85 | 0.92 | 0.78 | 0.88 |
| 2024-01-15 10:31:00 | ... | ... | ... | ... | ... | ... | ... |

---

## 📝 หมายเหตุ

- ข้อมูลจะถูกบันทึกทุกครั้งที่รันการประเมิน RAGAS
- ถ้า `GOOGLE_SHEETS_CLEAR_EXISTING=true` ข้อมูลเก่าจะถูกลบก่อนบันทึกใหม่
- ถ้า `GOOGLE_SHEETS_CLEAR_EXISTING=false` ข้อมูลใหม่จะถูกเพิ่มต่อท้าย
- Summary row จะถูกเพิ่มที่ท้ายสุดของข้อมูล

---

## 🔗 ลิงก์ที่เกี่ยวข้อง

- [Google Sheets API Documentation](https://developers.google.com/sheets/api)
- [gspread Documentation](https://docs.gspread.org/)
- [Google Cloud Service Accounts](https://cloud.google.com/iam/docs/service-accounts)
