"""
Ragas Evaluation Script สำหรับแชทบอทโหราศาสตร์

สคริปต์นี้ใช้ Ragas framework เพื่อประเมินประสิทธิภาพของระบบ RAG
โดยวัด metrics ต่างๆ เช่น:
- Faithfulness: คำตอบตรงกับบริบทที่ retrieve หรือไม่
- Answer Relevancy: คำตอบเกี่ยวข้องกับคำถามหรือไม่
- Context Precision: บริบทที่ retrieve มีความเกี่ยวข้องกับคำถามหรือไม่
- Context Recall: บริบทที่ retrieve ครอบคลุมข้อมูลที่จำเป็นหรือไม่
"""

import os
import json
import sys
import numpy as np
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from datetime import datetime

# โหลด environment variables
load_dotenv()

# Import Ragas
try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    from datasets import Dataset
except ImportError as e:
    print(f"❌ Error importing Ragas: {e}")
    print("กรุณาติดตั้ง Ragas ด้วยคำสั่ง: pip install ragas datasets")
    sys.exit(1)

# Import RAG system
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from app.retrieval_utils import ask_question_to_rag
from app.birth_date_parser import extract_birth_date_from_message

# ตั้งค่า logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_test_dataset_from_google_sheets(
    spreadsheet_id: Optional[str] = None,
    worksheet_name: str = "Dataset"
) -> List[Dict[str, Any]]:
    """
    โหลด test dataset จาก Google Sheets
    
    Args:
        spreadsheet_id: ID ของ Google Spreadsheet (ถ้า None จะใช้จาก GOOGLE_SHEETS_ID)
        worksheet_name: ชื่อ worksheet ที่จะอ่าน
        
    Returns:
        List[Dict]: รายการของ test cases
    """
    try:
        import gspread
        from google.oauth2.service_account import Credentials
        
        # ตรวจสอบ credentials path
        credentials_path = os.getenv("GOOGLE_SHEETS_CREDENTIALS_PATH")
        if not credentials_path or not os.path.exists(credentials_path):
            logger.warning("⚠️ ไม่พบ GOOGLE_SHEETS_CREDENTIALS_PATH")
            return []
        
        # โหลด credentials
        creds = Credentials.from_service_account_file(
            credentials_path,
            scopes=['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
        )
        client = gspread.authorize(creds)
        
        # ตรวจสอบ spreadsheet_id
        if spreadsheet_id is None:
            spreadsheet_id = os.getenv("GOOGLE_SHEETS_ID")
        
        if not spreadsheet_id:
            logger.warning("⚠️ ไม่พบ GOOGLE_SHEETS_ID")
            return []
        
        # แยก Spreadsheet ID จาก URL (ถ้ามี)
        if "/d/" in spreadsheet_id:
            parts = spreadsheet_id.split("/d/")
            if len(parts) > 1:
                spreadsheet_id = parts[1].split("/")[0].split("?")[0].split("#")[0]
        
        logger.info(f"📊 กำลังโหลด dataset จาก Google Sheets: {spreadsheet_id}")
        
        # เปิด spreadsheet
        spreadsheet = client.open_by_key(spreadsheet_id)
        
        # เปิด worksheet
        try:
            worksheet = spreadsheet.worksheet(worksheet_name)
        except gspread.exceptions.WorksheetNotFound:
            logger.error(f"❌ ไม่พบ worksheet: {worksheet_name}")
            return []
        
        # อ่านข้อมูลทั้งหมด
        all_values = worksheet.get_all_values()
        
        if len(all_values) < 2:
            logger.warning("⚠️ ไม่มีข้อมูลใน worksheet")
            return []
        
        # แปลงข้อมูลเป็น list of dicts
        headers = all_values[0]
        data = []
        
        for row in all_values[1:]:
            if not row[0]:  # ข้ามแถวว่าง
                continue
            
            # สร้าง dict จาก headers และ values
            item = {}
            for i, header in enumerate(headers):
                if i < len(row):
                    item[header] = row[i]
            
            # แปลงเป็นรูปแบบที่ evaluate_ragas.py ต้องการ
            test_case = {
                "question": item.get("คำถาม", ""),
                "ground_truth": item.get("คำตอบ (Ground Truth)", ""),
                "contexts": item.get("Contexts", "").split(" | ") if item.get("Contexts") else []
            }
            
            if test_case["question"]:  # มีคำถามเท่านั้น
                data.append(test_case)
        
        logger.info(f"✅ โหลด test dataset จาก Google Sheets สำเร็จ: {len(data)} test cases")
        return data
        
    except ImportError:
        logger.warning("⚠️ ไม่พบ gspread library")
        return []
    except Exception as e:
        logger.error(f"❌ Error loading from Google Sheets: {e}")
        import traceback
        traceback.print_exc()
        return []


def load_test_dataset(file_path: str = "test_dataset.json") -> List[Dict[str, Any]]:
    """
    โหลด test dataset จากไฟล์ JSON
    
    Args:
        file_path: path ไปยังไฟล์ test dataset
        
    Returns:
        List[Dict]: รายการของ test cases
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"✅ โหลด test dataset สำเร็จ: {len(data)} test cases")
        return data
    except FileNotFoundError:
        logger.error(f"❌ ไม่พบไฟล์ {file_path}")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"❌ Error parsing JSON: {e}")
        return []


def get_retrieved_contexts(question: str, user_id: str = "evaluation_user") -> List[str]:
    """
    ดึงบริบทที่ retrieve จากระบบ RAG โดยใช้วิธีเดียวกับ ask_question_to_rag
    
    Args:
        question: คำถาม
        user_id: user ID สำหรับการประเมิน
        
    Returns:
        List[str]: รายการของบริบทที่ retrieve
    """
    try:
        from pymongo import MongoClient
        from sentence_transformers import SentenceTransformer
        import numpy as np
        
        mongo_uri = os.getenv("MONGO_URL")
        if not mongo_uri:
            logger.warning("⚠️ ไม่พบ MONGO_URL ใน environment variables")
            return []
        
        # โหลด embedding model (ใช้ CPU เหมือนใน retrieval_utils)
        model = SentenceTransformer("minishlab/potion-multilingual-128M", device="cpu")
        query_embedding = model.encode(question)
        
        # เชื่อมต่อ MongoDB
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        db = client["astrobot_summary"]
        
        # ค้นหาใน collections ทั้งหมด (เหมือนใน retrieval_utils)
        collections_to_search = [
            "processed_text_chunks",
            "processed_image_chunks",
            "processed_table_chunks",
        ]
        
        all_contexts = []
        
        for collection_name in collections_to_search:
            try:
                collection = db[collection_name]
                
                # ตรวจสอบว่า collection มีข้อมูลหรือไม่
                doc_count = collection.count_documents({})
                if doc_count == 0:
                    continue
                
                # ดึงข้อมูลทั้งหมด
                docs = list(collection.find({}))
                
                if not docs:
                    continue
                
                # คำนวณ similarity scores (ใช้วิธีเดียวกับ retrieval_utils)
                similarities = []
                for doc in docs:
                    if 'embeddings' not in doc:
                        continue
                    
                    try:
                        doc_embedding = np.array(doc['embeddings'])
                        
                        # ตรวจสอบว่า dimensions ตรงกัน
                        if len(doc_embedding) != len(query_embedding):
                            continue
                        
                        # Cosine similarity
                        similarity = np.dot(query_embedding, doc_embedding) / (
                            np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                        )
                        similarities.append((similarity, doc))
                    except Exception:
                        continue
                
                # เรียงตาม similarity score และเลือก top 10 (เพิ่มจาก 5 เป็น 10 เพื่อให้ได้ข้อมูลมากขึ้น)
                similarities.sort(key=lambda x: x[0], reverse=True)
                top_docs = similarities[:10]
                
                # ใช้ threshold ที่ต่ำกว่า (0.05-0.08) เพื่อให้ได้ข้อมูลมากขึ้น
                threshold = 0.05
                
                # ดึง summary หรือ text จาก top documents ที่ผ่าน threshold
                for sim, doc in top_docs:
                    if sim > threshold:
                        # ใช้ text ต้นฉบับก่อน (ยาวกว่า summary) เพื่อให้ข้อมูลครบถ้วน
                        context_text = doc.get("text") or doc.get("summary", "")
                        if context_text and context_text.strip():
                            all_contexts.append(context_text.strip())
                
            except Exception as e:
                logger.warning(f"⚠️ Error searching in {collection_name}: {e}")
                continue
        
        client.close()
        
        # จำกัดจำนวน contexts (ใช้ top 10 จากทุก collections)
        return all_contexts[:10] if all_contexts else []
        
    except Exception as e:
        logger.error(f"❌ Error retrieving contexts: {e}")
        import traceback
        traceback.print_exc()
        return []


def run_rag_evaluation(test_cases: List[Dict[str, Any]], user_id: str = "evaluation_user") -> List[Dict[str, Any]]:
    """
    รันการประเมิน RAG system
    
    Args:
        test_cases: รายการของ test cases
        user_id: user ID สำหรับการประเมิน
        
    Returns:
        List[Dict]: ผลลัพธ์การประเมิน
    """
    results = []
    
    logger.info(f"🚀 เริ่มการประเมิน RAG system ด้วย {len(test_cases)} test cases...")
    
    for i, test_case in enumerate(test_cases, 1):
        question = test_case["question"]
        ground_truth = test_case.get("ground_truth", "")
        expected_contexts = test_case.get("contexts", [])
        
        logger.info(f"\n{'='*60}")
        logger.info(f"📝 Test Case {i}/{len(test_cases)}")
        logger.info(f"คำถาม: {question}")
        logger.info(f"{'='*60}")
        
        try:
            # 1. ดึงคำตอบจากระบบ RAG
            logger.info("🔄 กำลังดึงคำตอบจากระบบ RAG...")
            answer = ask_question_to_rag(question, user_id=user_id)
            logger.info(f"✅ ได้รับคำตอบ (ความยาว: {len(answer)} ตัวอักษร)")
            
            # 2. ดึงบริบทที่ retrieve
            logger.info("🔄 กำลังดึงบริบทที่ retrieve...")
            retrieved_contexts = get_retrieved_contexts(question, user_id)
            logger.info(f"✅ ดึงบริบทได้ {len(retrieved_contexts)} chunks")
            
            # 3. เตรียมข้อมูลสำหรับ Ragas
            result = {
                "question": question,
                "answer": answer,
                "contexts": retrieved_contexts if retrieved_contexts else expected_contexts,
                "ground_truth": ground_truth,
            }
            
            results.append(result)
            
            logger.info(f"✅ ประมวลผล test case {i} เสร็จสิ้น")
            
            # เพิ่ม delay ระหว่าง test cases เพื่อลด rate limiting (1-2 วินาที)
            import time
            if i < len(test_cases):
                time.sleep(1.5)  # รอ 1.5 วินาทีระหว่าง test cases
            
        except Exception as e:
            logger.error(f"❌ Error processing test case {i}: {e}")
            import traceback
            traceback.print_exc()
            
            # เพิ่ม delay แม้ในกรณี error เพื่อไม่ให้ API rate limit
            import time
            time.sleep(2)
            
            # เพิ่มผลลัพธ์ว่างเพื่อไม่ให้การประเมินหยุด
            results.append({
                "question": question,
                "answer": "",
                "contexts": [],
                "ground_truth": ground_truth,
            })
    
    return results


def evaluate_with_ragas(evaluation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    ประเมินผลลัพธ์ด้วย Ragas
    
    Args:
        evaluation_results: ผลลัพธ์จากการรัน RAG evaluation
        
    Returns:
        Dict: ผลลัพธ์การประเมินจาก Ragas
    """
    logger.info(f"\n{'='*60}")
    logger.info("📊 เริ่มการประเมินด้วย Ragas...")
    logger.info(f"{'='*60}\n")
    
    # เตรียมข้อมูลสำหรับ Ragas Dataset
    data = {
        "question": [r["question"] for r in evaluation_results],
        "answer": [r["answer"] for r in evaluation_results],
        "contexts": [r["contexts"] for r in evaluation_results],
        "ground_truth": [r["ground_truth"] for r in evaluation_results],
    }
    
    # เตือนถ้ามีคำตอบว่างเพื่อลดโอกาสที่ metric จะเป็น NaN
    empty_answers = sum(1 for ans in data["answer"] if not str(ans).strip())
    empty_contexts = sum(1 for ctx in data["contexts"] if not ctx)
    if empty_answers or empty_contexts:
        logger.warning(
            f"⚠️ พบคำตอบว่าง {empty_answers} รายการ และ contexts ว่าง {empty_contexts} รายการ "
            "อาจทำให้คะแนนบาง metric เป็น NaN หรือคะแนนต่ำ"
        )

    # สร้าง Dataset
    dataset = Dataset.from_dict(data)
    
    # กำหนด metrics ที่จะประเมิน
    metrics = [
        faithfulness,           # คำตอบตรงกับบริบทหรือไม่
        answer_relevancy,      # คำตอบเกี่ยวข้องกับคำถามหรือไม่
        context_precision,     # บริบทที่ retrieve มีความเกี่ยวข้องหรือไม่
        context_recall,        # บริบทที่ retrieve ครอบคลุมข้อมูลที่จำเป็นหรือไม่
    ]
    
    # รันการประเมิน
    try:
        logger.info("🔄 กำลังรันการประเมิน Ragas...")
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
        )
        
        logger.info("✅ การประเมินเสร็จสิ้น!")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Error during Ragas evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None


def convert_numpy_types(obj):
    """
    แปลง numpy types เป็น Python native types สำหรับ JSON serialization
    
    Args:
        obj: object ที่อาจมี numpy types
        
    Returns:
        object ที่แปลงเป็น Python native types แล้ว
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


def save_evaluation_report(ragas_result: Any, output_file: str = "ragas_evaluation_report.json"):
    """
    บันทึกรายงานการประเมิน
    
    Args:
        ragas_result: ผลลัพธ์จาก Ragas evaluation
        output_file: ไฟล์ที่จะบันทึก
        
    Returns:
        Dict: รายงานที่บันทึกแล้ว
    """
    try:
        # แปลงผลลัพธ์เป็น dictionary
        if hasattr(ragas_result, 'to_pandas'):
            df = ragas_result.to_pandas()
            # แทนค่า NaN ด้วย 0 เพื่อลดปัญหาการเฉลี่ยเป็น NaN
            df = df.fillna(0.0)
            
            # แปลง DataFrame เป็น dictionary และแปลง numpy types
            detailed_results = df.to_dict("records")
            detailed_results = convert_numpy_types(detailed_results)
            
            report = {
                "timestamp": datetime.now().isoformat(),
                "summary": {
                    "faithfulness": float(df["faithfulness"].mean()) if "faithfulness" in df.columns else None,
                    "answer_relevancy": float(df["answer_relevancy"].mean()) if "answer_relevancy" in df.columns else None,
                    "context_precision": float(df["context_precision"].mean()) if "context_precision" in df.columns else None,
                    "context_recall": float(df["context_recall"].mean()) if "context_recall" in df.columns else None,
                },
                "detailed_results": detailed_results,
            }
        else:
            report = {
                "timestamp": datetime.now().isoformat(),
                "result": str(ragas_result),
            }
        
        # แปลง numpy types ใน report ทั้งหมด
        report = convert_numpy_types(report)
        
        # บันทึกลงไฟล์
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ บันทึกรายงานการประเมินลง {output_file}")
        
        # แสดงสรุปผล
        if "summary" in report:
            logger.info("\n" + "="*60)
            logger.info("📊 สรุปผลการประเมิน")
            logger.info("="*60)
            for metric, score in report["summary"].items():
                if score is not None:
                    logger.info(f"  {metric}: {score:.4f}")
            logger.info("="*60)
        
        return report
        
    except Exception as e:
        logger.error(f"❌ Error saving evaluation report: {e}")
        import traceback
        traceback.print_exc()
        return None


def connect_to_google_sheets(credentials_path: Optional[str] = None) -> Optional[Any]:
    """
    เชื่อมต่อ Google Sheets API
    
    Args:
        credentials_path: path ไปยังไฟล์ service account credentials JSON
                         ถ้า None จะใช้จาก environment variable GOOGLE_SHEETS_CREDENTIALS_PATH
                         หรือ GOOGLE_SHEETS_CREDENTIALS (JSON string)
    
    Returns:
        gspread.Client หรือ None ถ้าเชื่อมต่อไม่สำเร็จ
    """
    try:
        import gspread
        from google.oauth2.service_account import Credentials
        
        # ตรวจสอบ credentials path
        if credentials_path is None:
            credentials_path = os.getenv("GOOGLE_SHEETS_CREDENTIALS_PATH")
        
        # ถ้ามี credentials path
        if credentials_path and os.path.exists(credentials_path):
            logger.info(f"📁 ใช้ credentials จากไฟล์: {credentials_path}")
            creds = Credentials.from_service_account_file(
                credentials_path,
                scopes=['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
            )
            client = gspread.authorize(creds)
            logger.info("✅ เชื่อมต่อ Google Sheets สำเร็จ (ใช้ service account file)")
            return client
        
        # ถ้ามี credentials เป็น JSON string ใน environment variable
        credentials_json = os.getenv("GOOGLE_SHEETS_CREDENTIALS")
        if credentials_json:
            logger.info("📁 ใช้ credentials จาก environment variable")
            import json as json_lib
            creds_info = json_lib.loads(credentials_json)
            creds = Credentials.from_service_account_info(
                creds_info,
                scopes=['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
            )
            client = gspread.authorize(creds)
            logger.info("✅ เชื่อมต่อ Google Sheets สำเร็จ (ใช้ service account JSON)")
            return client
        
        logger.warning("⚠️ ไม่พบ Google Sheets credentials")
        logger.info("💡 ตั้งค่า GOOGLE_SHEETS_CREDENTIALS_PATH หรือ GOOGLE_SHEETS_CREDENTIALS ใน .env")
        return None
        
    except ImportError:
        logger.error("❌ ไม่พบ gspread library. กรุณาติดตั้งด้วย: pip install gspread google-auth")
        return None
    except Exception as e:
        logger.error(f"❌ Error connecting to Google Sheets: {e}")
        import traceback
        traceback.print_exc()
        return None


def send_to_google_sheets(
    ragas_result: Any,
    spreadsheet_id: Optional[str] = None,
    worksheet_name: str = "RAGAS Evaluation",
    evaluation_results: Optional[List[Dict[str, Any]]] = None
) -> bool:
    """
    ส่งผลลัพธ์การประเมิน RAGAS ไปยัง Google Sheets
    
    Args:
        ragas_result: ผลลัพธ์จาก Ragas evaluation
        spreadsheet_id: ID ของ Google Spreadsheet (ถ้า None จะใช้จาก GOOGLE_SHEETS_ID)
        worksheet_name: ชื่อ worksheet ที่จะบันทึก
        evaluation_results: ผลลัพธ์การประเมิน RAG (optional)
    
    Returns:
        bool: True ถ้าส่งสำเร็จ, False ถ้าส่งไม่สำเร็จ
    """
    try:
        import gspread
        
        # เชื่อมต่อ Google Sheets
        client = connect_to_google_sheets()
        if client is None:
            return False
        
        # ตรวจสอบ spreadsheet_id
        if spreadsheet_id is None:
            spreadsheet_id = os.getenv("GOOGLE_SHEETS_ID")
        
        if not spreadsheet_id:
            logger.error("❌ ไม่พบ GOOGLE_SHEETS_ID ใน environment variables")
            logger.info("💡 ตั้งค่า GOOGLE_SHEETS_ID ใน .env (เช่น: https://docs.google.com/spreadsheets/d/SPREADSHEET_ID/edit)")
            return False
        
        logger.info(f"📊 กำลังส่งข้อมูลไปยัง Google Sheets: {spreadsheet_id}")
        
        # เปิด spreadsheet
        spreadsheet = client.open_by_key(spreadsheet_id)
        
        # ตรวจสอบว่า worksheet มีอยู่หรือไม่ ถ้าไม่มีให้สร้างใหม่
        try:
            worksheet = spreadsheet.worksheet(worksheet_name)
            logger.info(f"✅ พบ worksheet: {worksheet_name}")
        except gspread.exceptions.WorksheetNotFound:
            worksheet = spreadsheet.add_worksheet(title=worksheet_name, rows=1000, cols=20)
            logger.info(f"✅ สร้าง worksheet ใหม่: {worksheet_name}")
        
        # แปลงผลลัพธ์เป็น DataFrame
        if hasattr(ragas_result, 'to_pandas'):
            df = ragas_result.to_pandas()
            df = df.fillna(0.0)
        else:
            logger.warning("⚠️ ไม่สามารถแปลง ragas_result เป็น DataFrame ได้")
            return False
        
        # เตรียมข้อมูลสำหรับบันทึก
        # Header row
        headers = [
            "Timestamp",
            "Question",
            "Answer",
            "Ground Truth",
            "Faithfulness",
            "Answer Relevancy",
            "Context Precision",
            "Context Recall",
        ]
        
        # เพิ่มข้อมูลจาก evaluation_results ถ้ามี
        if evaluation_results:
            # สร้าง mapping ระหว่าง question กับ evaluation result
            eval_map = {r["question"]: r for r in evaluation_results}
            
            # เตรียมข้อมูล rows
            rows = []
            for idx, row in df.iterrows():
                question = row.get("question", "")
                eval_result = eval_map.get(question, {})
                
                row_data = [
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    question,
                    eval_result.get("answer", "")[:500],  # จำกัดความยาว
                    row.get("ground_truth", "")[:500],
                    round(float(row.get("faithfulness", 0.0)), 4),
                    round(float(row.get("answer_relevancy", 0.0)), 4),
                    round(float(row.get("context_precision", 0.0)), 4),
                    round(float(row.get("context_recall", 0.0)), 4),
                ]
                rows.append(row_data)
        else:
            # ใช้ข้อมูลจาก DataFrame เท่านั้น
            rows = []
            for idx, row in df.iterrows():
                row_data = [
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    row.get("question", ""),
                    "",  # answer (ไม่มีใน ragas_result)
                    row.get("ground_truth", "")[:500],
                    round(float(row.get("faithfulness", 0.0)), 4),
                    round(float(row.get("answer_relevancy", 0.0)), 4),
                    round(float(row.get("context_precision", 0.0)), 4),
                    round(float(row.get("context_recall", 0.0)), 4),
                ]
                rows.append(row_data)
        
        # เพิ่ม summary row
        summary_row = [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "=== SUMMARY ===",
            "",
            "",
            round(float(df["faithfulness"].mean()), 4) if "faithfulness" in df.columns else 0.0,
            round(float(df["answer_relevancy"].mean()), 4) if "answer_relevancy" in df.columns else 0.0,
            round(float(df["context_precision"].mean()), 4) if "context_precision" in df.columns else 0.0,
            round(float(df["context_recall"].mean()), 4) if "context_recall" in df.columns else 0.0,
        ]
        
        # ล้างข้อมูลเก่า (ถ้าต้องการ) หรือเพิ่มข้อมูลใหม่
        clear_existing = os.getenv("GOOGLE_SHEETS_CLEAR_EXISTING", "false").lower() == "true"
        
        if clear_existing:
            worksheet.clear()
            logger.info("🗑️ ล้างข้อมูลเก่าใน worksheet")
        
        # บันทึก headers
        worksheet.update(values=[headers], range_name='A1:H1')
        
        # บันทึกข้อมูล
        if rows:
            worksheet.update(values=rows, range_name=f'A2:H{len(rows)+1}')
            logger.info(f"✅ บันทึกข้อมูล {len(rows)} rows")
        
        # บันทึก summary
        summary_start_row = len(rows) + 3
        worksheet.update(values=[summary_row], range_name=f'A{summary_start_row}:H{summary_start_row}')
        worksheet.update(values=[["=== SUMMARY ==="]], range_name=f'A{summary_start_row}')
        logger.info(f"✅ บันทึก summary ที่ row {summary_start_row}")
        
        # Format header row
        worksheet.format('A1:H1', {
            'backgroundColor': {'red': 0.2, 'green': 0.4, 'blue': 0.8},
            'textFormat': {'bold': True, 'foregroundColor': {'red': 1.0, 'green': 1.0, 'blue': 1.0}}
        })
        
        logger.info(f"✅ ส่งข้อมูลไปยัง Google Sheets สำเร็จ!")
        logger.info(f"📊 Spreadsheet: {spreadsheet.url}")
        return True
        
    except ImportError:
        logger.error("❌ ไม่พบ gspread library. กรุณาติดตั้งด้วย: pip install gspread google-auth")
        return False
    except Exception as e:
        logger.error(f"❌ Error sending to Google Sheets: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """ฟังก์ชันหลักสำหรับรันการประเมิน"""
    logger.info("🚀 เริ่มระบบประเมิน Ragas สำหรับแชทบอทโหราศาสตร์")
    logger.info("="*60)
    
    # 1. โหลด test dataset (ลองจาก Google Sheets ก่อน ถ้าไม่มีใช้ JSON)
    google_sheets_enabled = os.getenv("GOOGLE_SHEETS_ENABLED", "false").lower() == "true"
    test_cases = []
    
    if google_sheets_enabled:
        logger.info("📊 กำลังโหลด dataset จาก Google Sheets...")
        test_cases = load_test_dataset_from_google_sheets(worksheet_name="Dataset")
        if test_cases:
            logger.info(f"✅ ใช้ dataset จาก Google Sheets: {len(test_cases)} test cases")
    
    # ถ้าไม่มีข้อมูลจาก Google Sheets หรือไม่ได้เปิดใช้งาน ให้ลองโหลดจาก JSON
    if not test_cases:
        dataset_file = "dataset_from_mongo.json"
        if not os.path.exists(dataset_file):
            dataset_file = "test_dataset.json"
            logger.info(f"📁 ใช้ dataset จากไฟล์: {dataset_file}")
        else:
            logger.info(f"📁 ใช้ dataset จาก MongoDB: {dataset_file}")
        
        test_cases = load_test_dataset(dataset_file)
        if not test_cases:
            logger.error("❌ ไม่มี test cases สำหรับประเมิน")
            logger.info("💡 ใช้คำสั่ง: python3 generate_ragas_dataset_from_mongo.py เพื่อสร้าง dataset จาก MongoDB")
            return
    
    # 2. รันการประเมิน RAG system
    evaluation_results = run_rag_evaluation(test_cases, user_id="ragas_evaluation")
    
    if not evaluation_results:
        logger.error("❌ ไม่มีผลลัพธ์จากการประเมิน")
        return
    
    # 3. ประเมินด้วย Ragas
    ragas_result = evaluate_with_ragas(evaluation_results)
    
    if ragas_result is None:
        logger.error("❌ การประเมิน Ragas ล้มเหลว")
        return
    
    # 4. บันทึกรายงาน
    report = save_evaluation_report(ragas_result, "ragas_evaluation_report.json")
    
    # 5. ส่งข้อมูลไปยัง Google Sheets (ถ้ามีการตั้งค่า)
    google_sheets_enabled = os.getenv("GOOGLE_SHEETS_ENABLED", "false").lower() == "true"
    if google_sheets_enabled:
        logger.info("\n" + "="*60)
        logger.info("📊 กำลังส่งข้อมูลไปยัง Google Sheets...")
        logger.info("="*60)
        success = send_to_google_sheets(
            ragas_result=ragas_result,
            evaluation_results=evaluation_results
        )
        if success:
            logger.info("✅ ส่งข้อมูลไปยัง Google Sheets สำเร็จ!")
        else:
            logger.warning("⚠️ ไม่สามารถส่งข้อมูลไปยัง Google Sheets ได้ (ตรวจสอบ credentials)")
    else:
        logger.info("\n💡 ต้องการส่งข้อมูลไปยัง Google Sheets? ตั้งค่า GOOGLE_SHEETS_ENABLED=true ใน .env")
    
    logger.info("\n✅ การประเมินเสร็จสิ้น!")


if __name__ == "__main__":
    main()

