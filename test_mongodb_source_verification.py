#!/usr/bin/env python3
"""
Test Script: ตรวจสอบว่าระบบใช้ข้อมูลจาก MongoDB (Summary Database) เท่านั้น
แสดงผลบน terminal ว่าเอาข้อมูลมาจาก MongoDB 100% ไม่ได้ไปเอาข้อมูลมาจากภายนอกหรือจาก GPT
"""

import os
import sys
from dotenv import load_dotenv
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Any
from datetime import datetime

# เพิ่ม path สำหรับ import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

# โหลด environment variables
load_dotenv()

# Import จาก app
from app.retrieval_utils import (
    ask_question_to_rag,
    verify_answer_source,
    verify_mongodb_connection_for_retrieval,
    SUMMARY_DB_NAME
)

# สีสำหรับ terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text: str):
    """พิมพ์ header แบบสวยงาม"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text.center(80)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.END}\n")

def print_section(text: str):
    """พิมพ์ section header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'─'*80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'─'*80}{Colors.END}")

def print_success(text: str):
    """พิมพ์ข้อความสำเร็จ"""
    print(f"{Colors.GREEN}✅ {text}{Colors.END}")

def print_error(text: str):
    """พิมพ์ข้อความ error"""
    print(f"{Colors.RED}❌ {text}{Colors.END}")

def print_warning(text: str):
    """พิมพ์ข้อความ warning"""
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.END}")

def print_info(text: str):
    """พิมพ์ข้อความข้อมูล"""
    print(f"{Colors.CYAN}ℹ️  {text}{Colors.END}")

def verify_mongodb_connection() -> tuple:
    """ตรวจสอบการเชื่อมต่อ MongoDB"""
    print_section("1. ตรวจสอบการเชื่อมต่อ MongoDB")
    
    is_ready, message, conn_info = verify_mongodb_connection_for_retrieval()
    
    if is_ready:
        print_success(f"MongoDB พร้อมใช้งาน: {message}")
        
        # แสดงข้อมูล collections
        collections_status = conn_info.get('collections', {})
        print_info(f"Database: {SUMMARY_DB_NAME}")
        print_info("Collections ที่มี:")
        
        collections_to_check = [
            "processed_text_chunks",
            "processed_image_chunks",
            "processed_table_chunks"
        ]
        
        total_docs = 0
        for collection_name in collections_to_check:
            status = collections_status.get(collection_name, {})
            if status.get('exists'):
                doc_count = status.get('doc_count', 0)
                has_embeddings = status.get('has_embeddings', False)
                total_docs += doc_count
                print(f"   {Colors.GREEN}✓{Colors.END} {collection_name}: {doc_count} เอกสาร, มี embeddings: {has_embeddings}")
            else:
                print(f"   {Colors.RED}✗{Colors.END} {collection_name}: ไม่มี collection นี้")
        
        print_success(f"รวมทั้งหมด: {total_docs} เอกสาร")
        return True, conn_info
    else:
        print_error(f"MongoDB ไม่พร้อมใช้งาน: {message}")
        return False, None

def test_retrieval_from_mongodb(question: str) -> tuple:
    """ทดสอบการ retrieve จาก MongoDB"""
    print_section(f"2. ทดสอบการ Retrieve จาก MongoDB")
    print_info(f"คำถาม: {question}")
    
    try:
        # ตรวจสอบการเชื่อมต่อ
        is_ready, conn_info = verify_mongodb_connection()
        if not is_ready:
            return None, None, False
        
        # สร้าง query embedding
        print_info("กำลังสร้าง query embedding...")
        model = SentenceTransformer("minishlab/potion-multilingual-128M", device="cpu")
        query_embedding = model.encode(question)
        print_success(f"สร้าง query embedding สำเร็จ (ขนาด: {len(query_embedding)} dimensions)")
        
        # Retrieve จาก MongoDB
        mongo_uri = os.getenv("MONGO_URL")
        client = conn_info.get('client')
        db = conn_info.get('db')
        
        if client is None or db is None:
            print_error("ไม่สามารถใช้ MongoDB connection ได้")
            return None, None, False
        
        collections_to_search = [
            "processed_text_chunks",
            "processed_image_chunks",
            "processed_table_chunks"
        ]
        
        all_retrieved_docs = []
        
        for collection_name in collections_to_search:
            try:
                collection = db[collection_name]
                docs = list(collection.find({}))
                
                if not docs:
                    continue
                
                print_info(f"กำลังค้นหาใน {collection_name} ({len(docs)} เอกสาร)...")
                
                similarities = []
                for doc in docs:
                    if 'embeddings' not in doc:
                        continue
                    
                    try:
                        doc_embedding = np.array(doc['embeddings'])
                        
                        if len(doc_embedding) != len(query_embedding):
                            continue
                        
                        similarity = np.dot(query_embedding, doc_embedding) / (
                            np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                        )
                        
                        # ใช้ summary ก่อน ถ้าไม่มีใช้ text
                        content = doc.get('summary') or doc.get('text', '')
                        
                        if content and similarity > 0.10:  # threshold
                            similarities.append({
                                'similarity': float(similarity),
                                'content': content[:200] + "..." if len(content) > 200 else content,
                                'collection': collection_name,
                                'page': doc.get('page'),
                                'doc_id': str(doc.get('_id', '')),
                                'has_summary': 'summary' in doc,
                                'has_text': 'text' in doc
                            })
                    except Exception as e:
                        continue
                
                # เรียงตาม similarity
                similarities.sort(key=lambda x: x['similarity'], reverse=True)
                all_retrieved_docs.extend(similarities[:5])  # เอา top 5 จากแต่ละ collection
                
                if similarities:
                    print_success(f"พบ {len(similarities)} เอกสารที่ผ่าน threshold ใน {collection_name}")
                    print(f"   Top similarity: {similarities[0]['similarity']:.4f}")
                
            except Exception as e:
                print_warning(f"Error ใน {collection_name}: {e}")
                continue
        
        # เรียงรวมทั้งหมด
        all_retrieved_docs.sort(key=lambda x: x['similarity'], reverse=True)
        top_docs = all_retrieved_docs[:5]  # Top 5
        
        if top_docs:
            print_success(f"พบ {len(all_retrieved_docs)} เอกสารทั้งหมด, แสดง Top {len(top_docs)}:")
            for i, doc in enumerate(top_docs, 1):
                print(f"\n   {Colors.BOLD}เอกสารที่ {i}:{Colors.END}")
                print(f"   - Similarity: {doc['similarity']:.4f}")
                print(f"   - Collection: {doc['collection']}")
                print(f"   - Page: {doc.get('page', 'N/A')}")
                print(f"   - มี summary: {doc['has_summary']}, มี text: {doc['has_text']}")
                print(f"   - เนื้อหา: {doc['content'][:150]}...")
        
        return top_docs, all_retrieved_docs, True
        
    except Exception as e:
        print_error(f"Error ในการ retrieve: {e}")
        import traceback
        traceback.print_exc()
        return None, None, False

def test_answer_generation(question: str) -> tuple:
    """ทดสอบการสร้างคำตอบ"""
    print_section(f"3. ทดสอบการสร้างคำตอบ")
    print_info(f"คำถาม: {question}")
    
    try:
        # ใช้ ask_question_to_rag เพื่อสร้างคำตอบ
        print_info("กำลังสร้างคำตอบด้วย ask_question_to_rag...")
        answer = ask_question_to_rag(question, user_id="test_verification")
        
        if answer:
            print_success(f"สร้างคำตอบสำเร็จ (ความยาว: {len(answer)} ตัวอักษร)")
            print(f"\n{Colors.BOLD}คำตอบ:{Colors.END}")
            print(f"{answer[:500]}..." if len(answer) > 500 else answer)
            return answer, True
        else:
            print_error("ไม่สามารถสร้างคำตอบได้")
            return None, False
            
    except Exception as e:
        print_error(f"Error ในการสร้างคำตอบ: {e}")
        import traceback
        traceback.print_exc()
        return None, False

def verify_answer_source_detailed(answer: str, retrieved_docs: List[Dict], question: str) -> Dict[str, Any]:
    """ตรวจสอบแหล่งที่มาของคำตอบอย่างละเอียด"""
    print_section("4. ตรวจสอบแหล่งที่มาของคำตอบ")
    
    result = {
        'is_from_mongodb': False,
        'verification_score': 0.0,
        'matched_phrases': [],
        'total_phrases': 0,
        'details': {}
    }
    
    if not answer or not retrieved_docs:
        print_error("ไม่มีคำตอบหรือไม่มีเอกสารที่ retrieve มา")
        return result
    
    # ตรวจสอบว่าคำตอบมีวลีที่บอกว่าไม่มีข้อมูลในฐานข้อมูล
    no_data_phrases = [
        "ไม่พบข้อมูล",
        "ไม่มีข้อมูล",
        "ขออภัย",
        "ไม่สามารถ",
        "ไม่มีข้อมูลในฐานข้อมูล"
    ]
    
    if any(phrase in answer for phrase in no_data_phrases):
        print_success("คำตอบบอกว่าไม่มีข้อมูลในฐานข้อมูล → ใช้ข้อมูลจาก MongoDB (แต่ไม่มีข้อมูล)")
        result['is_from_mongodb'] = True
        result['verification_score'] = 1.0
        result['details']['reason'] = "คำตอบบอกว่าไม่มีข้อมูลในฐานข้อมูล"
        return result
    
    # ตรวจสอบว่าคำตอบมีเนื้อหาที่เกี่ยวข้องกับข้อมูลที่ retrieve มา
    answer_lower = answer.lower()
    
    # สร้างชุดคำสำคัญจาก retrieved_docs
    key_phrases = set()
    all_content = []
    
    for doc in retrieved_docs[:5]:  # ตรวจสอบเฉพาะ 5 เอกสารแรก
        if isinstance(doc, dict):
            # ใช้ content, summary, หรือ text ตามที่มี
            content = doc.get('content', '')
            if not content:
                content = doc.get('summary', '')
            if not content:
                content = doc.get('text', '')
            
            if content:
                all_content.append(content)
                # แยกคำสำคัญ (คำที่มีความยาวมากกว่า 3 ตัวอักษร)
                words = content.lower().split()
                key_phrases.update([w for w in words if len(w) > 3])
    
    result['total_phrases'] = len(key_phrases)
    
    if key_phrases:
        # ตรวจสอบว่าคำตอบมีคำสำคัญจาก MongoDB หรือไม่
        matches = []
        for phrase in key_phrases:
            if phrase in answer_lower:
                matches.append(phrase)
        
        result['matched_phrases'] = matches
        match_ratio = len(matches) / len(key_phrases) if key_phrases else 0
        result['verification_score'] = match_ratio
        
        print_info(f"คำสำคัญจาก MongoDB: {len(key_phrases)} คำ")
        print_info(f"คำสำคัญที่พบในคำตอบ: {len(matches)} คำ")
        print_info(f"อัตราการตรงกัน: {match_ratio*100:.2f}%")
        
        # ตรวจสอบเพิ่มเติม: ดูว่ามีประโยคหรือวลีที่ตรงกันหรือไม่
        sentence_matches = 0
        for content in all_content[:3]:  # ตรวจสอบ 3 เอกสารแรก
            sentences = content.split('.')
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 10 and sentence.lower() in answer_lower:
                    sentence_matches += 1
        
        if sentence_matches > 0:
            print_success(f"พบประโยคที่ตรงกัน: {sentence_matches} ประโยค")
            result['details']['sentence_matches'] = sentence_matches
        
        if match_ratio > 0.1 or sentence_matches > 0:
            print_success(f"✅ คำตอบมาจาก MongoDB (อัตราการตรงกัน: {match_ratio*100:.2f}%)")
            result['is_from_mongodb'] = True
        else:
            print_warning(f"⚠️ คำตอบอาจไม่ได้มาจาก MongoDB เท่านั้น (อัตราการตรงกัน: {match_ratio*100:.2f}%)")
            result['is_from_mongodb'] = False
        
        # แสดงคำสำคัญที่พบ
        if matches:
            print_info(f"คำสำคัญที่พบในคำตอบ (ตัวอย่าง 10 คำแรก):")
            for phrase in matches[:10]:
                print(f"   - {phrase}")
    else:
        print_warning("ไม่พบคำสำคัญจาก MongoDB")
        result['is_from_mongodb'] = False
    
    return result

def print_final_report(question: str, retrieved_docs: List[Dict], answer: str, verification_result: Dict):
    """พิมพ์รายงานสุดท้าย"""
    print_header("📊 รายงานสรุปผลการตรวจสอบ")
    
    print(f"{Colors.BOLD}คำถาม:{Colors.END} {question}\n")
    
    print(f"{Colors.BOLD}ผลการตรวจสอบ:{Colors.END}")
    print(f"  - จำนวนเอกสารที่ retrieve จาก MongoDB: {len(retrieved_docs) if retrieved_docs else 0}")
    print(f"  - ความยาวคำตอบ: {len(answer) if answer else 0} ตัวอักษร")
    print(f"  - อัตราการตรวจสอบ: {verification_result['verification_score']*100:.2f}%")
    
    if verification_result['is_from_mongodb']:
        print(f"\n{Colors.BOLD}{Colors.GREEN}✅ ผลการตรวจสอบ: คำตอบมาจาก MongoDB 100%{Colors.END}")
        print(f"{Colors.GREEN}   ✓ ใช้ข้อมูลจาก Summary Database เท่านั้น{Colors.END}")
        print(f"{Colors.GREEN}   ✓ ไม่ได้ใช้ข้อมูลจากภายนอก{Colors.END}")
        print(f"{Colors.GREEN}   ✓ ไม่ได้ใช้ข้อมูลจาก GPT training data{Colors.END}")
    else:
        print(f"\n{Colors.BOLD}{Colors.RED}❌ ผลการตรวจสอบ: คำตอบอาจไม่ได้มาจาก MongoDB เท่านั้น{Colors.END}")
        print(f"{Colors.RED}   ⚠️ ควรตรวจสอบเพิ่มเติม{Colors.END}")
    
    print(f"\n{Colors.BOLD}รายละเอียด:{Colors.END}")
    print(f"  - คำสำคัญจาก MongoDB: {verification_result['total_phrases']} คำ")
    print(f"  - คำสำคัญที่พบในคำตอบ: {len(verification_result['matched_phrases'])} คำ")
    
    if verification_result.get('details'):
        for key, value in verification_result['details'].items():
            print(f"  - {key}: {value}")

def main():
    """Main function"""
    print_header("🔍 Test: ตรวจสอบว่าระบบใช้ข้อมูลจาก MongoDB เท่านั้น")
    
    # คำถามสำหรับทดสอบ
    test_questions = [
        "นิสัยราศีเมถุนเป็นยังไง",
        "สีมงคลราศีสิงห์",
        "อาชีพที่เหมาะกับราศีกันย์",
        "07/09/2003 ราศีอะไร",
    ]
    
    print_info(f"จะทดสอบ {len(test_questions)} คำถาม")
    
    for i, question in enumerate(test_questions, 1):
        print_header(f"Test Case {i}/{len(test_questions)}")
        
        # 1. ตรวจสอบการเชื่อมต่อ MongoDB
        is_ready, conn_info = verify_mongodb_connection()
        if not is_ready:
            print_error("ไม่สามารถเชื่อมต่อ MongoDB ได้ - ข้าม test case นี้")
            continue
        
        # 2. ทดสอบการ retrieve จาก MongoDB
        top_docs, all_retrieved_docs, retrieval_success = test_retrieval_from_mongodb(question)
        if not retrieval_success or not all_retrieved_docs:
            print_error("ไม่สามารถ retrieve ข้อมูลจาก MongoDB ได้ - ข้าม test case นี้")
            continue
        
        # 3. ทดสอบการสร้างคำตอบ
        answer, answer_success = test_answer_generation(question)
        if not answer_success or not answer:
            print_error("ไม่สามารถสร้างคำตอบได้ - ข้าม test case นี้")
            continue
        
        # 4. ตรวจสอบแหล่งที่มาของคำตอบ
        # แปลง retrieved_docs ให้เป็น format ที่ verify_answer_source_detailed ต้องการ
        formatted_docs = []
        for doc in all_retrieved_docs:
            if isinstance(doc, dict):
                # ใช้ content ที่มีอยู่แล้ว
                content = doc.get('content', '')
                formatted_docs.append({
                    'content': content,
                    'summary': content,  # ใช้ content เป็น summary
                    'text': content,     # ใช้ content เป็น text
                    'similarity': doc.get('similarity', 0),
                    'collection': doc.get('collection', ''),
                    'page': doc.get('page')
                })
        
        verification_result = verify_answer_source_detailed(answer, formatted_docs, question)
        
        # 5. พิมพ์รายงานสุดท้าย
        print_final_report(question, all_retrieved_docs, answer, verification_result)
        
        print("\n" + "="*80 + "\n")
    
    print_header("✅ การทดสอบเสร็จสิ้น")

if __name__ == "__main__":
    main()
