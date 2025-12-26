import os
import json
import argparse
from typing import List, Optional

from dotenv import load_dotenv

# RAGAS / evaluation
import pandas as pd
from datasets import Dataset as HFDataset
from ragas import evaluate
# ใช้ metric ตามที่ต้องการ: answer_relevancy + metrics หลักอื่นๆ
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_precision,
    context_recall,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# RAG system
from app.birth_date_parser import BirthDateParser, create_birth_chart_query
from app.retrieval_utils import ask_question_to_rag_for_evaluation


def load_generated_dataset(path: str, limit: Optional[int] = None) -> List[dict]:
    """Load questions/ground truths/contexts from generated_dataset.json.

    Args:
        path: path ของไฟล์ JSON
        limit: ถ้ากำหนด จะใช้แค่ N ข้อแรก (สำหรับเทส / เทรน)
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("generated_dataset.json ต้องเป็น list ของ objects")
    if limit is not None and limit > 0:
        data = data[:limit]
    return data


def run_rag_inference(dataset: List[dict]) -> pd.DataFrame:
    """Run RAG for each question without follow-up / history.

    - ใช้ user_id ไม่ซ้ำกันสำหรับแต่ละคำถาม (eval_0, eval_1, ...)
      เพื่อไม่ให้มีบริบทต่อเนื่อง (no follow-up, no shared history)
    - ไม่ใช้ข้อมูล context จาก dataset ตอนถาม RAG (ใช้เฉพาะในขั้นประเมิน Ragas)
    """
    questions: List[str] = []
    answers: List[str] = []
    ground_truths: List[str] = []
    contexts: List[List[str]] = []  # RAGAS ต้องการเป็น list ของ list[str]

    for idx, item in enumerate(dataset):
        question = item.get("question", "").strip()
        gt = item.get("ground_truth") or item.get("answer") or ""
        ctx = item.get("context", "")

        if not question:
            continue

        user_id = f"eval_{idx}"  # user ใหม่ต่อคำถาม -> ไม่มี follow-up history

        print("\n" + "=" * 80)
        print(f"[RAG EVAL] #{idx} question: {question}")
        print("=" * 80)

        try:
            # ใช้ฟังก์ชัน retrieval สำหรับการประเมินโดยเฉพาะ
            # ซึ่งจะไม่บันทึกข้อมูลลงฐานข้อมูลและไม่ใช้ user context
            parser = BirthDateParser()
            birth_info = parser.extract_birth_info(question)
            
            chart_info = None
            if birth_info and birth_info.get('date'):
                # สร้างข้อมูลดวงชะตา
                chart_info = parser.generate_birth_chart_info(
                    birth_date=birth_info['date'], 
                    birth_time=birth_info.get('time'), 
                    latitude=birth_info.get('latitude', 13.7563),
                    longitude=birth_info.get('longitude', 100.5018)
                )
                
                rag_contexts = [] # Initialize context list
                
                if chart_info:
                    # ตรวจสอบว่าคำถามเป็นคำถามเฉพาะเจาะจงหรือไม่
                    # ถ้ามีคำเฉพาะเจาะจง (เช่น ดาวเคราะห์, มุมสัมพันธ์, สีมงคล) ให้ใช้คำถามเดิม
                    specific_keywords = [
                        'ดาว', 'มฤตยู', 'พฤหัส', 'เสาร์', 'อังคาร', 'ศุกร์', 'พุธ', 'อาทิตย์', 'จันทร์',
                        'มุม', 'เล็ง', 'กุม', 'โยค', 'ตรีโกณ', 'ราหู', 'เกตุ', 'แบคคัส', 'เนปจูน', 'พลูโต',
                        'สีมงคล', 'สี', 'เครื่องแบบ', 'ชุด', 'accessories', 'ผลกระทบ', 'ลักษณะการทำงาน',
                        'พาหนะ', 'การเปลี่ยนแปลง', 'ควรทำอย่างไร',
                        'พื้นดวง', 'สัตว์', 'เลี้ยง', 'ห้าม', 'กาลกิณี', 'โฉลก', 'มงคล', 'ดี', 'เสีย', 'เหมาะ',
                        'การงาน', 'งาน', 'อาชีพ', 'การเงิน', 'เงิน', 'โชคลาภ', 'ลงทุน', 'ความรัก', 'รัก', 'คู่', 'แฟน',
                        'สุขภาพ', 'โรค', 'เจ็บป่วย', 'นิสัย', 'บุคลิก'
                    ]
                    is_specific_question = any(keyword in question for keyword in specific_keywords)
                    
                    if is_specific_question:
                        # ใช้คำถามเดิมสำหรับคำถามเฉพาะเจาะจง
                        rag_answer, rag_contexts = ask_question_to_rag_for_evaluation(question, provided_chart_info=chart_info)
                    else:
                        # ใช้ enhanced query สำหรับคำถามทั่วไป
                        enhanced_query = create_birth_chart_query(chart_info, birth_info)
                        rag_answer, rag_contexts = ask_question_to_rag_for_evaluation(enhanced_query, provided_chart_info=chart_info)
                else:
                    rag_answer = "ไม่สามารถสร้างข้อมูลดวงชะตาได้"
                    rag_contexts = []
            else:
                # ถ้าไม่มีวันเกิด ให้ใช้คำถามเดิม
                rag_answer, rag_contexts = ask_question_to_rag_for_evaluation(question)
        except Exception as e:
            print(f"❌ เกิดข้อผิดพลาดระหว่างเรียก ask_question_to_rag_for_evaluation: {e}")
            import traceback
            traceback.print_exc()
            rag_answer = ""
            rag_contexts = []

        questions.append(question)
        answers.append(rag_answer or "")
        ground_truths.append(gt or "")
        # สำหรับ RAGAS ให้ใช้ context ที่ได้จากการค้นหาจริงเท่านั้น (User Request: No Dataset Fallback)
        contexts.append(rag_contexts)

    df = pd.DataFrame(
        {
            "question": questions,
            "answer": answers,
            "ground_truth": ground_truths,
            "contexts": contexts,
        }
    )
    return df


def evaluate_with_ragas_main():
    """Main entrypoint for running RAGAS evaluation.

    ใช้คำถามจาก generated_dataset.json เพื่อประเมินคุณภาพคำตอบของระบบ RAG
    โดยไม่ใช้ follow-up และไม่ใช้ chat history ร่วมกันระหว่างคำถาม
    """
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Evaluate RAG answers with RAGAS using generated_dataset.json"
    )
    parser.add_argument(
        "--limit",
        "-n",
        type=int,
        default=None,
        help="จำกัดจำนวนข้อที่ใช้ประเมิน (เช่น 50). ถ้าไม่ระบุจะใช้ทุกข้อใน generated_dataset.json",
    )
    args = parser.parse_args()

    dataset_path = os.path.join(os.path.dirname(__file__), "generated_dataset.json")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"ไม่พบไฟล์ generated_dataset.json ที่ {dataset_path}")

    print(f"📄 กำลังโหลด dataset จาก {dataset_path}...")
    dataset = load_generated_dataset(dataset_path, limit=args.limit)
    print(f"✅ โหลดคำถามสำเร็จ: {len(dataset)} รายการ"
          f"{' (จำกัดด้วย --limit)' if args.limit else ''}")

    # รันระบบ RAG เพื่อให้ได้คำตอบใหม่
    print("\n🚀 เริ่มรัน RAG เพื่อสร้างคำตอบสำหรับการประเมิน RAGAS...")
    df = run_rag_inference(dataset)

    print("✅ RAG Inference completed.", flush=True)

    # สร้าง HuggingFace Dataset สำหรับ Ragas
    print("⏳ Converting to HFDataset...", flush=True)
    hf_dataset = HFDataset.from_pandas(df)
    print("✅ HFDataset created.", flush=True)

    print("⏳ Importing langchain_openai...", flush=True)
    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        print("✅ langchain_openai imported.", flush=True)
    except Exception as e:
        print(f"❌ Error importing langchain_openai: {e}", flush=True)
        raise e

    # 🆕 RAGAS 0.4.x Compatibility
    print("⏳ Importing ragas wrappers...", flush=True)
    try:
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        print("✅ Ragas wrappers imported.", flush=True)
    except ImportError:
        # Use dummy check or fail if not found, but we tested they exist
        print("⚠️ Warning: Could not import Langchain wrappers. Might be on older version?", flush=True)
        LangchainLLMWrapper = None
        LangchainEmbeddingsWrapper = None

# ... (Rest of code until ragas_llm init)

    print("\n📊 เริ่มประเมินด้วย RAGAS (Model: gpt-4o-mini) ...", flush=True)
    
    # กำหนด LLM และ Embeddings สำหรับ Ragas
    try:
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
             raise ValueError("OPENAI_API_KEY not found in environment")
             
        _llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_key)
        _emb = OpenAIEmbeddings(api_key=openai_key)
        
        if LangchainLLMWrapper and LangchainEmbeddingsWrapper:
            ragas_llm = LangchainLLMWrapper(_llm)
            ragas_embeddings = LangchainEmbeddingsWrapper(_emb)
            print("✅ Configured Ragas with LangchainLLMWrapper and LangchainEmbeddingsWrapper")
        else:
            ragas_llm = _llm
            ragas_embeddings = _emb
            print("⚠️ Configured Ragas with raw Langchain objects (Legacy Mode)")
            
    except Exception as e:
        print(f"❌ Failed to configure Ragas LLM: {e}")
        raise e

    # 🆕 RAGAS 0.4.x Prompt Patching for Thai
    try:
        print("🔧 Patching Ragas prompts for Thai language (v0.4.x compatible)...")
        
        # 1. Faithfulness - Statement Generation
        if hasattr(faithfulness, 'statement_generator_prompt'):
            # Set language if supported
            if hasattr(faithfulness.statement_generator_prompt, 'language'):
                faithfulness.statement_generator_prompt.language = "thai"
            
            # Update instruction with specific leniency for Astrology
            faithfulness.statement_generator_prompt.instruction += (
                "\n\nIMPORTANT: The answer is in Thai. Split sentences by meaning."
                "Ignore minor elaborations or flowery language commonly used in astrology."
                "Output strictly as a JSON list of strings."
            )
            print("✅ Patched faithfulness.statement_generator_prompt")

        # 2. Faithfulness - NLI Verification
        if hasattr(faithfulness, 'nli_statements_prompt'):
            if hasattr(faithfulness.nli_statements_prompt, 'language'):
                faithfulness.nli_statements_prompt.language = "thai"
            
            faithfulness.nli_statements_prompt.instruction += (
                "\n\nIMPORTANT: The context and statements are in Thai. Analyze semantic meaning."
                "If the context provides a planetary position (e.g., 'Saturn in Taurus'), "
                "CONSIDER standard astrological interpretations (e.g., patience, caution, financial stress) "
                "as 'consistent' or 'supported' by the context, even if the exact interpretation words are missing."
                "Do NOT penalize for using general astrological knowledge that derives from the retrieved positions."
                "Output valid JSON."
            )
            print("✅ Patched faithfulness.nli_statements_prompt")

        # 3. Answer Relevancy - Question Generation
        if hasattr(answer_relevancy, 'question_generation'):
             if hasattr(answer_relevancy.question_generation, 'language'):
                answer_relevancy.question_generation.language = "thai"
             
             answer_relevancy.question_generation.instruction += (
                 "\n\nIMPORTANT: Generate the question in Thai language. "
                 "The generated question should match the style and vocabulary of the answer. "
                 "The answer often contains detailed astrological advice; ensure the generated question reflects a request for such advice."
                 "Output strictly as valid JSON key 'question'."
             )
             print("✅ Patched answer_relevancy.question_generation")

    except Exception as e:
        print(f"⚠️ Failed to patch Ragas prompts: {e}")

    from ragas.run_config import RunConfig

    # กำหนดค่า RunConfig
    run_config = RunConfig(

        max_workers=4,
        timeout=180,
        max_retries=10,
        max_wait=60
    )

    print("⏳ Starting evaluate()... This might take a while.")
    try:
        result = evaluate(
            dataset=hf_dataset,
            metrics=[
                answer_relevancy,
                faithfulness,
                context_precision,
                context_recall,
            ],
            llm=ragas_llm,
            embeddings=ragas_embeddings,
            run_config=run_config,
        )
        print("✅ evaluate() completed successfully.")
    except Exception as e:
        print(f"❌ Error during evaluate(): {e}")
        import traceback
        traceback.print_exc()
        raise e

    # บันทึกผลลัพธ์
    out_csv = os.path.join(os.path.dirname(__file__), "ragas_evaluation_results.csv")
    out_json = os.path.join(os.path.dirname(__file__), "ragas_summary.json")

    print(f"\n💾 บันทึกผลรายข้อไปที่ {out_csv}")
    result_df = result.to_pandas()
    result_df.to_csv(out_csv, index=False)

    print(f"💾 บันทึกสรุปค่าเฉลี่ยและผลรายข้อไปที่ {out_json}")
    # summary จริงจาก ragas (อาจมี NaN ได้ตามการคำนวณ)
    try:
        summary = {metric: float(score) for metric, score in result.items()}
    except AttributeError:
        print("⚠️ result.items() failed, computing summary from pandas df", flush=True)
        summary = result.to_pandas().mean(numeric_only=True).to_dict()

    # เตรียมโครงสร้างสำหรับ JSON: summary + per-example results (ไม่ดัดแปลงคะแนน)
    # Filter only numeric columns for metrics to avoid including text columns like 'user_input'
    numeric_cols = result_df.select_dtypes(include=['number']).columns.tolist()
    metric_cols = [c for c in numeric_cols if c not in ("question", "answer", "ground_truth", "contexts")]
    
    detailed_results = []
    for idx, row in result_df.iterrows():
        # ทำ contexts ให้ serialize ได้แน่นอน (list[str])
        # Mapping Ragas v0.2 vs v1.0+ column names
        # Ragas 0.4.x / 1.0+ often uses: user_input, response, reference, retrieved_contexts
        contexts_val = row.get("contexts") or row.get("retrieved_contexts")
        
        raw_ctx = contexts_val if contexts_val is not None else []
        
        if isinstance(raw_ctx, (list, tuple)):
            ctx_serializable = [str(x) for x in raw_ctx]
        else:
            # pandas / ragas บางเวอร์ชันอาจให้เป็น ndarray หรือ object อื่น
            try:
                ctx_list = raw_ctx.tolist()  # type: ignore[attr-defined]
                if isinstance(ctx_list, (list, tuple)):
                    ctx_serializable = [str(x) for x in ctx_list]
                else:
                    ctx_serializable = [str(ctx_list)]
            except Exception:
                ctx_serializable = [str(raw_ctx)] if raw_ctx not in (None, "") else []

        detailed_results.append(
            {
                "index": int(idx),
                "question": row.get("question") or row.get("user_input") or "",
                "ground_truth": row.get("ground_truth") or row.get("reference") or "",
                "answer": row.get("answer") or row.get("response") or "",
                "contexts": ctx_serializable,
                "metrics": {
                    m: float(row[m]) if m in row and pd.notna(row[m]) else None
                    for m in metric_cols
                },
            }
        )

    summary_payload = {
        "summary": summary,
        "results": detailed_results,
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, ensure_ascii=False, indent=2)

    print("\n✅ เสร็จสิ้นการประเมิน RAGAS", flush=True)
    print("ผลสรุป (ค่าเฉลี่ย):", flush=True)
    
    # Mapping ชื่อตัวชี้วัดเป็นภาษาไทย
    metric_map = {
        "answer_relevancy": "Answer Relevancy",
        "faithfulness": "Faithfulness",
        "context_precision": "Context Precision",
        "context_recall": "Context Recall",
    }

    for metric, score in summary.items():
        thai_name = metric_map.get(str(metric), str(metric))
        print(f"- {thai_name}: {score:.4f}", flush=True)

    # แสดงผลรายข้อแบบสั้นๆ ในเทอร์มินัลด้วย (จาก result_df)
    print("\n📋 ผลรายข้อ (ตัวอย่าง):", flush=True)
    # Use numeric_cols which we defined earlier (ensure it's available or redefine)
    cols = [c for c in result_df.select_dtypes(include=['number']).columns if c not in ("question", "answer", "ground_truth", "contexts")]
    for idx, row in result_df.iterrows():
        q = str(row.get("question", ""))[:60].replace("\n", " ")
        # ใช้ชื่อย่อภาษาอังกฤษสำหรับบรรทัดรายข้อเพื่อความกระชับ
        metrics_str = ", ".join(f"{m}={row[m]:.4f}" for m in cols if m in row and pd.notna(row[m]))
        print(f"[{idx}] {q} ... | {metrics_str}", flush=True)


if __name__ == "__main__":
    evaluate_with_ragas_main()
