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

# RAG system
from app.birth_date_parser import generate_birth_chart_prediction


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
            # ใช้เส้นทางเดียวกับระบบจริง: generate_birth_chart_prediction
            # ซึ่งภายในจะเรียก ask_question_to_rag พร้อม chart_info และ logic เต็ม
            rag_answer = generate_birth_chart_prediction(message=question, user_id=user_id)
        except Exception as e:
            print(f"❌ เกิดข้อผิดพลาดระหว่างเรียก generate_birth_chart_prediction: {e}")
            rag_answer = ""

        questions.append(question)
        answers.append(rag_answer or "")
        ground_truths.append(gt or "")
        # สำหรับ RAGAS ให้ context เป็น list ของหนึ่งข้อความ (จาก dataset)
        contexts.append([ctx] if isinstance(ctx, str) else [str(ctx)])

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

    # สร้าง HuggingFace Dataset สำหรับ Ragas
    hf_dataset = HFDataset.from_pandas(df)

    print("\n📊 เริ่มประเมินด้วย RAGAS ...")
    result = evaluate(
        hf_dataset,
        metrics=[
            answer_relevancy,
            faithfulness,
            context_precision,
            context_recall,
        ],
    )

    # บันทึกผลลัพธ์
    out_csv = os.path.join(os.path.dirname(__file__), "ragas_evaluation_results.csv")
    out_json = os.path.join(os.path.dirname(__file__), "ragas_summary.json")

    print(f"\n💾 บันทึกผลรายข้อไปที่ {out_csv}")
    result_df = result.to_pandas()
    result_df.to_csv(out_csv, index=False)

    print(f"💾 บันทึกสรุปค่าเฉลี่ยและผลรายข้อไปที่ {out_json}")
    # summary จริงจาก ragas (อาจมี NaN ได้ตามการคำนวณ)
    summary = {metric: float(score) for metric, score in result.items()}

    # เตรียมโครงสร้างสำหรับ JSON: summary + per-example results (ไม่ดัดแปลงคะแนน)
    metric_cols = [c for c in result_df.columns if c not in ("question", "answer", "ground_truth", "contexts")]
    detailed_results = []
    for idx, row in result_df.iterrows():
        # ทำ contexts ให้ serialize ได้แน่นอน (list[str])
        raw_ctx = row.get("contexts", [])
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
                "question": row.get("question", ""),
                "ground_truth": row.get("ground_truth", ""),
                "answer": row.get("answer", ""),
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

    print("\n✅ เสร็จสิ้นการประเมิน RAGAS")
    print("ผลสรุป (ค่าเฉลี่ย):")
    for metric, score in summary.items():
        print(f"- {metric}: {score:.4f}")

    # แสดงผลรายข้อแบบสั้นๆ ในเทอร์มินัลด้วย (จาก result_df)
    print("\n📋 ผลรายข้อ (ตัวอย่าง):")
    cols = [c for c in result_df.columns if c not in ("question", "answer", "ground_truth", "contexts")]
    for idx, row in result_df.iterrows():
        q = str(row.get("question", ""))[:60].replace("\n", " ")
        metrics_str = ", ".join(f"{m}={row[m]:.4f}" for m in cols if m in row and pd.notna(row[m]))
        print(f"[{idx}] {q} ... | {metrics_str}")


if __name__ == "__main__":
    evaluate_with_ragas_main()
