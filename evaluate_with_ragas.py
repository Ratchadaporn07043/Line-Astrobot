import json
import pandas as pd
import os
from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness, context_precision, context_recall
from datasets import Dataset
from dotenv import load_dotenv

# Import the retrieval function from the app
# Ensure app is in python path or accessible
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.retrieval_utils import ask_question_to_rag
from pythainlp.tokenize import sent_tokenize
import typing as t
from ragas.metrics._faithfulness import StatementsAnswers, StatementFaithfulnessAnswers
from langchain_openai import ChatOpenAI
import time

# Define Thai Segmenter
class ThaiSegmenter:
    def segment(self, text: str):
        segments = sent_tokenize(text, engine="thaisum")
        # HOSTILE HACK: Ragas hardcodes a check for endsWith(".") in _faithfulness.py
        # We must append proper punctuation to pass this filter.
        return [s + "." for s in segments]


# Load environment variables (for OpenAI key used by Ragas and the App)
load_dotenv()

def main():
    print("Loading generated_dataset.json...")
    try:
        with open("generated_dataset.json", "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: generated_dataset.json not found.")
        return

    # Prepare data structure for Ragas
    ragas_data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }

    print(f"Evaluating {len(data)} items...")
    
    # Limit for testing if needed, but user wants full evaluation probably.
    # The generation might take time and cost money.
    # I will process all items.
    data = data[:10] # TEMPORARY LIMIT FOR VERIFICATION
    
    for i, item in enumerate(data):
        print(f"Processing {i+1}/{len(data)}...")
        
        # Add a small delay to avoid hitting rate limits too quickly during generation
        time.sleep(1)
        
        question = item.get("question")
        # In the dataset, 'ground_truth' is the expected answer.
        ground_truth = item.get("ground_truth")
        
        if not question:
            continue
            
        try:
            # Call the system to get the actual answer and retrieved contexts
            # We use a special flag 'return_retrieved_contexts' added to the function
            system_answer, retrieved_contexts = ask_question_to_rag(
                question=question,
                user_id="eval_script",
                return_retrieved_contexts=True
            )
            
            # Add to dataset
            ragas_data["question"].append(question)
            ragas_data["answer"].append(system_answer)
            # Truncate contexts to top 3 to avoid "context_length_exceeded" (400 Bad Request)
            # Ragas uses these contexts to evaluate faithfulness and relevance.
            # Top 3 should be sufficient and keeps text within token limits.
            ragas_data["contexts"].append(retrieved_contexts[:3]) # List[str]
            ragas_data["ground_truth"].append(ground_truth) # str
            
        except Exception as e:
            print(f"Error processing item {i}: {e}")
            continue

    # Convert to HuggingFace Dataset
    # Ragas expects 'ground_truth' to be the column name for single ground truth or 'ground_truths' for list?
    # Actually, recent ragas versions support 'ground_truth' as string or list.
    # To be safe, let's keep it consistent.
    
    dataset = Dataset.from_dict(ragas_data)
    
    print("Configuring Ragas for Thai language support...")
    # 1. Override Sentence Segmenter for Faithfulness
    faithfulness.sentence_segmenter = ThaiSegmenter()
    
    # 2. Customize Faithfulness Prompts (Decomposition & NLI)
    # Use English for INSTRUCTIONS (better for JSON structure) but Thai for EXAMPLES (better for content)
    faithfulness.statement_prompt.instruction = "จากคำถามและคำตอบที่ให้มา จงแยกประโยคย่อยๆ จากแต่ละประโยคในคำตอบ โดยข้อความต้องเป็นภาษาไทย และตอบกลับเป็น JSON ที่มีคีย์ 'statements' ตามตัวอย่าง"
    faithfulness.statement_prompt.examples = [
        {
            "question": "สีมงคลคืออะไร",
            "answer": "สีมงคลของวันนี้คือสีแดงและสีขาว",
            "sentences": "0:สีมงคลของวันนี้คือสีแดงและสีขาว.",
            "analysis": StatementsAnswers.parse_obj([
                {
                    "sentence_index": 0,
                    "simpler_statements": ["สีมงคลของวันนี้คือสีแดง", "สีมงคลของวันนี้คือสีขาว"]
                }
            ]).dicts()
        }
    ]
    
    faithfulness.nli_statements_message.instruction = "การอนุมานภาษาธรรมชาติ ใช้เฉพาะบริบทที่ให้มาเพื่อตรวจสอบว่าข้อความได้รับการสนับสนุนหรือไม่ ให้จำแนกเป็น 1 (จริง) หรือ 0 (เท็จ)"
    faithfulness.nli_statements_message.examples = [
        {
            "context": "แมวเป็นสัตว์เลี้ยงลูกด้วยนมที่ชอบนอน",
            "statements": '["แมวชอบนอน", "แมวบินได้"]', 
            "analysis": [
                {
                    "statement": "แมวชอบนอน",
                    "verdict": 1,
                    "reason": "บริบทระบุว่าแมวชอบนอน"
                },
                {
                    "statement": "แมวบินได้",
                    "verdict": 0,
                    "reason": "บริบทไม่ได้ระบุว่าแมวบินได้"
                }
            ]
        }
    ]

    # 3. Customize Answer Relevancy Prompt
    answer_relevancy.question_generation.instruction = "จงสร้างคำถามที่เป็นภาษาไทยจากคำตอบที่ให้มา คำถามต้องสั้น กระชับ และตรงประเด็น และต้องตอบกลับเป็น JSON format เท่านั้น โดยมี key คือ 'question' และ 'non_committal' (0 หรือ 1)"
    answer_relevancy.question_generation.examples = [
        {
            "answer": "สีแดงคือสีมงคล",
            "context": "สีแดงเป็นสีแห่งความโชคดี",
            "analysis": {
                "question": "สีอะไรคือสีมงคล",
                "non_committal": 0
            }
        }
    ]
    
    # 4. Customize Context Precision and Recall Prompts
    # Context Precision: Determine if context is relevant to the question
    context_precision.context_precision_prompt.instruction = "จากคำถามและบริบทที่ให้มา (Context) จงระบุว่าบริบทนั้นมีความเกี่ยวข้องและมีประโยชน์ต่อการตอบคำถามหรือไม่ โดยเหตุผล (reason) ต้องเป็นภาษาไทย"
    
    # Context Recall: Determine if ground truth can be found in context
    context_recall.context_recall_prompt.instruction = "จากบริบท (Context) และคำตอบที่ถูกต้อง (Ground Truth) จงวิเคราะห์ว่าแต่ละประโยคในคำตอบที่ถูกต้องนั้น สามารถอ้างอิงข้อมูลจากบริบทได้หรือไม่ โดยเหตุผล (reason) ต้องเป็นภาษาไทย"
    
    # 4. Configure Ragas with Custom LLM (to handle 429s)
    print("Configuring LLM with retry logic...")
    from ragas.llms import LangchainLLMWrapper
    
    # Use ChatOpenAI with explicit retry configuration
    openai_model = ChatOpenAI(
        model="gpt-4o-mini",
        max_retries=10,
        timeout=60,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Wrap with Ragas's LangchainLLMWrapper wrapper
    ragas_llm = LangchainLLMWrapper(langchain_llm=openai_model)

    # Assign the robust LLM to all metrics
    faithfulness.llm = ragas_llm
    answer_relevancy.llm = ragas_llm
    context_precision.llm = ragas_llm
    context_recall.llm = ragas_llm
    
    print("Running Ragas evaluation (this may take a while)...")
    
    from ragas.run_config import RunConfig
    
    # Ragas uses OpenAI by default for metrics. 
    # Make sure OPENAI_API_KEY is in env.
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not found in environment variables. Ragas evaluation might fail.")

    try:
        # Configure run settings to avoid Rate Limits (429)
        run_config = RunConfig(max_workers=1, timeout=120)
        
        results = evaluate(
            dataset,
            metrics=[
                answer_relevancy,
                faithfulness,
                context_precision,
                context_recall,
            ],
            run_config=run_config,
            raise_exceptions=False, # Prevent one failure from crashing the whole run
        )
        
        print("\nEvaluation Results:")
        print(results)
        
        # Save results to CSV (detailed)
        df = results.to_pandas()
        csv_filename = "ragas_evaluation_results.csv"
        df.to_csv(csv_filename, index=False)
        print(f"Detailed results saved to {csv_filename}")
        
        # Save summary to JSON
        with open("ragas_summary.json", "w", encoding="utf-8") as f:
            # results object is dict-like
            json.dump(dict(results), f, indent=2, ensure_ascii=False)
        print("Summary saved to ragas_summary.json")
        
    except Exception as e:
        print(f"Error during Ragas evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
