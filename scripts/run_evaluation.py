import os
import sys
import json
import time
import pandas as pd

# Resolve project root relative path to allow direct execution
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config_loader import load_config, is_api_key_set, get_api_key_env_var_name
from src.database.connection import get_engine, get_session_maker
from src.implementations.embeddings.models import get_embedding_model
from src.implementations.vector_store.stores import get_vector_store
from src.implementations.llm.clients import get_llm_client
from src.implementations.classifiers.intent_classifier import get_intent_classifier
from src.implementations.classifiers.fraud_classifier import get_fraud_classifier
from src.implementations.memory.history import StreamlitSessionMemory
from src.implementations.observability.tracers import ConsoleTracer
from src.implementations.evaluation.evaluators import LLMAsJudgeEvaluator, SemanticEvaluator
from src.pipeline import BankingRAGPipeline

def main():
    print("--- Starting RAG Pipeline Evaluation ---")
    config = load_config()

    # Check if QA pairs exist
    qa_path = "data/04_qa_pairs.json"
    if not os.path.exists(qa_path):
        print(f"Error: Ground truth dataset {qa_path} not found. Run train_models.py first.")
        return

    with open(qa_path, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)

    print(f"Loaded {len(qa_pairs)} evaluation Q&A pairs.")

    # Initialize pipeline dependencies
    engine = get_engine(config["database"]["connection_string"])
    session_maker = get_session_maker(engine)

    # Initialize model interfaces
    embedding_model = get_embedding_model(config)
    vector_store = get_vector_store(config, embedding_model)

    # Initialize LLM Client
    api_key_name = get_api_key_env_var_name(config)
    api_key_set = is_api_key_set(config)

    if not api_key_set:
        print(f"WARNING: {api_key_name} is not set in environment.")
        print("For local evaluation, we will run the pipeline using Mock/Rule-based fallbacks or local models.")

    # Setup Pipeline Components
    try:
        llm_client = get_llm_client(config)
    except Exception as e:
        print(f"Skipping API-based LLM setup: {e}. Fallback to mock LLM responses for validation.")
        class MockLLM:
            def generate_response(self, p, s=None): return "Mock response"
            def generate_structured_json(self, p, s):
                return {
                    "response": "This is a mock response because GEMINI_API_KEY is missing.",
                    "risk_level": "Low",
                    "suggested_action": "None"
                }
        llm_client = MockLLM()

    intent_classifier = get_intent_classifier(config, llm_client)
    fraud_classifier = get_fraud_classifier(config, session_maker)
    memory = StreamlitSessionMemory()
    tracer = ConsoleTracer()

    pipeline = BankingRAGPipeline(
        config=config,
        session_maker=session_maker,
        llm_client=llm_client,
        vector_store=vector_store,
        intent_classifier=intent_classifier,
        fraud_classifier=fraud_classifier,
        memory=memory,
        tracer=tracer
    )

    # Initialize Evaluators
    semantic_evaluator = SemanticEvaluator(embedding_model)

    llm_judge = None
    if api_key_set:
        try:
            llm_judge = LLMAsJudgeEvaluator(llm_client)
            print("LLM-as-a-judge initialized successfully.")
        except Exception as e:
            print(f"Could not load LLM-as-a-judge: {e}")

    results = []

    print("\nEvaluating QA pairs...")
    for idx, qa in enumerate(qa_pairs):
        question = qa["question"]
        ideal_answer = qa["answer"]
        category = qa["category"]
        expected_risk = qa.get("risk_level", "Low")
        expected_action = qa.get("suggested_action", "None")

        print(f"[{idx+1}/{len(qa_pairs)}] Category: {category} | Query: '{question[:40]}...'")

        # Run pipeline (Use a unique session for each QA pair to isolate history)
        eval_session = f"eval_session_{qa['id']}"
        start_t = time.time()
        pipeline_out = pipeline.run(eval_session, question)
        latency = (time.time() - start_t) * 1000

        generated_answer = pipeline_out["response"]
        retrieved_context = "\n\n".join([c['text'] for c in pipeline_out["retrieved_context"]])

        # 1. Evaluate with Semantic Similarity (Local, Free)
        sem_res = semantic_evaluator.evaluate(
            query=question,
            retrieved_context=retrieved_context,
            generated_response=generated_answer,
            ground_truth=ideal_answer
        )

        # 2. Evaluate with LLM judge if key available
        faithfulness = None
        answer_relevance = None
        context_recall = None

        if llm_judge and api_key_set:
            try:
                judge_res = llm_judge.evaluate(
                    query=question,
                    retrieved_context=retrieved_context,
                    generated_response=generated_answer,
                    ground_truth=ideal_answer
                )
                faithfulness = judge_res["faithfulness"]
                answer_relevance = judge_res["answer_relevance"]
                context_recall = judge_res["context_recall"]
            except Exception as e:
                print(f"  Grading error: {e}")

        results.append({
            "id": qa["id"],
            "category": category,
            "query": question,
            "ground_truth": ideal_answer,
            "generated": generated_answer,
            "expected_risk": expected_risk,
            "actual_risk": pipeline_out["risk_level"],
            "expected_action": expected_action,
            "actual_action": pipeline_out["suggested_action"],
            "latency_ms": latency,
            "semantic_similarity": sem_res.get("semantic_similarity", 0.0),
            "faithfulness": faithfulness,
            "answer_relevance": answer_relevance,
            "context_recall": context_recall
        })

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Compute aggregates
    avg_latency = df["latency_ms"].mean()
    avg_similarity = df["semantic_similarity"].mean()

    report_md = f"""# Banking RAG System Evaluation Report

Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overall Metrics
- **Total Test Cases**: {len(df)}
- **Average Latency**: {avg_latency:.2f} ms
- **Average Semantic Similarity (Cosine)**: {avg_similarity:.4f} (1-5 equivalent: {1.0 + (avg_similarity * 4.0):.2f})
"""

    if api_key_set and llm_judge:
        avg_faithfulness = df["faithfulness"].mean()
        avg_relevance = df["answer_relevance"].mean()
        avg_recall = df["context_recall"].mean()

        report_md += f"""- **Average Faithfulness (LLM Judge)**: {avg_faithfulness:.2f} / 5.0
- **Average Answer Relevance (LLM Judge)**: {avg_relevance:.2f} / 5.0
- **Average Context Recall (LLM Judge)**: {avg_recall:.2f} / 5.0
"""

    report_md += "\n## Test Case Detail Breakdown\n\n"
    report_md += "| ID | Category | Query | Expected Risk | Actual Risk | Sem. Sim | Faithfulness | Relevance | Recall |\n"
    report_md += "|---|---|---|---|---|---|---|---|---|\n"

    for _, r in df.iterrows():
        sim_val = f"{r['semantic_similarity']:.2f}"
        faith_val = f"{r['faithfulness']:.1f}" if pd.notna(r['faithfulness']) else "N/A"
        rel_val = f"{r['answer_relevance']:.1f}" if pd.notna(r['answer_relevance']) else "N/A"
        rec_val = f"{r['context_recall']:.1f}" if pd.notna(r['context_recall']) else "N/A"

        query_truncated = r['query'][:30] + "..." if len(r['query']) > 30 else r['query']
        report_md += f"| {r['id']} | {r['category']} | {query_truncated} | {r['expected_risk']} | {r['actual_risk']} | {sim_val} | {faith_val} | {rel_val} | {rec_val} |\n"

    # Save report
    os.makedirs("data", exist_ok=True)
    report_path = "data/evaluation_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"\n--- Evaluation Completed! Report saved to {report_path} ---")
    print(f"Average Latency: {avg_latency:.2f} ms")
    print(f"Average Semantic Similarity: {avg_similarity:.4f}")
    if api_key_set and llm_judge:
        print(f"Average Faithfulness: {df['faithfulness'].mean():.2f}/5.0")
        print(f"Average Answer Relevance: {df['answer_relevance'].mean():.2f}/5.0")
        print(f"Average Context Recall: {df['context_recall'].mean():.2f}/5.0")

if __name__ == "__main__":
    main()
