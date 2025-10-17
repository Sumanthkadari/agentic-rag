import json
from src.eval.bertScore import compute_bert_score
from src.eval.retriever_eval import evaluate_retriever
from src.eval.llm_as_judge import judge_response


def run_all_evaluations(dataset_file="data/eval_dataset.json"):
    """
    Run all evaluations (Retriever, BERTScore, and LLM-as-Judge)
    on a dataset of test queries.
    """
    with open(dataset_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    results = []

    for i, item in enumerate(dataset, 1):
        print(f"\n🧩 Evaluating Sample {i}/{len(dataset)}")
        query = item["query"]
        retrieved_docs = item["retrieved_docs"]
        relevant_docs = item["relevant_docs"]
        generated_answer = item["generated_answer"]
        reference_answer = item["reference_answer"]

        # --- Retriever Metrics ---
        p, r, f1 = evaluate_retriever(retrieved_docs, relevant_docs)
        print(f"Retriever -> Precision: {p:.2f}, Recall: {r:.2f}, F1: {f1:.2f}")

        # --- BERTScore (Answer Similarity) ---
        bp, br, bf1 = compute_bert_score([generated_answer], [reference_answer])
        print(f"BERTScore -> Precision: {bp:.2f}, Recall: {br:.2f}, F1: {bf1:.2f}")

        # --- LLM-as-Judge (Faithfulness / Helpfulness / Completeness) ---
        judge_scores = judge_response(query, " ".join(retrieved_docs[:2]), generated_answer)
        print("LLM Judge:", judge_scores)

        results.append({
            "query": query,
            "retriever": {"precision": p, "recall": r, "f1": f1},
            "bert_score": {"precision": bp, "recall": br, "f1": bf1},
            "llm_judge": judge_scores,
        })

    # Save combined results
    with open("data/eval_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\n✅ Evaluation complete! Results saved to data/eval_results.json.")


if __name__ == "__main__":
    run_all_evaluations()
