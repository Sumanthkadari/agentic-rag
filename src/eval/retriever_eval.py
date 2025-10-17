from typing import List, Tuple

def evaluate_retriever(retrieved_docs: List[str], relevant_docs: List[str]) -> Tuple[float, float, float]:
    """
    Compare retrieved vs relevant docs using overlap.
    """
    retrieved_set = set(retrieved_docs)
    relevant_set = set(relevant_docs)
    
    true_pos = len(retrieved_set & relevant_set)
    precision = true_pos / len(retrieved_set) if retrieved_set else 0
    recall = true_pos / len(relevant_set) if relevant_set else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0
    return precision, recall, f1

if __name__ == "__main__":
    retrieved = ["docA", "docB", "docC"]
    relevant = ["docB", "docC", "docD"]
    p, r, f1 = evaluate_retriever(retrieved, relevant)
    print(f"Retriever Eval -> Precision: {p:.2f}, Recall: {r:.2f}, F1: {f1:.2f}")
