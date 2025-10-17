from bert_score import score
from typing import List, Tuple

def compute_bert_score(candidates: List[str], references: List[str]) -> Tuple[float, float, float]:
    """
    Compute BERTScore precision, recall, and F1.
    """
    P, R, F1 = score(candidates, references, lang="en", verbose=True)
    return P.mean().item(), R.mean().item(), F1.mean().item()

if __name__ == "__main__":
    preds = ["Lucid Air uses OTA updates for infotainment fixes."]
    refs = ["Lucid Air infotainment system can be reset or updated via OTA."]
    p, r, f1 = compute_bert_score(preds, refs)
    print(f"BERTScore -> Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")
