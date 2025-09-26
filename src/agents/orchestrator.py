import logging
from typing import Dict, Any, Optional

from src.utils.ingestion.retriever import HybridRetriever
from src.agents.guardrails import redact_pii, safety_filter
from src.agents.router import route_intent
from src.agents.planner import decompose_query
from src.agents.synthesizer import synthesize_answer
from src.agents.validator import validate

logger = logging.getLogger("orchestrator")
logging.basicConfig(level=logging.INFO)

class Orchestrator:
    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever

    def run(self, query: str, user_id: Optional[str] = None, max_docs: int = 5) -> Dict[str, Any]:
        if not safety_filter(query):
            return {"answer": "Query blocked due to unsafe content.", "citations": [], "assets": [], "follow_ups": []}

        query = redact_pii(query)
        intent = route_intent(query)

        sub_queries = [query] if intent != "multi-step" else decompose_query(query)

        all_answers, all_citations, all_followups = [], [], []

        for sub in sub_queries:
            docs = self.retriever.hybrid_search(sub, top_k=max_docs)
            if not docs:
                continue
            synth = synthesize_answer(sub, docs)
            if not validate(synth["answer"], synth["citations"]):
                synth["answer"] = "Validation failed. Please refer to official docs."
            all_answers.append(synth["answer"])
            all_citations.extend(synth["citations"])
            all_followups.extend(synth["follow_ups"])

        return {
            "answer": "\n\n".join(all_answers),
            "citations": list(dict.fromkeys(all_citations)),
            "assets": [],
            "follow_ups": list(dict.fromkeys(all_followups)),
        }
