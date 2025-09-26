from src.agents.guardrails import PII_PATTERNS

def validate(answer: str, citations: list) -> bool:
    if not citations:
        return False
    if any(pat.search(answer) for pat in PII_PATTERNS.values()):
        return False
    return True
