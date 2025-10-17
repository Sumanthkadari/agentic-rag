import re

PII_PATTERNS = {
    "email": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "phone": re.compile(r"\+?\d{1,3}[-.\s]?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b"),
}

BLOCKLIST = ["bomb", "fraud", "malware", "hack bank"]

def redact_pii(text: str) -> str:
    for name, pat in PII_PATTERNS.items():
        text = pat.sub(f"[REDACTED_{name.upper()}]", text)
    return text

def safety_filter(query: str) -> bool:
    return not any(bad in query.lower() for bad in BLOCKLIST)