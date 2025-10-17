def route_intent(query: str) -> str:
    q = query.lower()
    if any(w in q for w in ["how do i", "steps", "procedure", "reset", "install"]):
        return "procedural"
    if any(w in q for w in ["error", "issue", "not working", "problem", "troubleshoot"]):
        return "troubleshoot"
    if " and " in q or "," in q or ";" in q or len(q.split()) > 20:
        return "multi-step"
    return "faq"