import re
from typing import List

def decompose_query(query: str) -> List[str]:
    return [q.strip() for q in re.split(r"\band\b|,|;", query, flags=re.IGNORECASE) if q.strip()]