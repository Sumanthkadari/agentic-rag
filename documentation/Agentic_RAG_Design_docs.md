# Agentic RAG System — Lucid Motors & Wells Fargo

## 3. Design Choices

### 3.1 Retrieval Strategy

The system uses **Hybrid Retrieval**:

- **Semantic Search:** SentenceTransformer (`all-MiniLM-L6-v2`)
- **Keyword Search:** BM25 (via `rank_bm25`)
- Results are merged, deduplicated, and ranked to ensure contextual and lexical relevance.

**Vector Database:**  
- **ChromaDB** (Persistent Client)  
- Documents stored with embeddings, text metadata, and unique IDs.

### 3.2 Orchestration Agents

| Agent | Responsibility |
|-------|----------------|
| **Orchestrator** | Manages workflow, applies safety filters, runs sub-agents. |
| **Guardrails** | Redacts PII, blocks unsafe queries (fraud, hacking, etc.). |
| **Router** | Classifies query intent (FAQ, troubleshoot, procedural, multi-step). |
| **Planner** | Decomposes compound queries using regex-based splitting. |
| **Retriever** | Performs hybrid retrieval from ChromaDB and BM25. |
| **Synthesizer** | Generates grounded responses using LLaMA-2 via Hugging Face API. |
| **Validator** | Checks for missing citations or PII leaks before finalizing response. |

---

## 4. Pipeline Flow

1. **User Query Input**
2. **Guardrails:**
   - Redact emails, SSNs, phone numbers
   - Reject blocked terms (e.g., “hack bank”, “malware”)
3. **Intent Routing:**
   Determines whether the query is:
   - FAQ (single factual lookup)
   - Troubleshoot (diagnostic)
   - Procedural (step-by-step)
   - Multi-step (requires decomposition)
4. **Planner/Decomposer:**
   Breaks compound queries into independent sub-queries.
5. **Retriever:**
   - Uses both BM25 + semantic retrieval.
   - Fetches top-k relevant document chunks.
6. **Synthesizer:**
   - Sends context + question to **LLaMA-2** model through Hugging Face API.
   - If API fails, falls back to rule-based summarization.
7. **Validator:**
   - Ensures grounding and citation presence.
   - Rejects answers containing any PII.
8. **Final Output:**
   ```json
   {
     "answer": "Generated grounded response",
     "citations": ["doc://1", "doc://2"],
     "assets": [],
     "follow_ups": ["Would you like more details?", "Do you want troubleshooting steps?"]
   }
   ```

---

## 5. Components

### 5.1 Data Ingestion (`chunker.py`)
- Loads PDFs using `PyPDF2`
- Splits content using `RecursiveCharacterTextSplitter`
- Exports `chunks.txt`

### 5.2 Embedding Indexing
- Embeds chunks using **SentenceTransformer**
- Stores embeddings + metadata into **ChromaDB**
- Supports reloading without recomputation

### 5.3 Retriever (`retriever.py`)
Implements:
- `semantic_search()`
- `bm25_search()`
- `hybrid_search()`  
with configurable `top_k` ranking.

### 5.4 Orchestration & Agents
- Each agent is modular, testable, and invoked in sequence.
- The orchestrator loops through sub-queries, validates responses, and merges citations.

### 5.5 Synthesizer (`synthesizer.py`)
- Uses **LLaMA 2 (7B Chat)** via Hugging Face Inference API.
- Prompts include explicit grounding instructions and fallback logic.

---

## 6. Guardrails & Safety

### PII Detection
Regex-based detection for:
- Email (`[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+`)
- SSN (`\d{3}-\d{2}-\d{4}`)
- Phone numbers (international format)

### Unsafe Query Filtering
Blocked keywords: `bomb`, `fraud`, `malware`, `hack bank`

### Hallucination Control
- The LLM must answer **only using context**.
- If not found → “I don’t know. Please check official documentation.”

---

## 7. Evaluation Plan

| Metric | Description | Method |
|--------|--------------|--------|
| **Grounding Accuracy** | How factually correct is the answer based on retrieved context | Manual verification |
| **Relevance Score** | How well retrieval matches intent | Cosine similarity on embeddings |
| **PII Safety** | Whether personal data is redacted | Regex checks |
| **Latency** | End-to-end response time | Logged timestamps |
| **Completeness** | Whether multi-step queries are fully answered | Sub-query comparison |
