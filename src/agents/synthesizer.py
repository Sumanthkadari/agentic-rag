# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# import torch
# from typing import List, Dict, Any
# import os
# import requests
# from dotenv import load_dotenv


# # Load environment variables
# load_dotenv()
# HF_TOKEN = os.getenv("HF_TOKEN")

# if not HF_TOKEN:
#     raise ValueError("[ERROR] HF_TOKEN not found in environment variables. Add it to your .env file.")

# # Use LLaMA 2 model (public) instead of Mistral
# MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
# API_URL = f"https://api-inference.huggingface.co/models/{MODEL_NAME}"
# HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}


# def _query_hf_api(prompt: str) -> str:
#     """
#     Call Hugging Face Inference API to generate text.
#     Returns the generated answer or None if failed.
#     """
#     try:
#         response = requests.post(
#             API_URL,
#             headers=HEADERS,
#             json={
#                 "inputs": prompt,
#                 "parameters": {"max_new_tokens": 400, "temperature": 0.2},
#             },
#             timeout=30  # timeout in seconds
#         )
#         response.raise_for_status()
#         result = response.json()

#         # HF API response can vary
#         if isinstance(result, list) and "generated_text" in result[0]:
#             return result[0]["generated_text"]
#         elif isinstance(result, dict) and "error" in result:
#             raise RuntimeError(result["error"])
#         else:
#             return str(result)

#     except Exception as e:
#         print(f"[WARN] Hugging Face API call failed. Error: {e}")
#         return None


# def synthesize_answer(query: str, docs: List[str]) -> Dict[str, Any]:
#     """
#     Synthesize an answer using retrieved docs and Hugging Face API.
#     Falls back to a template-based summary if API call fails.
#     """
#     # Combine top 3 docs (limit 1000 chars each)
#     context = "\n\n".join(d[:1000] for d in docs[:3])

#     prompt = f"""
# You are a helpful assistant. Answer the user's question using only the context below.

# Question:
# {query}

# Context:
# {context}

# Rules:
# - If the context does not contain the answer, say:
#   "I don’t know. Please check official documentation."
# - Be concise and clear.
# """

#     # Attempt generation via HF API
#     answer = _query_hf_api(prompt)

#     # Fallback to template-based answer
#     if not answer:
#         answer = f"Answer for: {query}\n\n" + "\n---\n".join(d[:300] for d in docs[:2])

#     citations = [f"doc://{i}" for i in range(len(docs[:2]))]
#     follow_ups = ["Would you like more details?", "Do you want troubleshooting steps?"]

#     return {
#         "answer": answer.strip(),
#         "citations": citations,
#         "assets": [],
#         "follow_ups": follow_ups,
#     }



# import os
# import requests
# from typing import List, Dict, Any
# from dotenv import load_dotenv

# # -------------------------
# # Load HF Token
# # -------------------------
# load_dotenv()
# HF_TOKEN = os.getenv("HF_TOKEN")
# print(HF_TOKEN)

# if not HF_TOKEN:
#     raise ValueError("[ERROR] HF_TOKEN not found in environment variables. Add it to your .env file.")

# # -------------------------
# # Model Configuration
# # -------------------------
# MODEL_NAME = "meta-llama/Llama-2-7b-hf"
# API_URL = f"https://api-inference.huggingface.co/models/{MODEL_NAME}"
# # API_URL=f"https://huggingface.co/meta-llama/Llama-2-7b-hf"
# HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

# # -------------------------
# # HF API Query Function
# # -------------------------
# def _query_hf_api(prompt: str, max_tokens: int = 400, temperature: float = 0.2) -> str:
#     """
#     Call Hugging Face Inference API to generate text.
#     """
#     payload = {
#         "inputs": prompt,
#         "parameters": {"max_new_tokens": max_tokens, "temperature": temperature}
#     }
#     try:
#         response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=60)
#         response.raise_for_status()
#         result = response.json()

#         if isinstance(result, list) and "generated_text" in result[0]:
#             return result[0]["generated_text"]
#         elif isinstance(result, dict) and "error" in result:
#             raise RuntimeError(result["error"])
#         else:
#             return str(result)
#     except Exception as e:
#         print(f"[WARN] Hugging Face API call failed: {e}")
#         return None

# # -------------------------
# # Synthesize Answer
# # -------------------------
# def synthesize_answer(query: str, docs: List[str]) -> Dict[str, Any]:
#     """
#     Synthesize an answer using retrieved docs and Hugging Face Llama 2 API.
#     Falls back to a template-based summary if API call fails.
#     """
#     # Combine top 3 docs (limit 1000 chars each)
#     context = "\n\n".join(d[:1000] for d in docs[:3])

#     prompt = f"""
# You are a helpful assistant. Answer the user's question using only the context below.

# Question:
# {query}

# Context:
# {context}

# Rules:
# - If the context does not contain the answer, say:
#   "I don’t know. Please check official documentation."
# - Be concise and clear.
# """

#     # Attempt generation via HF API
#     answer = _query_hf_api(prompt)

#     # Fallback template
#     if not answer:
#         answer = f"Answer for: {query}\n\n" + "\n---\n".join(d[:300] for d in docs[:2])

#     citations = [f"doc://{i}" for i in range(len(docs[:2]))]
#     follow_ups = ["Would you like more details?", "Do you want troubleshooting steps?"]

#     return {
#         "answer": answer.strip(),
#         "citations": citations,
#         "assets": [],
#         "follow_ups": follow_ups,
#     }

import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_NAME = "meta-llama/Llama-2-7b-hf"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    token=HF_TOKEN,
    torch_dtype=torch.float16,   # or float32 if CPU
    device_map="auto"            # requires accelerate
)
