# import os
# from typing import List, Dict, Any
# from dotenv import load_dotenv
# import openai


# load_dotenv()
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# if not OPENAI_API_KEY:
#     raise ValueError("[ERROR] OPENAI_API_KEY not found in environment variables. Add it to your .env file.")

# openai.api_key = OPENAI_API_KEY


# def synthesize_answer(query: str, docs: List[str]) -> Dict[str, Any]:
#     """
#     Synthesize an answer using retrieved docs and OpenAI GPT-4.
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

#     try:
#         response = openai.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[
#                 {"role": "system", "content": "You are an intelligent assistant that answers questions based on provided context."},
#                 {"role": "user", "content": prompt}
#             ],
#             temperature=0.2,
#             max_tokens=500
#         )
#         answer = response.choices[0].message.content.strip()
#     except Exception as e:
#         print(f"[WARN] OpenAI API call failed. Error: {e}")
#         # Fallback template
#         answer = f"Answer for: {query}\n\n" + "\n---\n".join(d[:300] for d in docs[:2])

#     citations = [f"doc://{i}" for i in range(len(docs[:2]))]
#     follow_ups = ["Would you like more details?", "Do you want troubleshooting steps?"]

#     return {
#         "answer": answer,
#         "citations": citations,
#         "assets": [],
#         "follow_ups": follow_ups,
#     }


import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI

# Load environment
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("[ERROR] OPENAI_API_KEY not found in environment variables. Add it to your .env file.")

client = OpenAI(api_key=OPENAI_API_KEY)


def synthesize_answer(query: str, docs: List[str]) -> Dict[str, Any]:
    """
    Synthesize an answer using retrieved docs and OpenAI GPT-4o-mini.
    Returns a dict with answer, citations, and follow-ups.
    """
    if not docs:
        # No retrieved docs, fallback to direct answer
        context = "No relevant documents found."
    else:
        # Combine top docs (limit 1000 chars each)
        context = "\n\n".join(str(d)[:1000] for d in docs[:3])

    prompt = f"""
You are a helpful assistant. Answer the user's question using only the context below.

Question:
{query}

Context:
{context}

Rules:
- If the context does not contain the answer, reply with:
  "I don’t know. Please check official documentation."
- Be concise and clear.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an intelligent assistant that answers based on provided context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=400
        )

        # ✅ Extract answer correctly
        answer = response.choices[0].message.content.strip()

    except Exception as e:
        print(f"[WARN] OpenAI API call failed: {e}")
        answer = f"I couldn’t generate a precise answer for '{query}'."
    
    # Build citations and follow-ups
    citations = [f"doc://{i}" for i in range(len(docs[:2]))]
    follow_ups = [
        "Would you like more technical details?",
        "Do you want troubleshooting steps for this?"
    ]

    return {
        "answer": answer,
        "citations": citations,
        "assets": [],
        "follow_ups": follow_ups,
    }
