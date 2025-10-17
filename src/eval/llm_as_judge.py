from openai import OpenAI
import os
from dotenv import load_dotenv
import json

load_dotenv()

def judge_response(question: str, context: str, answer: str) -> dict:
    """
    Use GPT-4 (or other OpenAI models) to evaluate response quality.
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    prompt = f"""
You are an expert evaluator. Rate the answer on a scale of 1 to 5 for:
1. Faithfulness (Is the answer grounded in context?)
2. Helpfulness (Is it clear, correct, and useful?)
3. Completeness (Does it cover all aspects?)

Question: {question}
Context: {context}
Answer: {answer}

Respond in JSON format:
{{"faithfulness": <1-5>, "helpfulness": <1-5>, "completeness": <1-5>}}
"""

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    try:
        return json.loads(completion.choices[0].message.content)
    except Exception:
        return {"faithfulness": 0, "helpfulness": 0, "completeness": 0}
