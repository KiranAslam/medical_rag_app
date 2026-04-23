import os
from dotenv import load_dotenv
from openai import OpenAI
from embedder import load_faiss_index
from config import get_api_key


load_dotenv()

client = OpenAI(
    api_key=get_api_key(),
    base_url="https://openrouter.ai/api/v1"
)

FREE_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"
vectorstore = load_faiss_index()


def build_prompt(query: str, context: str) -> str:
    return f"""You are a specialized medical assistant. Your job is to answer ONLY medical-related questions using the context provided.

If the question is NOT related to medicine, diseases, symptoms, treatments, or health, respond with exactly:
"I can only answer medical-related questions. Please ask about a disease, symptom, or medical condition."

If the question IS medical but context is insufficient, say:
"I don't have enough information about this condition in my knowledge base."

Context:
{context}

Question: {query}

Respond in this exact structured format for medical questions:

## Overview
(2-3 sentences about the disease/condition)

## Signs & Symptoms
- symptom 1
- symptom 2
- symptom 3

## Risk Factors
- risk factor 1
- risk factor 2

## Diagnosis
(how it is diagnosed)

## Treatment
- treatment 1
- treatment 2

## Prevention
- prevention tip 1
- prevention tip 2
"""


def get_rag_answer(query: str) -> tuple:
    retrieved_docs = vectorstore.similarity_search(query, k=5)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    prompt = build_prompt(query, context)

    response = client.chat.completions.create(
        model=FREE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    answer = response.choices[0].message.content
    sources = [doc.page_content[:200] for doc in retrieved_docs]
    return answer, sources

if __name__ == "__main__":
    query = "diabetes"
    answer, sources = get_rag_answer(query)
    print(answer)