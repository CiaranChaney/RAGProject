import requests
from sqlalchemy import create_engine, text
import numpy as np
from rag_embed import generate_embedding

# Database connection
engine = create_engine("postgresql://raguser:ragpassword@localhost/ragdb")

# LM Studio server URL
LM_STUDIO_URL = "http://localhost:1234/v1/completions"

def retrieve_relevant_chunks(query_embedding, top_k=3):
    query_embedding_str = ', '.join(map(str, query_embedding.tolist())) 
    formatted_query_embedding = f'[{query_embedding_str}]' 
    sql_query = text("""
        SELECT content
        FROM rag_items
        ORDER BY embedding <-> :embedding
        LIMIT :top_k;
    """)
    with engine.connect() as conn:
        result = conn.execute(sql_query, {"embedding": formatted_query_embedding, "top_k": top_k})
        return [row["content"] for row in result.mappings()]


def generate_completion(prompt):
    payload = {
        "model": "lmstudio-community/gemma-2-2b-it-GGUF/gemma-2-2b-it-Q8_0.gguf",  # Replace with the model name running on LM Studio
        "prompt": prompt,
        "max_tokens": 750,
        "temperature": 0.7
    }
    response = requests.post(f"{LM_STUDIO_URL}", json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["text"]

def main():
    query = "Where do I send complaints to UUSU?"
    
    query_embedding = generate_embedding(query)
    
    # Retrieve top-k relevant chunks
    retrieved_chunks = retrieve_relevant_chunks(query_embedding)
    context = "\n".join(retrieved_chunks)
    
    prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    response = generate_completion(prompt)
    print("Response:", response)

if __name__ == "__main__":
    main()
