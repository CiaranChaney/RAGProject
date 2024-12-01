from openai import OpenAI
import os
from sqlalchemy import create_engine, text
import numpy as np
from ragEmbed import generate_embedding

# Database connection
engine = create_engine("postgresql://raguser:ragpassword@localhost/ragdb")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"), 
)

def retrieve_relevant_chunks(query_embedding, top_k=100):
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
    """Generate completion using OpenAI's API."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
    )
    
    return response.choices[0].message.content

def main():
    query = input("Please enter your query: ")

    query_embedding = generate_embedding(query)

    retrieved_chunks = retrieve_relevant_chunks(query_embedding)
    context = "\n".join(retrieved_chunks)

    prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    response = generate_completion(prompt)
    print("Response:", response)

if __name__ == "__main__":
    main()
