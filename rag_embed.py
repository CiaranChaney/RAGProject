from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String
from pgvector.sqlalchemy import Vector
from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch

# Database connection
engine = create_engine("postgresql://raguser:ragpassword@localhost/ragdb")

# Define the table schema
metadata = MetaData()
rag_items = Table(
    "rag_items",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("content", String(1024)),
    Column("embedding", Vector(384))
)

# Load the embedding model
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def generate_embedding(text):
    """
    Generate embeddings for the given text using the pretrained model.
    """
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        output = model(**tokens)
    return np.mean(output.last_hidden_state.numpy(), axis=1)[0]

def process_file(file_path, chunk_size=500, overlap=100):
    """
    Process the file, split it into chunks, and return the list of chunks.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        document = file.read()

    chunks = [document[i:i+chunk_size] for i in range(0, len(document), chunk_size - overlap)]
    return chunks

def insert_into_db(chunks):
    """
    Insert the chunks and their embeddings into the database.
    """
    with engine.begin() as conn:  
        for chunk in chunks:
            embedding = generate_embedding(chunk)
            print(f"Inserting chunk: {chunk[:50]}...")
            conn.execute(rag_items.insert().values(content=chunk, embedding=embedding))
        print("Insertion complete.")


if __name__ == "__main__":
    file_path = "./Policies/WayFindingStrategySignageProtocol.txt"

    chunks = process_file(file_path)

    insert_into_db(chunks)
