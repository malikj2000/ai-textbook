from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import torch
from pinecone import Pinecone

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', trust_remote_code=True).to(device)

if model.tokenizer.eos_token is None:
    model.tokenizer.eos_token = "[EOS]"

model.max_seq_length = 384
model.tokenizer.padding_side = "right"

pc = Pinecone(api_key="c88b834d-ad5d-4a77-b08c-25fcc6843939")
index = pc.Index("textbook-index")

chunks = []

class QueryRequest(BaseModel):
    query: str
    
def add_eos(input_examples):
    return [input_example + model.tokenizer.eos_token for input_example in input_examples]

def embed_and_store_in_pinecone(textbook_chunks):
    global chunks
    chunks = textbook_chunks
    chunk_embeddings = model.encode(add_eos(textbook_chunks), normalize_embeddings=True).astype("float32")

    vectors = [(f"chunk-{i}", chunk_embeddings[i].tolist()) for i in range(len(chunk_embeddings))]

    index.upsert(vectors)

textbook_chunks = [
    "Judo throws are allowed in freestyle and folkstyle wrestling. However, one must follow the rules for slams.",
    "To become a radiology technician in Michigan, one must earn a high school diploma, an associate degree, and get licensed."
]
embed_and_store_in_pinecone(textbook_chunks)

@app.post("/retrieve/")
async def retrieve_passages(query_request: QueryRequest):
    query = query_request.query
    
    query_embedding = model.encode(add_eos([query]), normalize_embeddings=True).astype("float32")
    
    results = index.query(vector=[query_embedding.tolist()], top_k=2)
    
    top_chunks = [chunks[int(res['id'].split('-')[1])] for res in results['matches']]
    top_scores = [res['score'] for res in results['matches']]
    
    return {"top_chunks": top_chunks, "scores": top_scores}