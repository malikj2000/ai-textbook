from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import torch

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentenceTransformer('nvidia/NV-Embed-v2', trust_remote_code=True).to(device)
model.max_seq_length = 32768
model.tokenizer.padding_side = "right"

# store this in pinecone database
chunk_embeddings = []
chunks = []

class QueryRequest(BaseModel):
    query: str
    
def add_eos(input_examples):
    return [input_example + model.tokenizer.eos_token for input_example in input_examples]

def embed_textbook_chunks(textbook_chunks):
    global chunks, chunk_embeddings
    chunks = textbook_chunks
    chunk_embeddings = model.encode(add_eos(textbook_chunks), normalize_embeddings=True)

textbook_chunks = [
    "Judo throws are allowed in freestyle and folkstyle wrestling. However, one must follow the rules for slams.",
    "To become a radiology technician in Michigan, one must earn a high school diploma, an associate degree, and get licensed."
]
embed_textbook_chunks(textbook_chunks)

@app.post("/retrieve/")
async def retrieve_passages(query_request: QueryRequest):
    query = query_request.query
    
    query_embedding = model.encode(add_eos([query]), normalize_embeddings=True)
    query_tensor = torch.from_numpy(query_embedding)
    
    similarity_scores = (query_tensor @ torch.tensor(chunk_embeddings).T).tolist()[0]
    
    top_n_indices = sorted(range(len(similarity_scores)), key=lambda i: similarity_scores[i], reverse=True)[:2]
    top_chunks = [chunks[i] for i in top_n_indices]
    top_scores = [similarity_scores[i] for i in top_n_indices]
    
    return {"top_chunks": top_chunks, "scores": top_scores}