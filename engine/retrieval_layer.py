import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class EvidenceRetriever:
    """
    Offline-first FAISS retrieval layer.
    Stores and retrieves contextual evidence and fairness examples.
    """
    
    def __init__(self, model_name="BAAI/bge-base-en-v1.5", embedding_dim=768):
        # Initialize Embedding Model
        self.encoder = SentenceTransformer(model_name, local_files_only=True)
        self.embedding_dim = embedding_dim
        
        # Load Precomputed FAISS Index and Metadata
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        index_path = os.path.join(data_dir, "faiss.index")
        metadata_path = os.path.join(data_dir, "metadata.json")
        
        try:
            print("Loading precomputed FAISS index...")
            self.index = faiss.read_index(index_path)
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            self.documents = metadata.get("documents", [])
            print(f"Successfully loaded FAISS index with {self.index.ntotal} records.")
        except Exception as e:
            print(f"Warning: Failed to load precomputed FAISS index: {e}")
            print("Creating an empty fallback index.")
            self.index = faiss.IndexFlatL2(embedding_dim)
            self.documents = []

    def get_embedding(self, text: str) -> np.ndarray:
        """Generates an embedding for the given text."""
        if not text:
            return np.zeros((1, self.embedding_dim), dtype=np.float32)
        
        # Returns shape (embedding_dim,) -> reshape to (1, embedding_dim)
        embedding = self.encoder.encode(text, convert_to_numpy=True)
        return embedding.reshape(1, -1).astype('float32')

    def add_documents(self, docs: list[str]):
        """Adds text documents to the FAISS index."""
        if not docs:
            return
            
        embeddings = self.encoder.encode(docs, convert_to_numpy=True).astype('float32')
        self.index.add(embeddings)
        self.documents.extend(docs)

    def retrieve(self, query: str, top_k: int = 2) -> list[str]:
        """
        Retrieves top_k relevant contextual facts or evidence for the given query.
        """
        if not query or self.index.ntotal == 0:
            return []
            
        query_embedding = self.get_embedding(query)
        distances, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for idx in indices[0]:
            if idx != -1 and idx < len(self.documents):
                results.append(self.documents[idx])
                
        return results

# Singleton-like instance for the app
retriever_instance = None

def get_retriever():
    global retriever_instance
    if retriever_instance is None:
        retriever_instance = EvidenceRetriever()
    return retriever_instance
