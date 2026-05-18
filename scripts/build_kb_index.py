import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def build_offline_index():
    print("Building offline FAISS index...")
    
    # 1. Load the embedding model
    model_name = "BAAI/bge-base-en-v1.5"
    print(f"Loading SentenceTransformer: {model_name}")
    encoder = SentenceTransformer(model_name)
    embedding_dim = 768
    
    # 2. Define the baseline Knowledge Base
    initial_facts = [
        "Political framing often uses loaded words to elicit emotional responses and categorize groups into 'us vs them'.",
        "Gender stereotyping frequently assumes traditional roles, such as associating women with caregiving and men with leadership.",
        "Racial bias in media can manifest as over-representing certain groups in negative contexts (e.g., crime) and under-representing them in positive ones.",
        "Exclusionary framing ignores the perspectives of marginalized groups, presenting a single dominant viewpoint as universal.",
        "Emotional manipulation uses fear, outrage, or extreme scenarios to bypass logical reasoning and instill bias.",
        "Nationality bias can involve generalizing the behavior of individuals to an entire country, often relying on outdated tropes."
    ]
    
    # 3. Generate embeddings
    print("Generating embeddings for KB facts...")
    embeddings = encoder.encode(initial_facts, convert_to_numpy=True).astype('float32')
    
    # 4. Build FAISS index
    print("Building FAISS FlatL2 index...")
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings)
    
    # 5. Save to disk
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(data_dir, exist_ok=True)
    
    index_path = os.path.join(data_dir, "faiss.index")
    metadata_path = os.path.join(data_dir, "metadata.json")
    
    faiss.write_index(index, index_path)
    print(f"Saved FAISS index to {index_path}")
    
    with open(metadata_path, "w") as f:
        json.dump({"documents": initial_facts}, f, indent=4)
    print(f"Saved metadata to {metadata_path}")
    
    print("Offline build complete!")

if __name__ == "__main__":
    build_offline_index()
