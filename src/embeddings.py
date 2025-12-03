"""
Embeddings and Vector Database module.
Uses Together AI for embeddings and ChromaDB for storage.
"""
import os
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
from typing import List

load_dotenv()

# Together AI client for embeddings
client = OpenAI(
    api_key=os.getenv("TOGETHER_API_KEY"),
    base_url="https://api.together.xyz/v1"
)

# ChromaDB client (local, persistent storage)
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Embedding model
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"


def get_embedding(text: str) -> List[float]:
    """Get embedding vector for text."""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding


def get_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """Get embeddings for multiple texts at once"""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )
    return [item.embedding for item in response.data]


def get_or_create_collection(name: str = "documents"):
    """Get or create a ChromaDB collection."""
    return chroma_client.get_or_create_collection(
        name=name,
        metadata={"description": "Document embeddings for RAG"}
    )


def add_documents(
    documents: List[str],
    metadatas: List[dict] = None,
    ids: List[str] = None,
    collection_name: str = "documents"
):
    """Add documents to vector database."""
    collection = get_or_create_collection(collection_name)
    
    # Generate IDs if not provided
    if ids is None:
        existing_count = collection.count()
        ids = [f"doc_{existing_count + i}" for i in range(len(documents))]
    
    # Generate default metadata if not provided (ChromaDB requires non-empty)
    if metadatas is None:
        metadatas = [{"source": "manual", "index": i} for i in range(len(documents))]
    
    # Generate embeddings
    embeddings = get_embeddings_batch(documents)
    
    # Add to collection
    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )
    
    return ids


def search_documents(
    query: str,
    n_results: int = 3,
    collection_name: str = "documents"
) -> List[dict]:
    """Search for similar documents."""
    collection = get_or_create_collection(collection_name)
    
    # Get query embedding
    query_embedding = get_embedding(query)
    
    # Search
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    
    # Format results
    formatted = []
    for i in range(len(results["documents"][0])):
        formatted.append({
            "document": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i]
        })
    
    return formatted


def delete_collection(collection_name: str = "documents"):
    """Delete a collection"""
    try:
        chroma_client.delete_collection(collection_name)
        return True
    except Exception:
        return False


def get_collection_stats(collection_name: str = "documents") -> dict:
    """Get collection statistics"""
    collection = get_or_create_collection(collection_name)
    return {
        "name": collection_name,
        "count": collection.count()
    }
