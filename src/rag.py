"""
RAG (Retrieval-Augmented Generation) Pipeline.
Combines document search with LLM to answer questions based on your data.
"""
from src.embeddings import search_documents, add_documents, get_collection_stats
from src.llm import ask_llm, ask_llm_stream
from typing import Generator, List


RAG_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.

Rules:
- Only use information from the provided context
- If the context doesn't contain the answer, say "I don't have enough information to answer this question"
- Be concise and accurate
- Cite which part of the context you used

Context:
{context}
"""


def format_context(search_results: List[dict]) -> str:
    """Format search results into context string"""
    context_parts = []
    for i, result in enumerate(search_results, 1):
        context_parts.append(f"[{i}] {result['document']}")
    return "\n\n".join(context_parts)


def rag_query(
    question: str,
    n_results: int = 3,
    collection_name: str = "documents"
) -> dict:
    """
    RAG query - search for relevant documents and generate answer.
    
    Args:
        question: User's question
        n_results: Number of documents to retrieve
        collection_name: Name of the collection to search
    
    Returns:
        Dict with 'answer', 'sources', and 'context'
    """
    # Step 1: Search for relevant documents
    search_results = search_documents(question, n_results, collection_name)
    
    if not search_results:
        return {
            "answer": "No documents found in the knowledge base.",
            "sources": [],
            "context": ""
        }
    
    # Step 2: Format context
    context = format_context(search_results)
    
    # Step 3: Create prompt with context
    system_prompt = RAG_SYSTEM_PROMPT.format(context=context)
    
    # Step 4: Get LLM response
    answer = ask_llm(question, system_prompt)
    
    return {
        "answer": answer,
        "sources": search_results,
        "context": context
    }


def rag_query_stream(
    question: str,
    n_results: int = 3,
    collection_name: str = "documents"
) -> Generator[str, None, None]:
    """
    RAG query with streaming response.
    First yields sources info, then streams the answer.
    """
    # Step 1: Search for relevant documents
    search_results = search_documents(question, n_results, collection_name)
    
    if not search_results:
        yield "No documents found in the knowledge base."
        return
    
    # Step 2: Format context
    context = format_context(search_results)
    
    # Step 3: Create prompt with context
    system_prompt = RAG_SYSTEM_PROMPT.format(context=context)
    
    # Step 4: Stream LLM response
    for chunk in ask_llm_stream(question, system_prompt):
        yield chunk


def ingest_documents(
    documents: List[str],
    metadatas: List[dict] = None,
    collection_name: str = "documents"
) -> dict:
    """
    Ingest documents into the RAG system.
    
    Args:
        documents: List of text documents
        metadatas: Optional metadata for each document
        collection_name: Name of the collection
    
    Returns:
        Dict with ingestion stats
    """
    ids = add_documents(documents, metadatas, collection_name=collection_name)
    stats = get_collection_stats(collection_name)
    
    return {
        "added": len(ids),
        "total": stats["count"],
        "collection": collection_name
    }
