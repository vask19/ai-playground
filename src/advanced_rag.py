"""
Advanced RAG techniques:
- Metadata Filtering
- Hybrid Search (semantic + keyword)
- LLM Reranking
"""
import re
from typing import List, Optional
from src.embeddings import get_embedding, get_or_create_collection, add_documents
from src.llm import ask_llm


def keyword_search(query: str, documents: List[str]) -> List[dict]:
    """
    Simple keyword search - finds words in text.
    
    Returns list of dicts with 'document' and 'keyword_score'.
    """
    query_words = query.lower().split()
    results = []
    
    for i, doc in enumerate(documents):
        doc_lower = doc.lower()
        matches = sum(1 for word in query_words if word in doc_lower)
        if matches > 0:
            results.append({
                "index": i,
                "document": doc,
                "keyword_score": matches / len(query_words)
            })
    
    return sorted(results, key=lambda x: x["keyword_score"], reverse=True)


def semantic_search(
    query: str, 
    n_results: int = 5,
    collection_name: str = "documents",
    where_filter: Optional[dict] = None
) -> List[dict]:
    """
    Semantic search with optional metadata filtering.
    
    Args:
        query: Search query
        n_results: Number of results
        collection_name: ChromaDB collection name
        where_filter: Metadata filter, e.g. {"language": "python"}
    """
    collection = get_or_create_collection(collection_name)
    query_embedding = get_embedding(query)
    
    query_params = {
        "query_embeddings": [query_embedding],
        "n_results": n_results
    }
    
    if where_filter:
        query_params["where"] = where_filter
    
    results = collection.query(**query_params)
    
    formatted = []
    for i in range(len(results["documents"][0])):
        formatted.append({
            "document": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
            "semantic_score": 1 - results["distances"][0][i]
        })
    
    return formatted


def hybrid_search(
    query: str,
    n_results: int = 5,
    alpha: float = 0.5,
    collection_name: str = "documents",
    where_filter: Optional[dict] = None
) -> List[dict]:
    """
    Hybrid search - combines semantic and keyword search.
    
    Args:
        query: Search query
        n_results: Number of results to return
        alpha: Weight for semantic (0 = keyword only, 1 = semantic only)
        collection_name: ChromaDB collection name
        where_filter: Optional metadata filter
    
    Returns:
        List of results with hybrid_score, semantic_score, keyword_score
    """
    # Get semantic results
    semantic_results = semantic_search(
        query, 
        n_results=n_results * 2,  # Get more for better fusion
        collection_name=collection_name,
        where_filter=where_filter
    )
    
    # Get all documents for keyword search
    all_docs = [r["document"] for r in semantic_results]
    keyword_results = keyword_search(query, all_docs)
    
    # Create score map
    doc_scores = {}
    
    for r in semantic_results:
        doc_scores[r["document"]] = {
            "document": r["document"],
            "metadata": r["metadata"],
            "semantic_score": r["semantic_score"],
            "keyword_score": 0
        }
    
    for r in keyword_results:
        if r["document"] in doc_scores:
            doc_scores[r["document"]]["keyword_score"] = r["keyword_score"]
    
    # Calculate hybrid score
    for doc in doc_scores:
        s = doc_scores[doc]
        s["hybrid_score"] = alpha * s["semantic_score"] + (1 - alpha) * s["keyword_score"]
    
    # Sort by hybrid score
    results = sorted(doc_scores.values(), key=lambda x: x["hybrid_score"], reverse=True)
    
    return results[:n_results]


def rerank_with_llm(query: str, documents: List[str], top_k: int = 3) -> List[dict]:
    """
    Rerank documents using LLM relevance scoring.
    
    Args:
        query: User's query
        documents: List of documents to rerank
        top_k: Number of top results to return
    
    Returns:
        List of dicts with 'document' and 'relevance_score' (0-10)
    """
    reranked = []
    
    for doc in documents:
        prompt = f"""Rate how relevant this document is to the query.
Query: {query}
Document: {doc}

Reply with ONLY a number from 0 to 10, where 10 is perfectly relevant."""
        
        try:
            score_str = ask_llm(
                prompt, 
                "You are a relevance scoring system. Reply only with a single number 0-10."
            )
            numbers = re.findall(r'\d+', score_str)
            score = int(numbers[0]) if numbers else 5
            score = min(10, max(0, score))
        except Exception:
            score = 5
        
        reranked.append({
            "document": doc,
            "relevance_score": score
        })
    
    sorted_results = sorted(reranked, key=lambda x: x["relevance_score"], reverse=True)
    return sorted_results[:top_k]


def advanced_rag_query(
    query: str,
    n_results: int = 5,
    use_hybrid: bool = True,
    hybrid_alpha: float = 0.5,
    use_reranking: bool = False,
    rerank_top_k: int = 3,
    where_filter: Optional[dict] = None,
    collection_name: str = "documents"
) -> dict:
    """
    Advanced RAG query with configurable pipeline.
    
    Args:
        query: User's question
        n_results: Number of documents to retrieve
        use_hybrid: Use hybrid search instead of pure semantic
        hybrid_alpha: Semantic weight for hybrid search
        use_reranking: Apply LLM reranking
        rerank_top_k: Number of results after reranking
        where_filter: Metadata filter
        collection_name: ChromaDB collection name
    
    Returns:
        Dict with 'answer', 'sources', 'pipeline_info'
    """
    pipeline_info = {
        "search_type": "hybrid" if use_hybrid else "semantic",
        "reranking": use_reranking
    }
    
    # Step 1: Search
    if use_hybrid:
        search_results = hybrid_search(
            query, 
            n_results=n_results,
            alpha=hybrid_alpha,
            collection_name=collection_name,
            where_filter=where_filter
        )
    else:
        search_results = semantic_search(
            query,
            n_results=n_results,
            collection_name=collection_name,
            where_filter=where_filter
        )
    
    if not search_results:
        return {
            "answer": "No documents found.",
            "sources": [],
            "pipeline_info": pipeline_info
        }
    
    # Step 2: Reranking (optional)
    if use_reranking:
        docs_to_rerank = [r["document"] for r in search_results]
        reranked = rerank_with_llm(query, docs_to_rerank, top_k=rerank_top_k)
        final_docs = [r["document"] for r in reranked]
        pipeline_info["rerank_scores"] = [r["relevance_score"] for r in reranked]
    else:
        final_docs = [r["document"] for r in search_results[:rerank_top_k]]
    
    # Step 3: Generate answer
    context = "\n\n".join([f"[{i+1}] {doc}" for i, doc in enumerate(final_docs)])
    
    system_prompt = f"""Answer based ONLY on this context. Cite sources using [1], [2], etc.

Context:
{context}"""
    
    answer = ask_llm(query, system_prompt)
    
    return {
        "answer": answer,
        "sources": final_docs,
        "pipeline_info": pipeline_info
    }
