from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from enum import Enum
from typing import List, Optional
from src.llm import ask_llm, ask_llm_stream, LLMError, LLMConnectionError, LLMRateLimitError
from src.rag import rag_query, rag_query_stream, ingest_documents
from src.advanced_rag import advanced_rag_query, hybrid_search, semantic_search
from src.embeddings import get_collection_stats, delete_collection
from src import prompts

app = FastAPI(title="AI Playground API", version="2.0")


# === Models ===

class AssistantMode(str, Enum):
    basic = "basic"
    python = "python"
    translator = "translator"
    reviewer = "reviewer"
    json_extractor = "json_extractor"


MODE_TO_PROMPT = {
    AssistantMode.basic: prompts.ASSISTANT_BASIC,
    AssistantMode.python: prompts.PYTHON_EXPERT,
    AssistantMode.translator: prompts.TRANSLATOR_PL,
    AssistantMode.reviewer: prompts.CODE_REVIEWER,
    AssistantMode.json_extractor: prompts.JSON_EXTRACTOR,
}


class ChatRequest(BaseModel):
    message: str
    mode: AssistantMode = AssistantMode.basic


class ChatResponse(BaseModel):
    response: str
    mode: str


class RAGRequest(BaseModel):
    question: str
    n_results: int = 3


class RAGResponse(BaseModel):
    answer: str
    sources: List[str]


class AdvancedRAGRequest(BaseModel):
    question: str
    n_results: int = 5
    use_hybrid: bool = True
    hybrid_alpha: float = 0.5
    use_reranking: bool = False
    rerank_top_k: int = 3
    metadata_filter: Optional[dict] = None


class AdvancedRAGResponse(BaseModel):
    answer: str
    sources: List[str]
    pipeline_info: dict


class IngestRequest(BaseModel):
    documents: List[str]
    metadatas: Optional[List[dict]] = None


class SearchRequest(BaseModel):
    query: str
    n_results: int = 5
    search_type: str = "semantic"  # "semantic" or "hybrid"
    hybrid_alpha: float = 0.5
    metadata_filter: Optional[dict] = None


# === Basic Endpoints ===

@app.get("/ping")
def ping():
    return {"message": "pong"}


@app.get("/hello/{name}")
def hello(name: str):
    return {"message": f"Hello, {name}!"}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "ai-playground",
        "version": "2.0",
        "features": ["chat", "rag", "advanced-rag"]
    }


# === Chat Endpoints ===

@app.get("/modes")
def list_modes():
    return {"modes": [mode.value for mode in AssistantMode]}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Chat with LLM"""
    system_prompt = MODE_TO_PROMPT[request.mode]
    
    try:
        response = ask_llm(request.message, system_prompt)
        return ChatResponse(response=response, mode=request.mode)
    
    except LLMRateLimitError as e:
        raise HTTPException(status_code=429, detail=str(e))
    except LLMConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except LLMError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
def chat_stream(request: ChatRequest):
    """Streaming chat"""
    system_prompt = MODE_TO_PROMPT[request.mode]
    
    def generate():
        for chunk in ask_llm_stream(request.message, system_prompt):
            yield chunk
    
    return StreamingResponse(generate(), media_type="text/plain")


# === Basic RAG Endpoints ===

@app.post("/rag/ingest")
def rag_ingest(request: IngestRequest):
    """Ingest documents into RAG knowledge base"""
    try:
        result = ingest_documents(request.documents, request.metadatas)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag/query", response_model=RAGResponse)
def rag_query_endpoint(request: RAGRequest):
    """Basic RAG query"""
    try:
        result = rag_query(request.question, request.n_results)
        return RAGResponse(answer=result["answer"], sources=[s["document"] for s in result["sources"]])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag/query/stream")
def rag_query_stream_endpoint(request: RAGRequest):
    """Streaming RAG query"""
    def generate():
        for chunk in rag_query_stream(request.question, request.n_results):
            yield chunk
    
    return StreamingResponse(generate(), media_type="text/plain")


@app.get("/rag/stats")
def rag_stats():
    """Get RAG knowledge base statistics"""
    return get_collection_stats()


@app.delete("/rag/clear")
def rag_clear():
    """Clear RAG knowledge base"""
    success = delete_collection()
    return {"cleared": success}


# === Advanced RAG Endpoints ===

@app.post("/rag/search")
def rag_search(request: SearchRequest):
    """
    Search documents with semantic or hybrid search.
    
    - semantic: Pure embedding-based search
    - hybrid: Combines semantic + keyword search
    """
    try:
        if request.search_type == "hybrid":
            results = hybrid_search(
                request.query,
                n_results=request.n_results,
                alpha=request.hybrid_alpha,
                where_filter=request.metadata_filter
            )
        else:
            results = semantic_search(
                request.query,
                n_results=request.n_results,
                where_filter=request.metadata_filter
            )
        
        return {
            "query": request.query,
            "search_type": request.search_type,
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag/advanced", response_model=AdvancedRAGResponse)
def rag_advanced_query(request: AdvancedRAGRequest):
    """
    Advanced RAG with configurable pipeline:
    
    - Hybrid search (semantic + keyword)
    - Metadata filtering
    - LLM reranking
    
    Example request:
    {
        "question": "What is FastAPI?",
        "use_hybrid": true,
        "hybrid_alpha": 0.6,
        "use_reranking": true,
        "rerank_top_k": 3,
        "metadata_filter": {"language": "python"}
    }
    """
    try:
        result = advanced_rag_query(
            query=request.question,
            n_results=request.n_results,
            use_hybrid=request.use_hybrid,
            hybrid_alpha=request.hybrid_alpha,
            use_reranking=request.use_reranking,
            rerank_top_k=request.rerank_top_k,
            where_filter=request.metadata_filter
        )
        
        return AdvancedRAGResponse(
            answer=result["answer"],
            sources=result["sources"],
            pipeline_info=result["pipeline_info"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
