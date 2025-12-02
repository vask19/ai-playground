from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from enum import Enum
from src.llm import ask_llm, ask_llm_stream, LLMError, LLMConnectionError, LLMRateLimitError
from src import prompts

app = FastAPI(title="AI Playground API")


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


class ErrorResponse(BaseModel):
    error: str
    error_type: str


@app.get("/ping")
def ping():
    return {"message": "pong"}


@app.get("/hello/{name}")
def hello(name: str):
    return {"message": f"Hello, {name}!"}


@app.get("/modes")
def list_modes():
    return {"modes": [mode.value for mode in AssistantMode]}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
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
    system_prompt = MODE_TO_PROMPT[request.mode]
    
    def generate():
        for chunk in ask_llm_stream(request.message, system_prompt):
            yield chunk
    
    return StreamingResponse(generate(), media_type="text/plain")


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "ai-playground",
        "llm_provider": "together.ai"
    }
