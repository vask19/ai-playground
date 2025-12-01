from fastapi import FastAPI
from pydantic import BaseModel
from src.llm import ask_llm

app = FastAPI()


class ChatRequest(BaseModel):
    message: str
    system_prompt: str = "You are a helpful assistant."


class ChatResponse(BaseModel):
    response: str


@app.get("/ping")
def ping():
    return {"message": "pong"}


@app.get("/hello/{name}")
def hello(name: str):
    return {"message": f"Hello, {name}!"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    response = ask_llm(request.message, request.system_prompt)
    return ChatResponse(response=response)
