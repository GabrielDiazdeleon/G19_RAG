from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from app import get_response

router = APIRouter()

class ChatRequest(BaseModel):
    prompt: str

class ChatResponse(BaseModel):
    answer: str
    context: List[str]

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        result = get_response(request.prompt)
        return ChatResponse(
            answer=result["answer"],
            context=result["context"]
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )
