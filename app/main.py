from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam
from typing import List, Dict, Any, cast
import json
import asyncio
from contextlib import asynccontextmanager

# Load environment variables from .env file
load_dotenv()

from .models import ChatRequest, ChatResponse, ChatMessage
from .services.news_service import NewsService
from .utils.tools import get_tool_definitions

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.getenv("OPENAI_API_KEY"):
        raise Exception("OPENAI_API_KEY is not set in environment variables")
    if not os.getenv("EXA_API_KEY"):
        raise Exception("EXA_API_KEY is not set in environment variables")
    yield  # App runs here

# Initialize FastAPI app
app = FastAPI(
    title="Latest News Agent API",
    description="API for interacting with the AI-powered news agent",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    # allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services and clients
news_service = NewsService()
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# In-memory storage for conversation history (replace with a database in production)
conversation_history: List[Dict[str, Any]] = []

# Helper to convert our message format to OpenAI's ChatCompletionMessageParam

def to_openai_message(msg: Dict[str, Any]) -> ChatCompletionMessageParam:
    if msg["role"] == "system":
        return cast(ChatCompletionMessageParam, {"role": "system", "content": msg["content"]})
    elif msg["role"] == "user":
        return cast(ChatCompletionMessageParam, {"role": "user", "content": msg["content"]})
    elif msg["role"] == "assistant":
        m: Dict[str, Any] = {"role": "assistant", "content": msg.get("content", None)}
        if "tool_calls" in msg:
            m["tool_calls"] = msg["tool_calls"]
        return cast(ChatCompletionMessageParam, m)
    elif msg["role"] == "tool":
        return cast(ChatCompletionMessageParam, {
            "role": "tool",
            "content": msg["content"],
            "tool_call_id": msg["tool_call_id"]
        })
    else:
        raise ValueError(f"Unknown role: {msg['role']}")

@app.get("/", summary="Root Endpoint", tags=["Health"])
async def root():
    return {"message": "Welcome to the Latest News Agent API"}

@app.get("/api/health", summary="Health Check", tags=["Health"])
async def health_check():
    return {"status": "ok"}

@app.post("/api/chat", response_model=ChatResponse, summary="Handle Chat Messages", tags=["Chat"])
async def chat_handler(request: ChatRequest):
    global conversation_history
    
    # Add user message to conversation history
    user_message = {"role": "user", "content": request.message}
    conversation_history.append(user_message)
    
    # System prompt to guide the agent
    system_prompt = """
    You are an AI-powered news agent. Your primary role is to provide users with the latest news on their preferred topics.
    
    - First, you must collect the user's preferences: tone, format, language, style, and topics.
    - If preferences are not complete, ask the user for the missing information.
    - Once preferences are collected, you can fetch and summarize news using the available tools.
    - Use the user's preferences to tailor your responses.
    """
    
    # Build OpenAI message list
    openai_messages: List[ChatCompletionMessageParam] = [
        cast(ChatCompletionMessageParam, {"role": "system", "content": system_prompt})
    ] + [to_openai_message(msg) for msg in conversation_history]
    
    # Format tools as ChatCompletionToolParam
    openai_tools: List[ChatCompletionToolParam] = get_tool_definitions()  # already matches the required format
    
    try:
        # First API call to get the agent's response or tool calls
        response = await openai_client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=openai_messages,
            tools=openai_tools,
            tool_choice="auto"
        )
        
        response_message = response.choices[0].message
        tool_calls = getattr(response_message, "tool_calls", None)
        
        # If there are tool calls, execute them
        if tool_calls:
            available_tools = {
                "fetch_news": news_service.fetch_news,
                "summarize_news": news_service.summarize_articles
            }
            
            # Append the assistant's response to history (with tool_calls for traceability)
            conversation_history.append({
                "role": "assistant",
                "content": response_message.content,
                "tool_calls": [tc.model_dump() for tc in tool_calls]  # for traceability
            })
            
            # Execute all tool calls and add tool messages
            tool_results = []
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_to_call = available_tools.get(function_name)
                
                if function_to_call:
                    function_args = json.loads(tool_call.function.arguments)
                    if asyncio.iscoroutinefunction(function_to_call):
                        function_response = await function_to_call(**function_args)
                    else:
                        function_response = function_to_call(**function_args)
                    tool_message = {
                        "role": "tool",
                        "content": str(function_response),
                        "tool_call_id": tool_call.id
                    }
                    tool_results.append(tool_message)
            
            # Append tool results to the conversation
            conversation_history.extend(tool_results)
            
            # Build new OpenAI message list for the follow-up call
            openai_messages = [
                cast(ChatCompletionMessageParam, {"role": "system", "content": system_prompt})
            ] + [to_openai_message(msg) for msg in conversation_history]
            
            # Second API call to get the final response from the agent
            final_response = await openai_client.chat.completions.create(
                model="gpt-4-1106-preview",
                messages=openai_messages,
                tools=openai_tools
            )
            
            final_message = final_response.choices[0].message.content or ""
            conversation_history.append({"role": "assistant", "content": final_message})
            
            return ChatResponse(
                message=final_message,
                conversation_history=[ChatMessage(**msg) for msg in conversation_history],
                user_preferences=request.user_preferences,
                tools_used=[tc.function.name for tc in tool_calls]
            )
        else:
            # If no tool calls, just return the agent's message
            final_message = response_message.content or ""
            conversation_history.append({"role": "assistant", "content": final_message})
            
            return ChatResponse(
                message=final_message,
                conversation_history=[ChatMessage(**msg) for msg in conversation_history],
                user_preferences=request.user_preferences,
                tools_used=[]
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/clear", summary="Clear Conversation History", tags=["Chat"])
async def clear_history():
    global conversation_history
    conversation_history = []
    return {"message": "Conversation history cleared"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 