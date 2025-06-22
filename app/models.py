from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from enum import Enum


class ToneOfVoice(str, Enum):
    FORMAL = "formal"
    CASUAL = "casual"
    ENTHUSIASTIC = "enthusiastic"


class ResponseFormat(str, Enum):
    BULLET_POINTS = "bullet points"
    PARAGRAPHS = "paragraphs"


class Language(str, Enum):
    ENGLISH = "English"
    SPANISH = "Spanish"


class InteractionStyle(str, Enum):
    CONCISE = "concise"
    DETAILED = "detailed"


class UserPreferences(BaseModel):
    tone_of_voice: Optional[ToneOfVoice] = None
    response_format: Optional[ResponseFormat] = None
    language: Optional[Language] = None
    interaction_style: Optional[InteractionStyle] = None
    preferred_topics: Optional[List[str]] = None


class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: Optional[str] = None


class ChatRequest(BaseModel):
    message: str
    conversation_history: List[ChatMessage]
    user_preferences: UserPreferences


class ChatResponse(BaseModel):
    message: str
    conversation_history: List[ChatMessage]
    user_preferences: UserPreferences
    tools_used: Optional[List[str]] = None


class ToolCall(BaseModel):
    name: str
    arguments: Dict[str, Any]


class ToolResult(BaseModel):
    tool_name: str
    result: str
    success: bool
    error: Optional[str] = None


class NewsArticle(BaseModel):
    title: str
    url: str
    content: str
    published_date: Optional[str] = None
    source: Optional[str] = None


class NewsSearchResult(BaseModel):
    articles: List[NewsArticle]
    query: str
    total_results: int 