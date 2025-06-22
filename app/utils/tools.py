from typing import List, cast, Optional
from openai.types.chat import ChatCompletionToolParam
import json


def get_tool_definitions() -> List[ChatCompletionToolParam]:
    """
    Define the tools available for OpenAI tool calling.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "fetch_news",
                "description": "Fetch the latest news articles on a given topic using Exa AI",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query for news articles (e.g., 'artificial intelligence', 'climate change')"
                        },
                        "num_results": {
                            "type": "integer",
                            "description": "Number of articles to fetch (default: 5, max: 10)",
                            "default": 5
                        }
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "summarize_news",
                "description": "Summarize a news article or multiple articles to provide concise information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "articles": {
                            "type": "array",
                            "description": "Array of news articles to summarize",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "title": {"type": "string"},
                                    "url": {"type": "string"},
                                    "content": {"type": "string"}
                                }
                            }
                        },
                        "summary_length": {
                            "type": "string",
                            "description": "Length of summary: 'brief', 'detailed', or 'comprehensive'",
                            "default": "brief"
                        }
                    },
                    "required": ["articles"]
                }
            }
        }
    ]


def extract_tool_calls(response_content: str) -> list:
    """
    Extract tool calls from OpenAI response content.
    This handles the case where tool calls are embedded in the response.
    """
    tool_calls = []
    
    # Look for tool call patterns in the response
    if "tool_calls" in response_content:
        try:
            # Try to parse as JSON
            parsed = json.loads(response_content)
            if "tool_calls" in parsed:
                tool_calls = parsed["tool_calls"]
        except json.JSONDecodeError:
            pass
    
    return tool_calls


def format_tool_response(tool_name: str, result: str, success: bool = True, error: Optional[str] = None) -> str:
    """
    Format tool response for the AI model.
    """
    if success:
        return f"Tool '{tool_name}' executed successfully. Result: {result}"
    else:
        return f"Tool '{tool_name}' failed. Error: {error}"


openai_tools = cast(List[ChatCompletionToolParam], get_tool_definitions()) 