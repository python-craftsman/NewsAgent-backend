import httpx
import os
from typing import List, Optional
from ..models import NewsArticle, NewsSearchResult


class NewsService:
    def __init__(self):
        self.api_key = os.getenv("EXA_API_KEY")
        self.base_url = "https://api.exa.ai"
        
    async def fetch_news(self, query: str, num_results: int = 5) -> NewsSearchResult:
        """
        Fetch news articles using Exa AI API.
        """
        if not self.api_key:
            raise ValueError("EXA_API_KEY not found in environment variables")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "query": query,
            "numResults": min(num_results, 10),  # Limit to 10 results
            "includeDomains": ["news.yahoo.com", "reuters.com", "bbc.com", "cnn.com", "techcrunch.com"],
            "excludeDomains": [],
            "startCrawlDate": None,
            "endCrawlDate": None,
            "startPublishedDate": None,
            "endPublishedDate": None,
            "useAutoprompt": True,
            "type": "keyword"
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/search",
                    headers=headers,
                    json=payload,
                    timeout=30.0
                )
                response.raise_for_status()
                
                data = response.json()
                articles = []
                
                for result in data.get("results", []):
                    article = NewsArticle(
                        title=result.get("title", ""),
                        url=result.get("url", ""),
                        content=result.get("text", ""),
                        published_date=result.get("publishedDate"),
                        source=result.get("domain")
                    )
                    articles.append(article)
                
                return NewsSearchResult(
                    articles=articles,
                    query=query,
                    total_results=len(articles)
                )
                
        except httpx.HTTPStatusError as e:
            raise Exception(f"HTTP error occurred: {e.response.status_code} - {e.response.text}")
        except httpx.RequestError as e:
            raise Exception(f"Request error occurred: {e}")
        except Exception as e:
            raise Exception(f"Unexpected error: {e}")
    
    async def summarize_articles(self, articles: List[NewsArticle], summary_length: str = "brief") -> str:
        """
        Summarize a list of news articles.
        """
        if not articles:
            return "No articles to summarize."
        
        # Create a summary of the articles
        summary_parts = []
        
        for i, article in enumerate(articles[:5], 1):  # Limit to 5 articles for summary
            if summary_length == "brief":
                # Brief summary: title and key points
                summary_parts.append(f"{i}. {article.title}")
                if article.content:
                    # Take first 100 characters as a brief summary
                    brief_content = article.content[:100] + "..." if len(article.content) > 100 else article.content
                    summary_parts.append(f"   {brief_content}")
            else:
                # Detailed summary: title, source, and more content
                summary_parts.append(f"{i}. {article.title}")
                if article.source:
                    summary_parts.append(f"   Source: {article.source}")
                if article.content:
                    content_length = 200 if summary_length == "detailed" else 300
                    detailed_content = article.content[:content_length] + "..." if len(article.content) > content_length else article.content
                    summary_parts.append(f"   {detailed_content}")
            
            summary_parts.append("")  # Add spacing between articles
        
        return "\n".join(summary_parts) 