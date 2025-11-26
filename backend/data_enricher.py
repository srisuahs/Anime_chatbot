"""
Data Enrichment Module: Gather MAL & Web Data for NLP Context
Enriches prompts with real anime data before passing to DistilGPT-2
"""

import httpx
import logging
from typing import Optional, Dict, List, Any
from datetime import datetime
import re

logger = logging.getLogger(__name__)

class DataEnricher:
    """
    Enriches user queries and prompts with data from:
    1. MyAnimeList API - Anime details, user list info
    2. Google Search - Web search for anime-related info
    """
    
    def __init__(self, mal_access_token: Optional[str] = None):
        """
        Initialize data enricher
        
        Args:
            mal_access_token: User's MAL access token (optional for public queries)
        """
        self.mal_access_token = mal_access_token
        self.mal_base_url = "https://api.myanimelist.net/v2"
        self.google_api_key = None  # Set via environment if needed
        self.google_search_engine_id = None
        
    # ============ ENTITY EXTRACTION ============
    
    def extract_anime_names(self, text: str) -> List[str]:
        """
        Extract potential anime names from text
        Uses heuristics: capitalized words, quoted strings
        
        Args:
            text: User message
            
        Returns:
            List of potential anime names
        """
        # Extract quoted strings
        quoted = re.findall(r'"([^"]+)"', text)
        
        # Extract capitalized phrases (anime names often capitalized)
        capitalized = re.findall(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', text)
        
        return list(set(quoted + capitalized))
    
    def extract_genres(self, text: str) -> List[str]:
        """
        Extract genre keywords from text
        
        Args:
            text: User message
            
        Returns:
            List of genre keywords
        """
        known_genres = [
            "action", "adventure", "comedy", "drama", "ecchi", "fantasy",
            "hentai", "horror", "mahou shoujo", "mecha", "music", "mystery",
            "psychological", "romance", "sci-fi", "shounen", "shoujo", "slice of life",
            "sports", "supernatural", "thriller", "school", "military", "magic"
        ]
        
        text_lower = text.lower()
        found_genres = [g for g in known_genres if g in text_lower]
        
        return found_genres
    
    # ============ MAL DATA RETRIEVAL ============
    
    async def search_anime_on_mal(self, anime_name: str) -> Optional[Dict[str, Any]]:
        """
        Search for anime on MAL API
        
        Args:
            anime_name: Name of anime to search
            
        Returns:
            Anime data or None
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.mal_base_url}/anime",
                    params={
                        "query": anime_name,
                        "fields": "id,title,alternative_titles,synopsis,mean,status,num_episodes,genres,studios"
                    },
                    headers={"X-MAL-CLIENT-ID": "YOUR_CLIENT_ID"}  # Set in env
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("data") and len(data["data"]) > 0:
                        return {
                            "id": data["data"][0]["node"]["id"],
                            "title": data["data"][0]["node"]["title"],
                            "synopsis": data["data"][0]["node"].get("synopsis", ""),
                            "score": data["data"][0]["node"].get("mean", "N/A"),
                            "status": data["data"][0]["node"].get("status", ""),
                            "episodes": data["data"][0]["node"].get("num_episodes", "?"),
                            "genres": [g["name"] for g in data["data"][0]["node"].get("genres", [])],
                        }
        except Exception as e:
            logger.error(f"MAL search error for '{anime_name}': {e}")
        
        return None
    
    async def get_user_anime_list_stats(self, mal_access_token: str) -> Optional[Dict[str, Any]]:
        """
        Get user's anime list statistics from MAL
        
        Args:
            mal_access_token: User's MAL access token
            
        Returns:
            User stats or None
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.mal_base_url}/users/@me/anime_statistics",
                    headers={"Authorization": f"Bearer {mal_access_token}"}
                )
                
                if response.status_code == 200:
                    stats = response.json()
                    return {
                        "watching": stats.get("watching", 0),
                        "completed": stats.get("completed", 0),
                        "on_hold": stats.get("on_hold", 0),
                        "dropped": stats.get("dropped", 0),
                        "plan_to_watch": stats.get("plan_to_watch", 0),
                        "total_entries": stats.get("total_entries", 0),
                        "days_watched": stats.get("days_watched", 0),
                    }
        except Exception as e:
            logger.error(f"Error fetching user stats: {e}")
        
        return None
    
    # ============ WEB SEARCH ============
    
    async def search_anime_web(self, query: str) -> List[str]:
        """
        Search web for anime information (using Google Custom Search)
        
        Args:
            query: Search query
            
        Returns:
            List of relevant snippets
        """
        if not self.google_api_key:
            logger.warning("Google API key not configured, skipping web search")
            return []
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://www.googleapis.com/customsearch/v1",
                    params={
                        "key": self.google_api_key,
                        "cx": self.google_search_engine_id,
                        "q": f"anime {query}",
                        "num": 3  # Top 3 results
                    }
                )
                
                if response.status_code == 200:
                    results = response.json()
                    snippets = [
                        item.get("snippet", "")
                        for item in results.get("items", [])
                    ]
                    return [s for s in snippets if s]  # Filter empty
        except Exception as e:
            logger.error(f"Web search error: {e}")
        
        return []
    
    # ============ CONTEXT ENRICHMENT ============
    
    async def enrich_prompt(
        self,
        user_message: str,
        mal_access_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Enrich user message with contextual data
        
        Args:
            user_message: Original user message
            mal_access_token: Optional user MAL token
            
        Returns:
            {
                "original_message": str,
                "extracted_entities": List[str],
                "genres": List[str],
                "mal_data": Optional[Dict],
                "user_stats": Optional[Dict],
                "web_snippets": List[str],
                "enriched_context": str
            }
        """
        enrichment = {
            "original_message": user_message,
            "extracted_entities": [],
            "genres": [],
            "mal_data": None,
            "user_stats": None,
            "web_snippets": []
        }
        
        # Extract entities
        entities = self.extract_anime_names(user_message)
        genres = self.extract_genres(user_message)
        
        enrichment["extracted_entities"] = entities
        enrichment["genres"] = genres
        
        logger.info(f"Extracted entities: {entities}, genres: {genres}")
        
        # Get MAL data for first entity
        if entities:
            mal_result = await self.search_anime_on_mal(entities[0])
            if mal_result:
                enrichment["mal_data"] = mal_result
                logger.info(f"Found MAL data: {mal_result['title']}")
        
        # Get user stats
        if mal_access_token:
            user_stats = await self.get_user_anime_list_stats(mal_access_token)
            if user_stats:
                enrichment["user_stats"] = user_stats
                logger.info(f"User has {user_stats['total_entries']} anime tracked")
        
        # Web search
        web_results = await self.search_anime_web(user_message)
        if web_results:
            enrichment["web_snippets"] = web_results
            logger.info(f"Found {len(web_results)} web results")
        
        # Build enriched context string
        context_parts = []
        
        if enrichment["mal_data"]:
            mal = enrichment["mal_data"]
            context_parts.append(
                f"[ANIME_CONTEXT] Title: {mal['title']}, "
                f"Score: {mal['score']}, Episodes: {mal['episodes']}, "
                f"Status: {mal['status']}, Genres: {', '.join(mal['genres'][:3])}"
            )
        
        if enrichment["user_stats"]:
            stats = enrichment["user_stats"]
            context_parts.append(
                f"[USER_STATS] Watching: {stats['watching']}, "
                f"Completed: {stats['completed']}, "
                f"Plan to Watch: {stats['plan_to_watch']}"
            )
        
        if enrichment["web_snippets"]:
            context_parts.append(
                f"[WEB_INFO] {enrichment['web_snippets'][0][:100]}..."
            )
        
        enrichment["enriched_context"] = " ".join(context_parts)
        
        return enrichment
    
    async def build_enriched_prompt(
        self,
        user_message: str,
        intent: str,
        chat_history: List[Dict[str, str]],
        mal_access_token: Optional[str] = None
    ) -> str:
        """
        Build complete enriched prompt with MAL + web data
        
        Args:
            user_message: Current user message
            intent: Classified intent
            chat_history: Previous messages
            mal_access_token: Optional user MAL token
            
        Returns:
            Enriched prompt string
        """
        # Enrich message
        enrichment = await self.enrich_prompt(user_message, mal_access_token)
        
        # Build prompt
        prompt_parts = []
        
        # Add enrichment context
        if enrichment["enriched_context"]:
            prompt_parts.append(enrichment["enriched_context"])
        
        # Add entities
        entities_str = ", ".join(enrichment["extracted_entities"])
        if entities_str:
            prompt_parts.append(f"[INTENT={intent}] [ENTITIES={entities_str}]")
        else:
            prompt_parts.append(f"[INTENT={intent}]")
        
        # Add genres
        if enrichment["genres"]:
            prompt_parts.append(f"[GENRES={', '.join(enrichment['genres'])}]")
        
        # Add chat history
        for msg in chat_history[-5:]:  # Last 5 exchanges
            prompt_parts.append(f"<USER>: {msg['user']}\n<BOT>: {msg['bot']}")
        
        # Add current message
        prompt_parts.append(f"<USER>: {user_message}\n<BOT>:")
        
        return "\n".join(prompt_parts)


# Global enricher instance
_enricher = None

def get_data_enricher(mal_token: Optional[str] = None) -> DataEnricher:
    """Get or create data enricher instance"""
    global _enricher
    if _enricher is None:
        _enricher = DataEnricher(mal_token)
    return _enricher
