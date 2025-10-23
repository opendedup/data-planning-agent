"""
Search Fan-out Generator

Generates related search queries to expand search scope when initial queries
return no results.
"""

import json
import logging
from typing import Any, List

logger = logging.getLogger(__name__)


class SearchFanoutGenerator:
    """Generates related search queries to expand search scope."""
    
    def __init__(self, gemini_client: Any):
        """
        Initialize the fan-out generator.
        
        Args:
            gemini_client: GeminiClient instance for generating related queries
        """
        self.gemini_client = gemini_client
    
    def generate_related_queries(self, original_query: str, num_queries: int = 4) -> List[str]:
        """
        Generate related search queries based on different interpretations.
        
        Args:
            original_query: The user's original intent
            num_queries: Number of related queries to generate
            
        Returns:
            List of related search query strings
        """
        # Use Gemini to generate semantically related queries
        prompt = f"""Given this user query: "{original_query}"

Generate {num_queries} related search queries that interpret this question from different angles or related topics.

Examples:
- Original: "should I trade for aaron rodgers"
  Related: ["quarterback performance metrics", "fantasy football player analysis", "NFL player statistics", "trade value analysis"]

- Original: "customer churn prediction"
  Related: ["customer retention data", "subscription cancellation patterns", "user engagement metrics", "customer lifetime value"]

- Original: "trending products"
  Related: ["product sales data", "inventory movement", "customer purchase patterns", "seasonal trends"]

Generate ONLY the queries as a JSON array, nothing else. No markdown, no explanation:"""
        
        try:
            # Import here to avoid circular dependency
            import google.generativeai as genai
            
            # Generate related queries using Gemini
            response = self.gemini_client.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,  # Some creativity for diverse queries
                ),
                safety_settings=self.gemini_client.safety_settings,
            )
            
            # Extract text and parse JSON
            response_text = response.text.strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith("```"):
                lines = response_text.split("\n")
                response_text = "\n".join(lines[1:-1]) if len(lines) > 2 else response_text
            
            # Parse JSON
            related_queries = json.loads(response_text)
            
            if not isinstance(related_queries, list):
                logger.warning(f"Expected list of queries, got {type(related_queries)}")
                return []
            
            # Validate and clean
            valid_queries = [
                str(q).strip() 
                for q in related_queries 
                if q and isinstance(q, str) and len(str(q).strip()) > 0
            ]
            
            logger.info(f"Generated {len(valid_queries)} related queries for: {original_query}")
            logger.debug(f"Related queries: {valid_queries}")
            
            return valid_queries[:num_queries]
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from Gemini response: {e}")
            logger.debug(f"Response text: {response_text}")
            return []
        except Exception as e:
            logger.error(f"Failed to generate related queries: {e}", exc_info=True)
            return []

