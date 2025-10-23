"""
Simplified Vertex AI Search Client for Planning Agent

Lightweight client for querying the Vertex AI Search datastore
to get context about available data assets.
"""

import logging
from typing import Any, Dict, List, Optional

from google.cloud import discoveryengine_v1beta as discoveryengine
from google.api_core import retry

logger = logging.getLogger(__name__)


class SearchResult:
    """Simple search result container."""
    
    def __init__(
        self,
        document_id: str,
        snippet: str,
        metadata: Dict[str, Any],
    ):
        """
        Initialize search result.
        
        Args:
            document_id: Document ID
            snippet: Content snippet
            metadata: Document metadata
        """
        self.document_id = document_id
        self.snippet = snippet
        self.metadata = metadata
        
        # Extract common metadata fields
        self.project_id = metadata.get("project_id", "")
        self.dataset_id = metadata.get("dataset_id", "")
        self.table_id = metadata.get("table_id", "")
        self.description = metadata.get("description", "")
        self.row_count = metadata.get("row_count")
        self.column_count = metadata.get("column_count")
        self.has_pii = metadata.get("has_pii", False)
        self.has_phi = metadata.get("has_phi", False)


class VertexSearchClient:
    """
    Simplified Vertex AI Search client for planning agent.
    
    Queries the datastore to understand available data assets
    without exposing technical details.
    """
    
    def __init__(
        self,
        project_id: str,
        location: str = "global",
        datastore_id: str = "data-discovery-metadata",
    ):
        """
        Initialize Vertex AI Search client.
        
        Args:
            project_id: GCP project ID
            location: GCP location for Vertex AI Search (typically 'global')
            datastore_id: ID of the data store
        """
        self.project_id = project_id
        self.location = location
        self.datastore_id = datastore_id
        
        # Initialize Google Cloud client
        self.search_client = discoveryengine.SearchServiceClient()
        
        # Build resource path
        self.serving_config = (
            f"projects/{project_id}/locations/{location}/"
            f"collections/default_collection/dataStores/{datastore_id}/"
            f"servingConfigs/default_config"
        )
        
        logger.info(
            f"Initialized VertexSearchClient for project={project_id}, "
            f"datastore={datastore_id}"
        )
    
    def search(
        self,
        query: str,
        max_results: int = 5,
        timeout: float = 10.0,
    ) -> List[SearchResult]:
        """
        Execute a search query against the datastore.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            timeout: Query timeout in seconds
        
        Returns:
            List of SearchResult objects
        """
        try:
            # Create search request
            search_request = discoveryengine.SearchRequest(
                serving_config=self.serving_config,
                query=query,
                page_size=max_results,
            )
            
            # Execute search with retry
            logger.debug(f"Searching datastore: query='{query}', max_results={max_results}")
            response = self.search_client.search(
                request=search_request,
                timeout=timeout,
                retry=retry.Retry(
                    initial=1.0,
                    maximum=5.0,
                    multiplier=2.0,
                    predicate=retry.if_exception_type(Exception),
                ),
            )
            
            # Parse results
            results = []
            for response_item in response.results:
                document = response_item.document
                
                # Extract structured data
                struct_data = document.struct_data if hasattr(document, 'struct_data') else {}
                if struct_data is None:
                    struct_data = {}
                
                # Extract snippet
                snippet = self._extract_snippet(response_item, document)
                
                # Create result
                result = SearchResult(
                    document_id=document.id,
                    snippet=snippet,
                    metadata=dict(struct_data),
                )
                
                results.append(result)
            
            logger.info(f"Search completed: found {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def _extract_snippet(
        self,
        response_item: Any,
        document: Any,
    ) -> str:
        """
        Extract content snippet from search result.
        
        Args:
            response_item: Response item from search
            document: Document object
        
        Returns:
            Content snippet string
        """
        # Try to get snippet from derived_struct_data
        if hasattr(response_item, 'derived_struct_data'):
            derived_data = response_item.derived_struct_data
            if derived_data:
                snippets = derived_data.get('snippets', [])
                if snippets:
                    return snippets[0].get('snippet', '')
        
        # Fallback to first 200 chars of content
        if hasattr(document, 'content'):
            content = document.content
            # Handle protobuf Content object
            if hasattr(content, 'raw_bytes'):
                try:
                    text = content.raw_bytes.decode('utf-8') if content.raw_bytes else ''
                    return text[:200] + "..." if len(text) > 200 else text
                except (AttributeError, UnicodeDecodeError):
                    pass
            # Handle dict-like content
            elif isinstance(content, dict):
                text = content.get('text', '')
                return text[:200] + "..." if len(text) > 200 else text
        
        return ""
    
    def format_results_for_context(
        self,
        results: List[SearchResult],
        include_technical_details: bool = False,
    ) -> str:
        """
        Format search results as context for prompts.
        
        Args:
            results: List of search results
            include_technical_details: Whether to include row counts, etc.
        
        Returns:
            Formatted context string (without table names)
        """
        if not results:
            return "No directly matching data found in the catalog."
        
        context_parts = ["Based on the data catalog, we have:"]
        
        for i, result in enumerate(results, 1):
            desc_parts = []
            
            # Describe domain/area without table name
            if result.dataset_id:
                domain = result.dataset_id.replace('_', ' ').title()
                desc_parts.append(f"data in the {domain} domain")
            
            # Add scale if available
            if include_technical_details and result.row_count:
                desc_parts.append(f"({result.row_count:,} records)")
            
            # Add field count if available
            if include_technical_details and result.column_count:
                desc_parts.append(f"with {result.column_count} fields")
            
            # Add description or snippet
            content = result.description or result.snippet
            if content:
                # Truncate if too long
                if len(content) > 150:
                    content = content[:150] + "..."
                context_parts.append(f"{i}. {' '.join(desc_parts)}: {content}")
            else:
                context_parts.append(f"{i}. {' '.join(desc_parts)}")
        
        return "\n".join(context_parts)
    
    def search_with_fanout(
        self,
        primary_query: str,
        related_queries: List[str],
        max_results_per_query: int = 3,
    ) -> Dict[str, Any]:
        """
        Execute primary search, and if empty, execute related searches.
        
        Args:
            primary_query: Original user query
            related_queries: List of related queries to try
            max_results_per_query: Max results per query
            
        Returns:
            Dict mapping query to results: {
                "primary": [...],
                "related": {
                    "query1": [...],
                    "query2": [...],
                }
            }
        """
        # Execute primary search
        logger.info(f"Executing primary search: {primary_query}")
        primary_results = self.search(primary_query, max_results_per_query * 2)
        
        if primary_results:
            # Found results with primary query - no need to fan out
            logger.info(f"Primary search found {len(primary_results)} results - skipping fan-out")
            return {
                "primary": primary_results,
                "related": {}
            }
        
        # Primary search returned nothing - fan out to related queries
        logger.info(f"Primary search returned 0 results - executing fan-out with {len(related_queries)} related queries")
        related_results = {}
        for related_query in related_queries:
            logger.debug(f"Searching related query: {related_query}")
            results = self.search(related_query, max_results_per_query)
            if results:
                logger.info(f"Related query '{related_query}' found {len(results)} results")
                related_results[related_query] = results
            else:
                logger.debug(f"Related query '{related_query}' found 0 results")
        
        logger.info(f"Fan-out complete: {len(related_results)} related queries found results")
        return {
            "primary": [],
            "related": related_results
        }
    
    def format_fanout_results(
        self,
        fanout_results: Dict[str, Any],
        original_query: str,
    ) -> tuple[str, str]:
        """
        Format fan-out search results for prompts.
        
        Args:
            fanout_results: Results from search_with_fanout
            original_query: The original user query
            
        Returns:
            Tuple of (context_type, formatted_context) where:
            - context_type: "exact_match", "related_match", or "no_match"
            - formatted_context: Formatted string for prompt
        """
        primary = fanout_results.get("primary", [])
        related = fanout_results.get("related", {})
        
        if primary:
            # Exact match found
            logger.debug("Formatting exact match results")
            context = self.format_results_for_context(primary, include_technical_details=True)
            return ("exact_match", context)
        
        elif related:
            # Related matches found
            logger.debug(f"Formatting related match results ({len(related)} related topics)")
            context_parts = [
                f"I don't have data that exactly matches your query about '{original_query}', "
                f"but I found related data that might be relevant:\n"
            ]
            
            for query, results in related.items():
                context_parts.append(f"\nRelated to: {query}")
                context_parts.append(self.format_results_for_context(results, include_technical_details=True))
            
            return ("related_match", "\n".join(context_parts))
        
        else:
            # No matches at all
            logger.debug("No matches found in primary or related searches")
            searched_topics = list(related.keys()) if related else []
            context = (
                f"I searched the data catalog for:\n"
                f"- {original_query}\n"
            )
            
            if searched_topics:
                context += "\n".join(f"- {q}" for q in searched_topics)
                context += "\n"
            
            context += "\nNo matching data was found. Could you provide more context about what you're trying to accomplish?"
            
            return ("no_match", context)

