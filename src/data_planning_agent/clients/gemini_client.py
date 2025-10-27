"""
Gemini API Client

Client for interacting with Google's Gemini API for conversational
requirement gathering and Data PRP generation.
"""

import asyncio
import logging
from typing import Any, Optional
import json

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from ..models.prp_schema import DataProductRequirementPrompt
from ..models.session import PlanningSession
from ..models.prp_schema import PRPAnalysisResult

logger = logging.getLogger(__name__)


class GeminiClient:
    """
    Client for Gemini API interactions.

    Handles conversation management and PRP generation using Gemini models.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.5-pro",
        temperature: float = 1.0,
        max_retries: int = 3,
        context: Optional[str] = None,
        vertex_search_client: Optional[Any] = None,
        enable_reflection: bool = True,
    ):
        """
        Initialize Gemini client.

        Args:
            api_key: Gemini API key
            model_name: Model name to use
            temperature: Sampling temperature (0.0-1.0)
            max_retries: Maximum retry attempts
            context: Optional organizational context to prepend to all prompts
            vertex_search_client: Optional VertexSearchClient for querying data catalog
            enable_reflection: Whether to enable question reflection/self-correction
        """
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries
        self.context = context
        self.enable_reflection = enable_reflection
        
        # Store search client for pre-querying datastore
        self.vertex_search_client = vertex_search_client

        # Configure API
        genai.configure(api_key=api_key)
        
        # Initialize model WITHOUT grounding tools (using pre-query approach instead)
        self.model = genai.GenerativeModel(model_name)

        # Disable safety filters for all categories
        # Must be a list of dicts, not a dict
        self.safety_settings = [
            {"category": HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": HarmBlockThreshold.BLOCK_NONE},
            {"category": HarmCategory.HARM_CATEGORY_HATE_SPEECH, "threshold": HarmBlockThreshold.BLOCK_NONE},
            {"category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, "threshold": HarmBlockThreshold.BLOCK_NONE},
            {"category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, "threshold": HarmBlockThreshold.BLOCK_NONE},
        ]

        search_status = "with data catalog search" if vertex_search_client else "without data catalog"
        if context:
            logger.info(
                f"Initialized Gemini client with model: {model_name} "
                f"({search_status}, {len(context)} chars of context, safety filters disabled)"
            )
        else:
            logger.info(f"Initialized Gemini client with model: {model_name} ({search_status}, safety filters disabled)")

    def _build_prompt_with_context(self, base_prompt: str) -> str:
        """
        Prepend organizational context to a prompt if available.

        Args:
            base_prompt: The base prompt text

        Returns:
            Prompt with context prepended (if context exists)
        """
        if self.context:
            return f"{self.context}\n\n---\n\n{base_prompt}"
        return base_prompt

    def _preprocess_intent(self, intent: str) -> str:
        """
        Preprocess user intent to avoid safety filter triggers.
        
        This helps avoid Gemini's grounding attack filter by framing
        queries as hypothetical data analysis requirements rather than
        requests for real-time information.
        
        Args:
            intent: Raw user intent
            
        Returns:
            Preprocessed intent that's less likely to trigger safety filters
        """
        # Add context that this is about data analysis planning, not real-time queries
        return f"""The user wants to analyze data to answer this business question: "{intent}"

Note: This is a data product requirement planning session. The user is describing what kind of analysis they want to build, not asking for real-time information."""

    async def _reflect_on_questions(
        self, 
        questions: str, 
        conversation_context: str = "",
    ) -> str:
        """
        Apply self-correction/reflection to generated questions.
        
        Reviews questions for clarity, efficiency, and proper focus on
        product structure rather than tactical inputs.
        
        Args:
            questions: The generated questions to review
            conversation_context: Context from conversation history
            
        Returns:
            Refined questions (or original if already optimal)
        """
        if not self.enable_reflection:
            logger.debug("Question reflection disabled, returning original questions")
            return questions
        
        context_section = f"\n\nConversation context:\n{conversation_context}\n" if conversation_context else ""
        
        prompt = f"""Review these questions for defining a data product:

{questions}
{context_section}

As a critic, evaluate these questions:

1. **Avoid Tactical Inputs**: Do these avoid asking for specific entity IDs, current values, or instance-specific data?
2. **Efficiency**: Can any questions be combined or simplified?
3. **Clarity**: Are they clear and actionable?
4. **Product Focus**: Do they focus on product structure (what to build) vs execution inputs (how to run it)?

CRITICAL DISTINCTIONS:
- ❌ BAD: "What is your current market share?" (asking for specific value)
- ✅ GOOD: "What metrics should we calculate?" (asking about structure)

- ❌ BAD: "Which customers do you want to analyze?" (asking for specific entities)
- ✅ GOOD: "What customer segments should the tool support?" (asking about structure)

If the questions are already optimal, return them EXACTLY as provided.
If improvements are needed, provide the refined questions.

Return ONLY the questions (original or improved), nothing else:"""

        try:
            # Run Gemini API call in thread pool to avoid blocking async event loop
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,  # Lower temperature for focused critique
                ),
                safety_settings=self.safety_settings,
            )
            
            # Check if response was blocked
            if not response.candidates or not response.candidates[0].content.parts:
                logger.warning("Reflection blocked by safety filters, returning original questions")
                return questions
            
            refined_questions = response.text.strip()
            
            # Basic sanity check - if response is too short, probably an error
            if len(refined_questions) < len(questions) * 0.5:
                logger.warning("Reflection returned suspiciously short response, keeping original")
                return questions
            
            logger.info(f"Question reflection applied (original: {len(questions)} chars, refined: {len(refined_questions)} chars)")
            return refined_questions
            
        except Exception as e:
            logger.warning(f"Error during question reflection, returning original: {e}")
            return questions

    async def _synthesize_search_query(self, conversation_text: str) -> str:
        """
        Synthesize current data search query from full conversation context.
        
        This accumulates clarifications and details over time, building a richer
        search query as the conversation evolves.
        
        Args:
            conversation_text: Full conversation history
            
        Returns:
            Synthesized search query string incorporating all relevant context
        """
        prompt = f"""Based on this conversation between a data analyst and user:

{conversation_text}

What data is the user CURRENTLY trying to find or analyze?
Extract the current data need and respond with ONLY a short search query that incorporates ALL relevant context from the conversation.

IMPORTANT:
- Include clarifying details that would help find the right data
- Keep it concise (10-15 words max)
- Combine the main topic with relevant qualifiers
- Do not explain, just provide the search query

Examples:
- Initial: "should I trade for Sam Darnold?" + Clarification: "6 points per TD" 
  → "NFL quarterback Sam Darnold 6 point touchdown scoring fantasy"

- Initial: "customer churn" + Clarification: "subscription business" + "monthly data"
  → "customer churn subscription monthly retention data"

- Initial: "sales trends" + Clarification: "last quarter" + "by region"
  → "sales trends Q4 regional breakdown"

Current search query:"""

        try:
            # Run Gemini API call in thread pool to avoid blocking async event loop
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,  # Lower temp for focused extraction
                ),
                safety_settings=self.safety_settings,
            )
            
            if not response.candidates or not response.candidates[0].content.parts:
                logger.warning("Failed to synthesize search query from conversation")
                return ""
            
            query = response.text.strip()
            # Remove quotes if present
            query = query.strip('"').strip("'")
            
            logger.info(f"Synthesized search query: {query}")
            return query
            
        except Exception as e:
            logger.error(f"Error synthesizing search query: {e}")
            return ""
    
    def _has_conversation_pivoted(self, session: PlanningSession) -> bool:
        """
        Detect if the conversation has pivoted to a significantly different topic.
        
        Used primarily for logging and debugging. The synthesis method handles
        both pivots and clarifications naturally.
        
        Args:
            session: Planning session with conversation history
            
        Returns:
            True if conversation has pivoted to new topic
        """
        # Simple heuristic: if we have fewer than 2 turns, no pivot yet
        if not session.conversation_history or len(session.conversation_history) < 2:
            return False
        
        # Get latest user response (filter for user turns)
        user_turns = [turn for turn in session.conversation_history if turn.speaker == "user"]
        if not user_turns:
            return False
        
        latest_response = user_turns[-1].content
        
        # If latest response is very short (< 10 chars), probably just an answer
        if len(latest_response.strip()) < 10:
            return False
        
        # Check if latest response contains question words indicating new query
        question_indicators = ['should i', 'can i', 'how do', 'what is', 'who is', 'when should', 'why', 'where']
        latest_lower = latest_response.lower()
        
        has_question_pattern = any(indicator in latest_lower for indicator in question_indicators)
        
        # If it looks like a new question (has question pattern and is substantial)
        if has_question_pattern and len(latest_response.split()) > 5:
            logger.debug(f"Detected potential conversation pivot")
            return True
        
        return False
    
    def _build_detailed_data_section(
        self,
        search_results: list,
        datastore_context: str,
        analysis_result: PRPAnalysisResult,
    ) -> str:
        """
        Build detailed data requirements section with conceptual target views and data gaps.
        """
        # Format Target Views from analysis
        target_views_md = []
        for view in analysis_result.target_views:
            schema_md = "\n".join(
                f"  - `{col.get('column_name')}` ({col.get('data_type', 'UNKNOWN')}): {col.get('description', '')}"
                for col in view.get("schema", [])
            )
            calculated_fields_md = "\n".join(
                f"  - {field}" for field in view.get("calculated_fields", [])
            )
            
            view_md = f"""**View Name**: `{view.get('view_name')}`
- **Purpose**: {view.get('purpose')}
- **Grain**: {view.get('grain')}
- **Schema**:\n{schema_md}"""
            if calculated_fields_md:
                view_md += f"\n- **Calculated Fields**:\n{calculated_fields_md}"
            target_views_md.append(view_md)
        
        target_views_section = "\n\n---\n\n".join(target_views_md)

        # Format Data Gaps from analysis into a JSON block for machine parsing
        data_gaps_section = ""
        if analysis_result.data_gaps:
            gaps_dict = analysis_result.model_dump(include={'data_gaps'})
            gaps_json = json.dumps(gaps_dict, indent=2)
            data_gaps_section = f"""### Data Gaps and Limitations
```json
{gaps_json}
```"""

        # Build the "Available Source Data" section from search results
        available_sources = ""
        if search_results:
            # Deduplicate by table_id to avoid showing the same table multiple times
            seen_tables = {}
            for result in search_results:
                if not result.table_id:
                    continue
                # Skip if we've already added this table
                if result.table_id not in seen_tables:
                    seen_tables[result.table_id] = result
            
            source_sections = []
            for result in seen_tables.values():
                source_info = [f"- **`{result.table_id}`**"]
                
                if result.description:
                    desc_summary = result.description[:150] + "..." if len(result.description) > 150 else result.description
                    source_info.append(f"  - {desc_summary}")
                
                metadata_parts = []
                if result.row_count:
                    metadata_parts.append(f"{result.row_count:,} records")
                if result.column_count:
                    metadata_parts.append(f"{result.column_count} columns")
                if result.has_pii:
                    metadata_parts.append("contains PII")
                
                if metadata_parts:
                    source_info.append(f"  - Metadata: {', '.join(metadata_parts)}")
                
                if result.schema:
                    key_cols = [f"`{field.get('name')}`" for field in result.schema[:5]]
                    if key_cols:
                        more_indicator = f" (and {len(result.schema) - 5} more)" if len(result.schema) > 5 else ""
                        source_info.append(f"  - Key columns: {', '.join(key_cols)}{more_indicator}")
                
                source_sections.append("\n".join(source_info))
            
            available_sources = "\n\n".join(source_sections) if source_sections else ""
        
        if not available_sources:
            available_sources = "- No specific data catalog information available"

        return f"""### Target Views/Tables
{target_views_section}

{data_gaps_section}

### Available Source Data (For Reference)
The following existing tables could potentially be used as source data:

{available_sources}"""
    
    def _format_data_references(self, search_results: list) -> str:
        """
        Format table and field references from search results using structured schema.
        
        Creates a natural language summary of found tables and their key fields
        to make questions more data-aware and grounded.
        
        Args:
            search_results: List of SearchResult objects
            
        Returns:
            Natural language summary like: "table `nba_team_defend_stats` 
            with fields like `team_name`, `d_fg_pct`, and `pct_plusminus`"
        """
        if not search_results:
            return ""
        
        # Get first result with schema
        result = None
        for r in search_results:
            if r.schema:
                result = r
                break
        
        if not result or not result.table_id:
            return ""
        
        summary = f"table `{result.table_id}`"
        
        # Extract key fields (first 5 from schema)
        if result.schema:
            field_names = [f.get('name', '') for f in result.schema[:5] if f.get('name')]
            if field_names:
                if len(field_names) == 1:
                    field_text = f"`{field_names[0]}`"
                elif len(field_names) == 2:
                    field_text = f"`{field_names[0]}` and `{field_names[1]}`"
                else:
                    field_text = ", ".join(f"`{f}`" for f in field_names[:-1]) + f", and `{field_names[-1]}`"
                
                summary = f"{summary} with fields like {field_text}"
        
        return summary

    def _query_datastore(
        self, 
        query: str, 
        max_results: int = 5,
        enable_fanout: bool = True,
    ) -> tuple[str, str, list]:
        """
        Query Vertex AI Search datastore with intelligent fan-out.
        
        Args:
            query: Search query string
            max_results: Maximum number of results per query
            enable_fanout: Whether to use fan-out strategy on no results
            
        Returns:
            Tuple of (context_type, formatted_context, raw_results) where:
            - context_type: "exact_match", "related_match", or "no_match"
            - formatted_context: Context string for prompt
            - raw_results: List of SearchResult objects for structured access
        """
        if not self.vertex_search_client:
            return ("no_client", "", [])
        
        try:
            if not enable_fanout:
                # Simple single search (backward compatible)
                results = self.vertex_search_client.search(query, max_results)
                if results:
                    context = self.vertex_search_client.format_results_for_context(results, True)
                    return ("exact_match", context, results)
                else:
                    return ("no_match", "No matching data found in the catalog.", [])
            
            # Fan-out search strategy
            # 1. Generate related queries
            from .search_fanout import SearchFanoutGenerator
            
            fanout_gen = SearchFanoutGenerator(self)
            related_queries = fanout_gen.generate_related_queries(query, num_queries=4)
            
            if not related_queries:
                # Failed to generate related queries - fall back to simple search
                logger.warning("Failed to generate related queries, falling back to simple search")
                results = self.vertex_search_client.search(query, max_results)
                if results:
                    context = self.vertex_search_client.format_results_for_context(results, True)
                    return ("exact_match", context, results)
                else:
                    return ("no_match", "No matching data found in the catalog.", [])
            
            # 2. Execute fan-out search
            fanout_results = self.vertex_search_client.search_with_fanout(
                primary_query=query,
                related_queries=related_queries,
                max_results_per_query=3,
            )
            
            # 3. Format results based on what was found
            context_type, formatted_context = self.vertex_search_client.format_fanout_results(
                fanout_results=fanout_results,
                original_query=query,
            )
            
            # 4. Extract all results for structured access
            all_results = []
            all_results.extend(fanout_results.get("primary", []))
            for related_results in fanout_results.get("related", {}).values():
                all_results.extend(related_results)
            
            logger.info(f"Datastore query type: {context_type}")
            return (context_type, formatted_context, all_results)
            
        except Exception as e:
            logger.warning(f"Failed to query datastore: {e}", exc_info=True)
            return ("error", "", [])

    def _classify_product_type_sync(self, intent: str) -> str:
        """
        Classify what type of data product the user is requesting (synchronous version).
        
        Args:
            intent: User's initial intent
            
        Returns:
            Product type code: 'A' (one-time), 'B' (reusable), 'C' (framework), 'D' (monitoring)
        """
        prompt = f"""Based on this user intent:

"{intent}"

What type of data product are they asking for?

A) ONE-TIME ANALYSIS: Answer one specific question right now
B) REUSABLE TOOL: A tool to use repeatedly with different inputs  
C) DECISION FRAMEWORK: Systematic approach to recurring decisions
D) MONITORING: Ongoing tracking and updates

Respond with ONLY the letter (A, B, C, or D):"""
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(temperature=0.1),
                safety_settings=self.safety_settings,
            )
            
            if response.candidates and response.candidates[0].content.parts:
                result = response.text.strip().upper()
                # Extract first letter
                for char in result:
                    if char in ['A', 'B', 'C', 'D']:
                        logger.info(f"Classified product type: {char}")
                        return char
            
            # Default to one-time if unclear
            logger.warning("Could not classify product type, defaulting to 'A' (one-time)")
            return 'A'
            
        except Exception as e:
            logger.error(f"Error classifying product type: {e}")
            return 'A'  # Default to one-time
    
    async def generate_initial_questions(self, initial_intent: str) -> str:
        """
        Generate initial clarifying questions with intelligent fan-out.

        Args:
            initial_intent: The initial business intent from the user

        Returns:
            Up to 4 clarifying questions (preferably multiple choice)
        """
        # Preprocess to avoid safety filters
        preprocessed_intent = self._preprocess_intent(initial_intent)
        
        # Classify product type (run in thread pool to avoid blocking async event loop)
        product_type = await asyncio.to_thread(
            self._classify_product_type_sync,
            initial_intent
        )
        
        # Query datastore with fan-out (run in thread pool to avoid blocking async event loop)
        context_type, datastore_context, search_results = await asyncio.to_thread(
            self._query_datastore,
            initial_intent, 
            max_results=5,
            enable_fanout=True
        )
        
        # Extract analytical insights from search results
        insights = []
        for result in search_results:
            if result.analytical_insights:
                insights.extend(result.analytical_insights[:2])  # Top 2 per result
        
        # Build insights section if we have any
        insights_section = ""
        if insights:
            insights_section = f"""
EXAMPLE ANALYSES FROM CATALOG:
{chr(10).join(f'- {insight}' for insight in insights[:3])}

These show the types of analyses this data commonly supports.
"""
        
        # Build product-type-specific guidance
        if product_type == 'A':  # One-time analysis
            guidance = """This appears to be a ONE-TIME ANALYSIS request. Help them define:
- What specific question they're trying to answer
- What metrics/data would answer it
- What timeframe is relevant
- What would constitute a clear answer

DO NOT ask for specific input values like product names, customer IDs, dates, or entity identifiers."""
        
        elif product_type in ['B', 'C']:  # Reusable tool/framework
            guidance = """This appears to be a REUSABLE TOOL/FRAMEWORK request. Help them define:
- What type of comparisons or decisions this tool supports
- What metrics are most important for the comparison
- What parameters users would provide when using the tool
- What makes a "good" vs "bad" outcome
- How often this would be used

DO NOT ask for specific instance inputs - focus on the STRUCTURE of the tool."""
        
        elif product_type == 'D':  # Monitoring
            guidance = """This appears to be a MONITORING/DASHBOARD request. Help them define:
- What metrics to track over time
- What changes or thresholds trigger interest
- How often to update (daily, weekly, real-time)
- What alerts or notifications are needed

DO NOT ask for specific current values - focus on what to MONITOR."""
        
        else:
            guidance = ""
        
        # Build prompt based on context type
        if context_type == "exact_match":
            # Found exact match - guide toward that data
            # Extract data references for natural language integration
            data_summary = self._format_data_references(search_results)
            data_intro = f"I found {data_summary} that may be relevant." if data_summary else "I found some data that may be relevant."
            
            prompt = f"""You are an expert data analyst helping define a DATA PRODUCT.

User's intent: {preprocessed_intent}

{guidance}

AVAILABLE DATA (exact match):
{data_intro}

{datastore_context}
{insights_section}

Ask up to 3 questions to define the DATA PRODUCT STRUCTURE (not to gather specific inputs).

FOCUS ON:
- What type of analysis framework to build
- What calculations, aggregations, or transformations are needed
- What time granularities or periods to calculate
- What comparisons or benchmarks to compute
- What defines success or a good outcome

DO NOT ASK FOR:
- Specific product names, customer IDs, or entity identifiers
- Current business metrics or account details
- Specific dates or time periods for this run
- Any values that would be "inputs" to the tool rather than part of its definition


IMPORTANT: Prefer multiple choice questions (a, b, c, d format).

Generate your questions now:"""

        elif context_type == "related_match":
            # Found related data - present as alternative
            # Extract data references for natural language integration
            data_summary = self._format_data_references(search_results)
            data_intro = f"I found {data_summary} that might be related." if data_summary else "I found some related data."
            
            prompt = f"""You are an expert data analyst helping define a DATA PRODUCT.

User's intent: {preprocessed_intent}

{guidance}

RELATED DATA FOUND:
{data_intro}

{datastore_context}
{insights_section}

Ask up to 3 questions to define the DATA PRODUCT:
1. Whether the related data could meet their needs
2. What adjustments to their goal might make it feasible
3. What metrics and structure would work best

FOCUS ON: Product structure, not tactical inputs
DO NOT ASK FOR: Specific entity names, current values, or instance-specific data

Present options as multiple choice where appropriate.

Generate your questions now:"""

        elif context_type == "no_match":
            # No data found - ask for more context
            prompt = f"""You are an expert data analyst helping define a DATA PRODUCT.

User's intent: {preprocessed_intent}

{guidance}

{datastore_context}

Since no exact matching data was found, ask 2-3 questions to understand the DATA PRODUCT they want:
1. More context about their goal
2. What type of analysis framework they need (one-time vs. reusable)
3. What metrics and outcomes matter most

FOCUS ON: Product structure, not tactical inputs
DO NOT ASK FOR: Specific entity names, current values, or instance-specific data

Present options as multiple choice where appropriate.

Generate your questions now:"""

        else:  # no_client or error
            # Fallback to basic questions
            prompt = f"""You are an expert data analyst. A user has provided this intent:

{preprocessed_intent}

Ask up to 3 clarifying questions about their data requirements.
Focus on: metrics, dimensions, filters, comparisons, timeline.

Generate your questions now:"""

        try:
            # Apply context if available
            prompt = self._build_prompt_with_context(prompt)
            
            # Run Gemini API call in thread pool to avoid blocking async event loop
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.temperature,
                ),
                safety_settings=self.safety_settings,
            )
            
            # Debug logging
            logger.debug(f"Response candidates: {len(response.candidates) if response.candidates else 0}")
            if response.candidates:
                finish_reason = response.candidates[0].finish_reason
                logger.debug(f"Finish reason: {finish_reason} (name: {finish_reason.name if hasattr(finish_reason, 'name') else 'unknown'})")
                logger.debug(f"Safety ratings: {response.candidates[0].safety_ratings}")
            
            # Check if response was blocked
            if not response.candidates or not response.candidates[0].content.parts:
                finish_reason = response.candidates[0].finish_reason if response.candidates else None
                safety_ratings = response.candidates[0].safety_ratings if response.candidates else []
                
                # Log complete response details for debugging
                logger.error("=" * 80)
                logger.error("GEMINI API RESPONSE BLOCKED")
                logger.error("=" * 80)
                logger.error(f"Original intent: {initial_intent}")
                logger.error(f"Finish reason: {finish_reason} (raw value: {finish_reason})")
                logger.error(f"Finish reason name: {finish_reason.name if hasattr(finish_reason, 'name') else 'N/A'}")
                logger.error(f"Number of candidates: {len(response.candidates) if response.candidates else 0}")
                
                if response.candidates:
                    candidate = response.candidates[0]
                    logger.error(f"Candidate finish_reason: {candidate.finish_reason}")
                    logger.error(f"Candidate safety_ratings: {candidate.safety_ratings}")
                    logger.error(f"Candidate content: {candidate.content}")
                    logger.error(f"Candidate content.parts: {candidate.content.parts if hasattr(candidate.content, 'parts') else 'N/A'}")
                
                # Try to get additional response metadata
                logger.error(f"Response prompt_feedback: {response.prompt_feedback if hasattr(response, 'prompt_feedback') else 'N/A'}")
                logger.error(f"Full response object: {response}")
                logger.error("=" * 80)
                
                if finish_reason == 3:  # SAFETY (correct enum value)
                    logger.warning("Response blocked by safety filters despite BLOCK_NONE setting")
                    raise ValueError(
                        "The response was blocked by Gemini's safety filters. This can happen with:\n"
                        "  • Questions asking for real-time/recent information (e.g., 'last week')\n"
                        "  • Sports scores or betting-related queries\n"
                        "  • Specific names of people or organizations\n\n"
                        "Try rephrasing your intent to focus on the data analysis need:\n"
                        "  ✗ 'What NFL team had the highest score last week?'\n"
                        "  ✓ 'I want to analyze NFL team performance metrics'\n"
                        "  ✓ 'I need a dashboard showing team scores and rankings'"
                    )
                elif finish_reason == 2:  # MAX_TOKENS
                    logger.warning("Response exceeded max token limit")
                    raise ValueError(
                        "The AI response was too long. This is an internal error - please try again."
                    )
                else:
                    logger.warning(f"Response blocked with finish_reason: {finish_reason}")
                    raise ValueError(
                        f"The AI couldn't generate a response (reason: {finish_reason}). "
                        "Please try rephrasing your intent."
                    )
            
            questions = response.text.strip()
            logger.debug(f"Generated initial questions ({len(questions)} chars)")
            
            # Apply reflection to refine questions
            refined_questions = await self._reflect_on_questions(
                questions=questions,
                conversation_context=f"Initial intent: {initial_intent}",
            )
            
            return refined_questions
        except ValueError:
            # Re-raise ValueError (our custom errors)
            raise
        except Exception as e:
            logger.error(f"Error generating initial questions: {e}", exc_info=True)
            raise

    async def generate_follow_up_questions(self, session: PlanningSession) -> tuple[str, bool]:
        """
        Generate follow-up questions based on conversation history.

        Args:
            session: The planning session with conversation history

        Returns:
            Tuple of (questions text, is_complete flag)
            - questions: The follow-up questions to ask
            - is_complete: True if requirements are sufficient
        """
        conversation_text = session.get_conversation_text()
        
        # Always synthesize search query from full conversation
        # This naturally accumulates context (clarifications) and handles pivots
        logger.info("Synthesizing search query from full conversation context")
        search_query = await self._synthesize_search_query(conversation_text)
        
        # Fall back to initial intent if synthesis fails
        if not search_query:
            logger.warning("Synthesis failed, falling back to initial intent")
            initial_intent = session.initial_intent if hasattr(session, 'initial_intent') else ""
            search_query = initial_intent
        
        # Log if we detected a topic pivot (for debugging/telemetry)
        if self._has_conversation_pivoted(session):
            logger.info(f"Topic pivot detected - synthesized query: {search_query}")
        else:
            logger.info(f"Context accumulation - synthesized query: {search_query}")
        
        # Query datastore with synthesized search query (run in thread pool to avoid blocking async event loop)
        if search_query:
            context_type, datastore_context, search_results = await asyncio.to_thread(
                self._query_datastore,
                search_query,
                max_results=5,
                enable_fanout=True
            )
        else:
            context_type, datastore_context, search_results = ("no_client", "", [])
        
        # Extract available fields from search results for validation
        available_fields = set()
        for result in search_results:
            if result.schema:
                for field in result.schema:
                    if 'name' in field:
                        available_fields.add(field['name'])
        
        # Build schema section if we have fields
        schema_section = ""
        if available_fields:
            field_list = ', '.join(sorted(list(available_fields)[:15]))  # Show up to 15 fields
            schema_section = f"""

AVAILABLE FIELDS IN DATA:
{field_list}

Use these to validate what metrics and dimensions are actually available.
"""
        
        # Build prompt with datastore context
        context_section = f"\n\nAVAILABLE DATA:\n{datastore_context}{schema_section}\n" if datastore_context else "\n"

        prompt = f"""You are an expert data analyst defining a DATA PRODUCT (not answering a one-time question).

Conversation so far:
{conversation_text}
{context_section}

Your task: Determine if you have enough information to define the DATA PRODUCT STRUCTURE.

REMEMBER: You're defining WHAT TO BUILD, not gathering inputs to run it.

A complete Data Product definition requires:
1. **Product Type**: One-time analysis, reusable tool, decision framework, or dashboard
2. **Objective**: What business question or decision this supports
3. **Key Metrics**: What measurements matter (not specific values, but which metrics to include)
4. **Dimensions**: What breakdowns or comparisons (not specific entities, but what types)
5. **Success Criteria**: What makes a good/useful output
6. **Frequency**: One-time or recurring? If recurring, how often?

AVOID asking for:
- Specific entity identifiers (product names, customer IDs, user IDs)
- Current state or starting values
- Specific time windows for "this run"
- Any inputs that would be provided when USING the tool

GOOD QUESTIONS (product structure):
- "What metrics best measure success for this comparison?"
- "Should this be a one-time analysis or a reusable weekly tool?"
- "What makes a 'good' vs 'bad' outcome in your evaluation?"

BAD QUESTIONS (tactical inputs):
- "What is your current market share?" ❌
- "What date range do you want to analyze?" ❌  
- "Which specific customer segment are you targeting?" ❌

Based on available data and the conversation so far:
- Understand if we have data to build this product
- Guide them toward feasible product designs
- Suggest alternative approaches if exact data doesn't exist

If you have SUFFICIENT information to define the data product structure, respond with exactly: "COMPLETE"

If you MUST ask more questions (only for fundamental product definition gaps), ask up to 3:
- STRONGLY prefer multiple choice
- Focus on PRODUCT STRUCTURE, not execution inputs
- Use available data to ask informed questions

Provide your response now (either "COMPLETE" or your questions):"""

        try:
            # Apply context if available
            prompt = self._build_prompt_with_context(prompt)
            
            # Run Gemini API call in thread pool to avoid blocking async event loop
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.temperature,
                ),
                safety_settings=self.safety_settings,
            )
            
            # Check if response was blocked
            if not response.candidates or not response.candidates[0].content.parts:
                finish_reason = response.candidates[0].finish_reason if response.candidates else None
                
                # Log complete response details for debugging
                logger.error("=" * 80)
                logger.error("GEMINI API RESPONSE BLOCKED (follow-up questions)")
                logger.error("=" * 80)
                logger.error(f"Finish reason: {finish_reason} (raw value: {finish_reason})")
                logger.error(f"Finish reason name: {finish_reason.name if hasattr(finish_reason, 'name') else 'N/A'}")
                
                if response.candidates:
                    candidate = response.candidates[0]
                    logger.error(f"Candidate finish_reason: {candidate.finish_reason}")
                    logger.error(f"Candidate safety_ratings: {candidate.safety_ratings}")
                    logger.error(f"Candidate content: {candidate.content}")
                
                logger.error(f"Response prompt_feedback: {response.prompt_feedback if hasattr(response, 'prompt_feedback') else 'N/A'}")
                logger.error(f"Full response object: {response}")
                logger.error("=" * 80)
                
                if finish_reason == 3:  # SAFETY (correct enum value)
                    logger.warning("Response blocked by safety filters")
                    raise ValueError(
                        "The response was blocked by safety filters. Please try rephrasing "
                        "your responses in more general terms."
                    )
                elif finish_reason == 2:  # MAX_TOKENS
                    logger.warning("Response exceeded max token limit")
                    raise ValueError(
                        "The AI response was too long. This is an internal error - please try again."
                    )
                else:
                    logger.warning(f"Response blocked with finish_reason: {finish_reason}")
                    raise ValueError(
                        f"The AI couldn't generate a response (reason: {finish_reason}). "
                        "Please try rephrasing."
                    )
            
            result = response.text.strip()

            # Check if complete
            if result.upper().startswith("COMPLETE"):
                logger.info("Requirements gathering complete")
                return ("Requirements gathering complete!", True)

            logger.debug(f"Generated follow-up questions ({len(result)} chars)")
            
            # Apply reflection to refine questions
            refined_questions = await self._reflect_on_questions(
                questions=result,
                conversation_context=conversation_text,
            )
            
            return (refined_questions, False)

        except ValueError:
            # Re-raise ValueError (our custom errors)
            raise
        except Exception as e:
            logger.error(f"Error generating follow-up questions: {e}", exc_info=True)
            raise

    async def _analyze_requirements(self, session: PlanningSession) -> PRPAnalysisResult:
        """
        First pass of PRP generation: Analyze conversation and extract structured data.

        Args:
            session: The planning session with conversation history

        Returns:
            A PRPAnalysisResult object with structured target views and data gaps.
        """
        conversation_text = session.get_conversation_text()
        prompt = f"""Analyze the following conversation and generate a structured JSON object that defines the data requirements.

Conversation:
{conversation_text}

Your task is to identify two key things:
1.  **Target Views**: The conceptual data views the user wants to create.
2.  **Data Gaps**: Any critical missing information needed to build those views.

Respond with a JSON object that follows this exact schema:
{{
    "target_views": [
        {{
            "view_name": "...",
            "purpose": "...",
            "grain": "...",
            "schema": [
                {{"column_name": "...", "data_type": "...", "description": "..."}}
            ],
            "calculated_fields": ["..."]
        }}
    ],
    "data_gaps": [
        {{
            "gap_id": "gap_01",
            "description": "e.g., Missing source for game outcomes (Win/Loss/Push)",
            "target_view": "e.g., live_bet_performance",
            "required_information": "A source table containing final game scores (e.g., team_score, opponent_score) and a game identifier to join on."
        }}
    ]
}}

Provide ONLY the JSON object in your response.
"""
        try:
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1, response_mime_type="application/json"
                ),
                safety_settings=self.safety_settings,
            )
            if not response.candidates or not response.candidates[0].content.parts:
                raise ValueError("Failed to analyze requirements: empty response from Gemini.")
            
            analysis_json = response.text.strip()
            return PRPAnalysisResult.parse_raw(analysis_json)
        except Exception as e:
            logger.error(f"Error during requirements analysis: {e}", exc_info=True)
            raise

    async def generate_data_prp(self, session: PlanningSession) -> str:
        """
        Generate a complete Data Product Requirement Prompt from conversation.

        Args:
            session: The planning session with conversation history

        Returns:
            The generated Data PRP as markdown text
        """
        conversation_text = session.get_conversation_text()
        
        # Step 1: Analyze requirements to get structured data
        analysis_result = await self._analyze_requirements(session)

        # Step 2: Use the analysis to generate the final PRP
        
        # Always synthesize from full conversation for final assessment
        # This ensures we capture the complete evolved understanding with all details
        logger.info("Synthesizing search query for final data availability assessment")
        search_query = await self._synthesize_search_query(conversation_text)
        
        # Fall back to initial intent if synthesis fails
        if not search_query:
            initial_intent = session.initial_intent if hasattr(session, 'initial_intent') else ""
            search_query = initial_intent
            
            # Log assumption about search query fallback
            session.add_assumption(
                "Used initial user intent for data search as conversation synthesis "
                "did not produce a refined query."
            )
        
        if search_query:
            context_type, datastore_context, search_results = await asyncio.to_thread(
                self._query_datastore,
                search_query,
                max_results=5,
                enable_fanout=True
            )
            
            # Log assumption if no datastore results found
            if not search_results or len(search_results) == 0:
                session.add_assumption(
                    "No matching data assets found in catalog. Data requirements are based on "
                    "user description only."
                )
        else:
            context_type, datastore_context, search_results = ("no_client", "", [])
        
        # Build detailed data requirements section using the structured analysis
        detailed_data_section = self._build_detailed_data_section(
            search_results, datastore_context, analysis_result
        )
        
        # Build prompt with datastore context
        context_section = f"\n\nAVAILABLE DATA:\n{datastore_context}\n" if datastore_context else "\n"
        
        # Build assumptions section for prompt
        tracked_assumptions = ""
        if session.assumptions:
            tracked_assumptions = "\n\nTracked Assumptions from Planning Process:\n" + "\n".join(
                f"- {assumption}" for assumption in session.assumptions
            )

        prompt = f"""Generate a Data Product Requirement Prompt (Data PRP) from this conversation.

Conversation:
{conversation_text}
{context_section}{tracked_assumptions}

CRITICAL: This PRP defines a DATA PRODUCT, not a one-time query result.

Structure your Data PRP with these sections:

# Data Product Requirement Prompt

## 1. Product Type
Is this a one-time analysis, reusable tool, decision framework, or dashboard?

## 2. Business Objective
What business need or decision does this support? (Not the specific instance, but the general need)

## 3. Product Functionality
What does this data product DO? Describe its capabilities.

## 4. Key Metrics
What measurements/calculations are needed? (Define the metrics, not specific values)

## 5. Dimensions & Breakdowns  
What categories, segments, or comparisons? (Define the structure, not specific entities)

## 6. Success Criteria
What makes this product useful? What defines a "good" output?

## 7. Usage Pattern
- **Frequency**: One-time, daily, weekly, on-demand?
- **Audience**: Who uses this?
- **Triggers**: What prompts someone to use it?

## 8. Example Usage Scenario
Show how someone would USE this product:
- What inputs they provide
- What outputs they get
- How they make decisions with it

## 9. Data Requirements

{detailed_data_section}

**Assessment:**
- How well do the available source tables cover the target view requirements?
- What transformations or joins are needed to build the target views?
- What data gaps or limitations exist?
- What assumptions are being made about data availability and quality?
- Are there any data quality considerations (PII, PHI, freshness, completeness)?

## 10. Assumptions & Defaults

Document all assumptions and default choices made during planning. Use the format [ASSUMPTION-01], [ASSUMPTION-02], etc.

REQUIRED: Include these tracked assumptions from the planning process:
{tracked_assumptions if session.assumptions else "- No assumptions were tracked during the planning process."}

ALSO identify and list any additional implicit assumptions from the conversation, such as:
- Default values chosen when user didn't specify (e.g., "closing" vs "opening" lines)
- Data availability assumptions (e.g., assuming certain fields exist)
- Time period defaults (e.g., defaulting to "current season" if not specified)
- Calculation method assumptions (e.g., which confidence metric to use)
- Scope assumptions (e.g., "all games" vs specific subset)

Example format:
- [ASSUMPTION-01]: User did not specify opening vs closing Vegas lines. Defaulted to 'closing_vegas_spread' for backtesting stability.
- [ASSUMPTION-02]: Assumed game_id can be used as a consistent primary key across all required data sources.

---

IMPORTANT DISTINCTIONS:
- **Product Definition**: "Compare any two products using revenue, growth rate, and market penetration"
- **NOT Instance Execution**: "Compare Product A to Product B for Q3 results"

- **Product Definition**: "Metrics include: revenue, growth rate, customer acquisition cost, market share"
- **NOT Instance Values**: "Product A has $2.8M revenue and 15% growth"

IMPORTANT FOR SECTION 9 (Data Requirements):
- Define CONCEPTUAL target views/tables that need to be CREATED (don't exist yet)
- These represent the desired OUTPUT data structure for the data product
- Use "Available Source Data" section to show what existing tables could be used as INPUTS
- Target views should answer the analytical questions from the requirements
- Do NOT directly copy existing table schemas into the target views section

NOTE: This Data PRP focuses on DATA requirements only. Visualization and presentation format will be handled by the Presentation Planning Agent.

Generate the complete Data PRP now, following the exact format above:"""

        try:
            # Apply context if available
            prompt = self._build_prompt_with_context(prompt)
            
            # Run Gemini API call in thread pool to avoid blocking async event loop
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,  # Lower temperature for structured output
                ),
                safety_settings=self.safety_settings,
            )
            
            # Check if response was blocked
            if not response.candidates or not response.candidates[0].content.parts:
                finish_reason = response.candidates[0].finish_reason if response.candidates else None
                
                # Log complete response details for debugging
                logger.error("=" * 80)
                logger.error("GEMINI API RESPONSE BLOCKED (Data PRP generation)")
                logger.error("=" * 80)
                logger.error(f"Finish reason: {finish_reason} (raw value: {finish_reason})")
                logger.error(f"Finish reason name: {finish_reason.name if hasattr(finish_reason, 'name') else 'N/A'}")
                
                if response.candidates:
                    candidate = response.candidates[0]
                    logger.error(f"Candidate finish_reason: {candidate.finish_reason}")
                    logger.error(f"Candidate safety_ratings: {candidate.safety_ratings}")
                    logger.error(f"Candidate content: {candidate.content}")
                
                logger.error(f"Response prompt_feedback: {response.prompt_feedback if hasattr(response, 'prompt_feedback') else 'N/A'}")
                logger.error(f"Full response object: {response}")
                logger.error("=" * 80)
                
                if finish_reason == 3:  # SAFETY (correct enum value)
                    logger.warning("Response blocked by safety filters")
                    raise ValueError(
                        "The Data PRP generation was blocked by safety filters. "
                        "Please review your conversation for potentially sensitive content."
                    )
                elif finish_reason == 2:  # MAX_TOKENS
                    logger.warning("Data PRP generation exceeded max token limit")
                    raise ValueError(
                        "The Data PRP response was too long. This is an internal error - please try again."
                    )
                else:
                    logger.warning(f"Response blocked with finish_reason: {finish_reason}")
                    raise ValueError(
                        f"The AI couldn't generate the Data PRP (reason: {finish_reason})."
                    )
            
            data_prp = response.text.strip()
            logger.info(f"Generated Data PRP ({len(data_prp)} chars)")
            return data_prp
        except ValueError:
            # Re-raise ValueError (our custom errors)
            raise
        except Exception as e:
            logger.error(f"Error generating Data PRP: {e}", exc_info=True)
            raise

