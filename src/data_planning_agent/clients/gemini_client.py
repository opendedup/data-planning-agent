"""
Gemini API Client

Client for interacting with Google's Gemini API for conversational
requirement gathering and Data PRP generation.
"""

import logging
from typing import Optional

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from ..models.prp_schema import DataProductRequirementPrompt
from ..models.session import PlanningSession

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
        temperature: float = 0.7,
        max_retries: int = 3,
        context: Optional[str] = None,
    ):
        """
        Initialize Gemini client.

        Args:
            api_key: Gemini API key
            model_name: Model name to use
            temperature: Sampling temperature (0.0-1.0)
            max_retries: Maximum retry attempts
            context: Optional organizational context to prepend to all prompts
        """
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries
        self.context = context

        # Configure API
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

        # Disable safety filters for all categories
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        if context:
            logger.info(
                f"Initialized Gemini client with model: {model_name} "
                f"(with {len(context)} chars of context, safety filters disabled)"
            )
        else:
            logger.info(f"Initialized Gemini client with model: {model_name} (safety filters disabled)")

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

    async def generate_initial_questions(self, initial_intent: str) -> str:
        """
        Generate initial clarifying questions based on the user's intent.

        Args:
            initial_intent: The initial business intent from the user

        Returns:
            Up to 4 clarifying questions (preferably multiple choice)
        """
        prompt = f"""You are an expert data analyst helping to gather requirements for a data product. A user has provided this initial business intent:

"{initial_intent}"

Your task is to ask up to 4 focused clarifying questions to understand the requirements better. Follow these guidelines:

1. Ask up to 4 questions at a time (maximize efficiency)
2. STRONGLY prefer multiple choice questions (format: a) option, b) option, c) option, d) Other/Custom)
3. Only use open-ended questions when specific details absolutely need to be collected
4. Focus on these categories in order of priority:
   - Objective: What is the business goal?
   - Audience: Who will use this?
   - Key Metrics: What measurements are needed?
   - Dimensions: How should data be segmented?
   - Filters: What conditions apply?
   - Comparisons: Are comparisons needed?
   - Timeline: When is this needed?

Format your response as a numbered list with clear multiple choice options where appropriate.

Example format:
1. What is the primary audience for this analysis?
   a) Executives (high-level summary)
   b) Regional managers (summary + detail)
   c) Data analysts (detailed data)
   d) Other (please specify)

2. What key metrics define "trending" for your use case?
   a) Unit sales volume
   b) Revenue growth
   c) Profit margin
   d) Multiple metrics (please specify)

Generate your questions now:"""

        try:
            # Apply context if available
            prompt = self._build_prompt_with_context(prompt)
            
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=1000,
                ),
                safety_settings=self.safety_settings,
            )
            
            # Check if response was blocked
            if not response.candidates or not response.candidates[0].content.parts:
                finish_reason = response.candidates[0].finish_reason if response.candidates else None
                if finish_reason == 2:  # SAFETY
                    logger.warning("Response blocked by safety filters")
                    raise ValueError(
                        "The response was blocked by safety filters. Please try rephrasing "
                        "your intent in more general terms, avoiding specific names or "
                        "potentially sensitive topics."
                    )
                else:
                    logger.warning(f"Response blocked with finish_reason: {finish_reason}")
                    raise ValueError(
                        f"The AI couldn't generate a response (reason: {finish_reason}). "
                        "Please try rephrasing your intent."
                    )
            
            questions = response.text.strip()
            logger.debug(f"Generated initial questions ({len(questions)} chars)")
            return questions
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

        prompt = f"""You are an expert data analyst gathering requirements for a data product. You've been having a conversation with a user.

Here is the conversation so far:

{conversation_text}

Your task is to determine if you have enough information to write a complete Data Product Requirement Prompt (Data PRP), or if you need to ask more questions.

A complete Data PRP requires:
1. Clear objective and business goal
2. Target audience identified
3. Key metrics defined
4. Dimensions/breakdowns specified
5. Any necessary filters
6. Timeline/delivery expectations

If you have SUFFICIENT information, respond with exactly: "COMPLETE"

If you need MORE information, ask up to 4 focused clarifying questions following these guidelines:
- STRONGLY prefer multiple choice questions
- Only ask about topics not already covered
- Focus on filling gaps in the requirements
- Be concise and specific

Provide your response now (either "COMPLETE" or your questions):"""

        try:
            # Apply context if available
            prompt = self._build_prompt_with_context(prompt)
            
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=1000,
                ),
                safety_settings=self.safety_settings,
            )
            
            # Check if response was blocked
            if not response.candidates or not response.candidates[0].content.parts:
                finish_reason = response.candidates[0].finish_reason if response.candidates else None
                if finish_reason == 2:  # SAFETY
                    logger.warning("Response blocked by safety filters")
                    raise ValueError(
                        "The response was blocked by safety filters. Please try rephrasing "
                        "your responses in more general terms."
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
            return (result, False)

        except ValueError:
            # Re-raise ValueError (our custom errors)
            raise
        except Exception as e:
            logger.error(f"Error generating follow-up questions: {e}", exc_info=True)
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

        prompt = f"""You are an expert data analyst. You've had a conversation with a user to gather requirements for a data product. Based on this conversation, generate a complete Data Product Requirement Prompt (Data PRP).

Here is the conversation:

{conversation_text}

Generate a Data PRP in this EXACT markdown format:

# Data Product Requirement Prompt

## 1. Executive Summary

* **Objective:** [one-sentence summary of the business goal]
* **Target Audience:** [who will use this analysis]
* **Key Question:** [the primary question to be answered]

## 2. Business Context

[Write a detailed paragraph explaining the business scenario, the problem to be solved, and how the insights will be used to drive decisions. This should be 3-5 sentences providing context.]

## 3. Data Requirements

### 3.1. Key Metrics

[List the specific metrics needed, one per bullet point]

### 3.2. Dimensions & Breakdowns

[List how data should be segmented or grouped, one per bullet point]

### 3.3. Filters

[List any conditions or constraints, one per bullet point. If none, write "* No specific filters"]

## 4. Success Criteria

* **Primary Metric:** [main success indicator]
* **Timeline:** [delivery and update expectations]

Generate the complete Data PRP now, following the exact format above:"""

        try:
            # Apply context if available
            prompt = self._build_prompt_with_context(prompt)
            
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,  # Lower temperature for structured output
                    max_output_tokens=2000,
                ),
                safety_settings=self.safety_settings,
            )
            
            # Check if response was blocked
            if not response.candidates or not response.candidates[0].content.parts:
                finish_reason = response.candidates[0].finish_reason if response.candidates else None
                if finish_reason == 2:  # SAFETY
                    logger.warning("Response blocked by safety filters")
                    raise ValueError(
                        "The Data PRP generation was blocked by safety filters. "
                        "Please review your conversation for potentially sensitive content."
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

