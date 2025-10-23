"""Test with rephrased prompts."""
import os
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

load_dotenv()

# Configure API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-pro")

# Safety settings
safety_settings = [
    {"category": HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": HarmBlockThreshold.BLOCK_NONE},
    {"category": HarmCategory.HARM_CATEGORY_HATE_SPEECH, "threshold": HarmBlockThreshold.BLOCK_NONE},
    {"category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, "threshold": HarmBlockThreshold.BLOCK_NONE},
    {"category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, "threshold": HarmBlockThreshold.BLOCK_NONE},
]

# Test different phrasings
test_prompts = [
    "what players should I trade Aaron Rodgers for?",  # Original
    "what players should I trade a veteran quarterback for?",  # Generic
    "I want to analyze player trading scenarios for NFL quarterbacks",  # More formal
    "Help me evaluate quarterback trading options in professional football",  # Even more formal
]

for i, prompt in enumerate(test_prompts, 1):
    print(f"\nTest {i}: {prompt[:60]}...")
    try:
        response = model.generate_content(
            f"User query: {prompt}\n\nGenerate 2-3 clarifying questions about this data analysis request.",
            generation_config=genai.types.GenerationConfig(temperature=0.7, max_output_tokens=1000),
            safety_settings=safety_settings,
        )
        if response.candidates and response.candidates[0].content.parts:
            print(f"✓ SUCCESS - Finish reason: {response.candidates[0].finish_reason.name}")
            print(f"Response preview: {response.text[:100]}...")
        else:
            print(f"✗ FAILED - Finish reason: {response.candidates[0].finish_reason.name}")
            print(f"Has content parts: {bool(response.candidates[0].content.parts)}")
    except Exception as e:
        print(f"✗ ERROR: {e}")

