"""Simple test of Gemini API to debug the issue."""
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

# Test 1: Simple prompt
print("Test 1: Simple prompt...")
response1 = model.generate_content(
    "What is 2+2?",
    generation_config=genai.types.GenerationConfig(temperature=0.7, max_output_tokens=1000),
    safety_settings=safety_settings,
)
print(f"Response: {response1.text}")
print(f"Finish reason: {response1.candidates[0].finish_reason.name}\n")

# Test 2: Aaron Rodgers prompt
print("Test 2: Aaron Rodgers prompt...")
response2 = model.generate_content(
    "what players should I trade Aaron Rodgers for?",
    generation_config=genai.types.GenerationConfig(temperature=0.7, max_output_tokens=1000),
    safety_settings=safety_settings,
)
print(f"Has candidates: {bool(response2.candidates)}")
if response2.candidates:
    print(f"Finish reason: {response2.candidates[0].finish_reason.name}")
    print(f"Has content parts: {bool(response2.candidates[0].content.parts)}")
    if response2.candidates[0].content.parts:
        print(f"Response: {response2.text}")
    else:
        print("No content parts!")
        print(f"Safety ratings: {response2.candidates[0].safety_ratings}")

