"""Manual Gemini connectivity smoke check.

Run explicitly from the project root with:
    python scripts/check_gemini.py

This script makes one real Gemini API request. It is intentionally not part of
normal application startup or automated test discovery.
"""

from dotenv import load_dotenv
from google import genai


def check_gemini_connection():
    """Send a small request to confirm that local Gemini credentials work."""
    load_dotenv()

    client = genai.Client()
    response = client.interactions.create(
        model="gemini-3.8-flash",
        input="Say hello in one short sentence."
    )

    print("Gemini connection successful.")
    print("Response:", response.output_text)


if __name__ == "__main__":
    check_gemini_connection()
