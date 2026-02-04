# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from typing import Literal
import os
from dotenv import load_dotenv

load_dotenv()  # loads .env file if it exists

app = FastAPI(
    title="Simple Text Analysis API",
    description="Summarization & Sentiment using Gemini",
    version="2025.02",
)

# Enable CORS to allow requests from other origins (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set")

genai.configure(api_key=GEMINI_API_KEY)

# Use the most recent stable model as of early 2025
# You can change to "gemini-1.5-flash-latest" or "gemini-2.0-flash" etc. later
MODEL_NAME = "gemini-flash-latest"

model = genai.GenerativeModel(MODEL_NAME)


# ────────────────────────────────────────────────
# Request / Response models
# ────────────────────────────────────────────────

class TextRequest(BaseModel):
    text: str


class SummarizeResponse(BaseModel):
    summary: str
    model_used: str


class SentimentResponse(BaseModel):
    sentiment: Literal["POSITIVE", "NEGATIVE", "NEUTRAL"]
    confidence: float  # 0–1
    explanation: str
    model_used: str


# ────────────────────────────────────────────────
# Routes
# ────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": genai.utils.get_current_utc_iso()}


@app.post("/summarize", response_model=SummarizeResponse)
async def summarize(request: TextRequest):
    if not request.text.strip():
        raise HTTPException(400, "Text cannot be empty")

    try:
        prompt = f"""Summarize the following text concisely in 1–3 sentences:

{request.text}

Summary:"""

        response = model.generate_content(prompt)

        summary_text = response.text.strip()

        return SummarizeResponse(
            summary=summary_text,
            model_used=MODEL_NAME
        )

    except Exception as e:
        raise HTTPException(500, f"Gemini error: {str(e)}")


@app.post("/sentiment", response_model=SentimentResponse)
async def sentiment(request: TextRequest):
    if not request.text.strip():
        raise HTTPException(400, "Text cannot be empty")

    try:
        prompt = f"""Analyze the sentiment of the following text.
Return ONLY JSON in this exact format:

{{
  "sentiment": "POSITIVE" | "NEGATIVE" | "NEUTRAL",
  "confidence": 0.0–1.0,
  "explanation": "one short sentence explaining your choice"
}}

Text:
{request.text}
"""

        response = model.generate_content(prompt)

        # Gemini often returns valid JSON when asked strictly
        import json
        try:
            data = json.loads(response.text.strip())
        except json.JSONDecodeError:
            # fallback — sometimes it adds markdown or extra text
            cleaned = response.text.strip().removeprefix("```json").removesuffix("```").strip()
            data = json.loads(cleaned)

        return SentimentResponse(
            sentiment=data["sentiment"],
            confidence=float(data["confidence"]),
            explanation=data["explanation"],
            model_used=MODEL_NAME
        )

    except Exception as e:
        raise HTTPException(500, f"Gemini error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)