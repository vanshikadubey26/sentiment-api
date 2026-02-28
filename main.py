import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Literal
from openai import OpenAI

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CommentRequest(BaseModel):
    comment: str

class SentimentResponse(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    rating: int

@app.post("/comment", response_model=SentimentResponse)
async def analyze_comment(request: CommentRequest):

    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="API key missing")

        client = OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Return ONLY valid JSON with keys sentiment and rating."
                },
                {
                    "role": "user",
                    "content": f"""
Analyze sentiment:

Comment: {request.comment}

Return exactly:
{{
  "sentiment": "positive | negative | neutral",
  "rating": 1-5
}}
"""
                }
            ],
            temperature=0
        )

        content = response.choices[0].message.content.strip()

        result = json.loads(content)

        # Safety validation
        if result["sentiment"] not in ["positive", "negative", "neutral"]:
            raise ValueError("Invalid sentiment")

        if not (1 <= int(result["rating"]) <= 5):
            raise ValueError("Invalid rating")

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def home():
    return {"message": "Sentiment API Running"}