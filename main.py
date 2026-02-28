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
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY not found")

        client = OpenAI(api_key=api_key)

        response = client.responses.create(
            model="gpt-4.1-mini",
            input=f"""
Analyze this comment and return ONLY valid JSON:

Comment: {request.comment}

Return exactly:
{{
  "sentiment": "positive | negative | neutral",
  "rating": 1-5
}}

Rules:
5 = highly positive
4 = positive
3 = neutral
2 = negative
1 = highly negative
"""
        )

        content = response.output_text.strip()

        result = json.loads(content)

        if result["sentiment"] not in ["positive", "negative", "neutral"]:
            raise ValueError("Invalid sentiment value")

        if not (1 <= int(result["rating"]) <= 5):
            raise ValueError("Invalid rating value")

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def home():
    return {"message": "Sentiment API Running"}