import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Literal
from openai import OpenAI

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Model
class CommentRequest(BaseModel):
    comment: str

# Response Model
class SentimentResponse(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    rating: int

@app.post("/comment", response_model=SentimentResponse)
async def analyze_comment(request: CommentRequest):

    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")

        client = OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a sentiment analysis API. Return only valid JSON."
                },
                {
                    "role": "user",
                    "content": f"""
Analyze the comment and return JSON:

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

Comment:
{request.comment}
"""
                }
            ],
            temperature=0
        )

        content = response.choices[0].message.content.strip()

        import json
        result = json.loads(content)

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def home():
    return {"message": "Sentiment API Running"}