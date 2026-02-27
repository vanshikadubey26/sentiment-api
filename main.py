from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import os
import json

# Load environment variables
load_dotenv()

# Check API key exists
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError(" OPENAI_API_KEY missing! Add to .env file.")

client = OpenAI(api_key=api_key)

app = FastAPI(title="Sentiment API")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CommentRequest(BaseModel):
    comment: str

json_schema = {
    "type": "object",
    "properties": {
        "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
        "rating": {"type": "integer", "minimum": 1, "maximum": 5}
    },
    "required": ["sentiment", "rating"],
    "additionalProperties": False
}

@app.post("/comment")
async def analyze_comment(request: CommentRequest):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"Analyze: {request.comment}"}],
            response_format={"type": "json_schema", "json_schema": json_schema}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"status": "ready", "endpoint": "/comment"}
