from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import os
import json

# Load your API key FIRST
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Create the FastAPI app
app = FastAPI()

# Request model
class CommentRequest(BaseModel):
    comment: str

# JSON Schema (forces perfect structure)
json_schema = {
    "type": "object",
    "properties": {
        "sentiment": {
            "type": "string",
            "enum": ["positive", "negative", "neutral"]
        },
        "rating": {
            "type": "integer",
            "minimum": 1,
            "maximum": 5
        }
    },
    "required": ["sentiment", "rating"],
    "additionalProperties": False
}

@app.post("/comment")
async def analyze_comment(request: CommentRequest):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": f"Analyze sentiment: {request.comment}"}
            ],
            response_format={
                "type": "json_schema",
                "json_schema": json_schema
            }
        )
        result = json.loads(response.choices[0].message.content)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"message": "Sentiment API ready!"}
