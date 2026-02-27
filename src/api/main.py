from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.llm.recommender import RecommendationEngine

app = FastAPI(title="AI Restaurant Recommendation Service")

# Initialize engine
engine = RecommendationEngine()

class UserPreferences(BaseModel):
    query: str
    location: Optional[str] = None
    max_price: Optional[int] = None

class RecommendationResponse(BaseModel):
    recommendation: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the AI Restaurant Recommendation API"}

@app.post("/recommend", response_model=RecommendationResponse)
def get_recommendation(prefs: UserPreferences):
    try:
        response = engine.get_recommendations(
            query=prefs.query,
            location=prefs.location,
            max_price=prefs.max_price
        )
        return RecommendationResponse(recommendation=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
