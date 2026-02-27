import pytest
import os
import sys
from dotenv import load_dotenv

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.llm.recommender import RecommendationEngine

@pytest.fixture(scope="module")
def engine():
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY not found in environment")
    return RecommendationEngine()

def test_full_rag_pipeline_execution(engine):
    """Verifies that the full pipeline from search to LLM response works."""
    query = "authentic North Indian food"
    # Testing without filters to ensure we get results for the LLM to process
    response = engine.get_recommendations(query)
    
    assert isinstance(response, str)
    assert len(response) > 50
    # Check if it looks like a recommendation
    assert "restaurant" in response.lower() or "kitchen" in response.lower() or "food" in response.lower()
    assert "error" not in response.lower()

def test_recommender_handles_no_results(engine):
    """Verifies the engine's response when no restaurants match criteria."""
    # Using a location that definitely won't exist in the Bangalore Zomato dataset
    query = "sushi"
    location = "Pluto Planet" 
    
    response = engine.get_recommendations(query, location=location)
    assert "couldn't find any restaurants" in response.lower()
