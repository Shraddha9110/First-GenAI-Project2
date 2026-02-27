import pytest
from unittest.mock import MagicMock, patch
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.llm.groq_client import GroqService

def test_groq_prompt_structure():
    # We test that the service correctly handles missing API keys gracefully
    # and has the correct logic structure
    service = GroqService(api_key="mock_key")
    
    # Mock the Groq client and its response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Mocked recommendation"))]
    service.client.chat.completions.create = MagicMock(return_value=mock_response)
    
    result = service.generate_recommendation("Indian food", "Context about Jalsa")
    
    assert result == "Mocked recommendation"
    service.client.chat.completions.create.assert_called_once()

def test_groq_missing_api_key():
    # Should handle missing key by returning an error message
    service = GroqService(api_key=None)
    # Ensure it doesn't crash but returns the expected error string
    with patch.dict(os.environ, {}, clear=True):
        service.api_key = None
        service.client = None
        result = service.generate_recommendation("test", "test")
        assert "GROQ_API_KEY not found" in result

def test_recommender_integration_logic():
    from src.llm.recommender import RecommendationEngine
    import pandas as pd
    
    engine = RecommendationEngine()
    
    # Mock searcher to return specific data
    mock_df = pd.DataFrame([{
        'document_string': 'Test Restaurant is great.',
        'location': 'Test Place',
        'approx_cost_two': 500
    }])
    engine.searcher.search = MagicMock(return_value=mock_df)
    
    # Mock LLM to avoid actual API call
    engine.llm.generate_recommendation = MagicMock(return_value="LLM response")
    
    response = engine.get_recommendations("query", "location", 1000)
    
    assert response == "LLM response"
    engine.searcher.search.assert_called_with("query", top_k=5, location="location", max_price=1000)
    engine.llm.generate_recommendation.assert_called()
