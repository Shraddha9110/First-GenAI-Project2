import pytest
import os
import pandas as pd
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_ui_data_dependency():
    """Verify that the UI can access the metadata for the location dropdown."""
    metadata_path = 'vector_store/metadata.pkl'
    assert os.path.exists(metadata_path), "Vector store metadata missing! Run Phase 2 first."
    
    df = pd.read_pickle(metadata_path)
    assert not df.empty
    assert 'location' in df.columns
    unique_locations = df['location'].unique()
    assert len(unique_locations) > 0
    assert "Banashankari" in unique_locations

def test_recommendation_engine_initialization_for_ui():
    """Verify the UI can successfully load the RecommendationEngine."""
    from src.llm.recommender import RecommendationEngine
    try:
        engine = RecommendationEngine()
        assert engine is not None
        assert engine.searcher is not None
        assert engine.llm is not None
    except Exception as e:
        pytest.fail(f"RecommendationEngine failed to initialize for UI: {e}")

def test_ui_environment_variables():
    """Check if the UI has access to the Groq API key."""
    from dotenv import load_dotenv
    load_dotenv()
    assert os.getenv("GROQ_API_KEY") is not None, "GROQ_API_KEY missing from .env"
    assert os.getenv("GROQ_API_KEY").startswith("gsk_"), "GROQ_API_KEY format is invalid"

if __name__ == "__main__":
    # Run the tests manually if called as a script
    print("Verifying Phase 5 UI Integrations...")
    test_ui_data_dependency()
    print("✅ UI Data Dependency: OK")
    test_recommendation_engine_initialization_for_ui()
    print("✅ Recommendation Engine Init: OK")
    test_ui_environment_variables()
    print("✅ UI Environment Setup: OK")
    print("\nAll Phase 5 checks passed!")
