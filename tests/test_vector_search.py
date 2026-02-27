import pytest
import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.vector_db.search import RestaurantSearch

@pytest.fixture(scope="module")
def searcher():
    if not os.path.exists('vector_store/restaurant_index.faiss'):
        pytest.fail("FAISS index not found. Complete Phase 2 ingestion first.")
    return RestaurantSearch()

def test_vector_search_returns_results(searcher):
    results = searcher.search("Italian pasta", top_k=5)
    assert len(results) > 0
    assert len(results) <= 5
    assert 'name' in results.columns
    assert 'document_string' in results.columns

def test_location_filtering(searcher):
    # Search for something in a specific location
    results = searcher.search("North Indian", location="Banashankari", top_k=5)
    for _, row in results.iterrows():
        assert "Banashankari" in row['location']

def test_price_filtering(searcher):
    # Search with a maximum price
    max_p = 500
    results = searcher.search("Cafe", max_price=max_p, top_k=5)
    for _, row in results.iterrows():
        assert row['approx_cost_two'] <= max_p

def test_semantic_relevance(searcher):
    # Search for "pizza" and check if it's in the document strings
    results = searcher.search("pizza", top_k=3)
    found_relevant = False
    for _, row in results.iterrows():
        if "pizza" in row['document_string'].lower() or "italian" in row['document_string'].lower():
            found_relevant = True
            break
    assert found_relevant, "Semantic search did not find relevant results for 'pizza'"
