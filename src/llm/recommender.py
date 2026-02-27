import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.vector_db.search import RestaurantSearch
from src.llm.groq_client import GroqService

class RecommendationEngine:
    def __init__(self):
        self.searcher = RestaurantSearch()
        self.llm = GroqService()

    def get_recommendations(self, query, location=None, max_price=None, min_rating=0.0):
        # 1. Retrieve relevant restaurants from Vector DB (FAISS)
        retrieved_results = self.searcher.search(query, top_k=5, location=location, max_price=max_price, min_rating=min_rating)
        
        if retrieved_results.empty:
            return "I couldn't find any restaurants matching your specific criteria. Try adjusting your filters!"

        # 2. Format context for LLM
        context = ""
        for i, (_, row) in enumerate(retrieved_results.iterrows()):
            context += f"{i+1}. {row['document_string']}\n"

        # 3. Get synthesis from Groq LLM
        recommendation = self.llm.generate_recommendation(query, context)
        
        return recommendation

if __name__ == "__main__":
    # Test run
    engine = RecommendationEngine()
    print("Testing Recommendation Engine (RAG flow)...")
    
    # Example Query
    query = "I want some authentic spicy North Indian food"
    response = engine.get_recommendations(query, location="Banashankari", max_price=1000)
    print("\n--- Recommendation ---")
    print(response)
