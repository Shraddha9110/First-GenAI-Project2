import os
import sys
from dotenv import load_dotenv

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.llm.recommender import RecommendationEngine

def test_live_rag_flow():
    load_dotenv()
    
    if not os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API_KEY") == "your_groq_api_key_here":
        print("\n⚠️  SKIPPED: Please add your real GROQ_API_KEY to the .env file first.")
        return

    print("\n🚀 Starting Live Integration Test...")
    engine = RecommendationEngine()
    
    query = "authentic North Indian food"
    location = None
    max_price = None
    
    print(f"Query: {query}")
    print(f"Filters: Location={location}, Max Price={max_price}")
    
    print("\nStep 1: Retrieving data from FAISS...")
    print("Step 2: Sending context to Groq (Llama 3)...")
    
    response = engine.get_recommendations(query, location=location, max_price=max_price)
    
    print("\n--- 🍽️  LLM RECOMMENDATION ---")
    print(response)
    print("\n--- End of Test ---")

if __name__ == "__main__":
    test_live_rag_flow()
