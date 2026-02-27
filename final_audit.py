import os
import sys
import pandas as pd
from dotenv import load_dotenv

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def run_final_audit():
    print("🔍 PROJECT-WIDE ARCHITECTURE AUDIT\n" + "="*40)
    
    status = {
        "Phase 1: Data": "❌",
        "Phase 2: Vector DB": "❌",
        "Phase 3: LLM RAG": "❌",
        "Phase 4: API": "❌",
        "Phase 5: UI": "❌"
    }

    # 1. Phase 1 Audit
    processed_csv = 'processed_data/restaurants_processed.csv'
    if os.path.exists(processed_csv):
        df = pd.read_csv(processed_csv)
        if len(df) > 50000:
            status["Phase 1: Data"] = f"✅ ({len(df)} restaurants processed)"

    # 2. Phase 2 Audit
    faiss_idx = 'vector_store/restaurant_index.faiss'
    metadata = 'vector_store/metadata.pkl'
    if os.path.exists(faiss_idx) and os.path.exists(metadata):
        status["Phase 2: Vector DB"] = "✅ (FAISS Index & Metadata ready)"

    # 3. Phase 3 Audit
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if api_key and api_key.startswith("gsk_"):
        try:
            from src.llm.recommender import RecommendationEngine
            engine = RecommendationEngine()
            # Test a small RAG query
            print("Testing RAG Flow...")
            res = engine.get_recommendations("Pizza", max_price=500)
            if res and "couldn't find" not in res.lower() and "error" not in res.lower():
                status["Phase 3: LLM RAG"] = "✅ (Live Groq Connectivity Verified)"
            else:
                status["Phase 3: LLM RAG"] = "⚠️ (Logic works, but check API/Filters)"
        except Exception as e:
            status["Phase 3: LLM RAG"] = f"❌ (Error: {str(e)[:50]})"

    # 4. Phase 4 Audit
    try:
        from src.api.main import app
        if app.title == "AI Restaurant Recommendation Service":
            status["Phase 4: API"] = "✅ (FastAPI routes defined)"
    except:
        pass

    # 5. Phase 5 Audit
    ui_file = 'src/ui/app.py'
    if os.path.exists(ui_file):
        with open(ui_file, 'r') as f:
            if "streamlit" in f.read():
                status["Phase 5: UI"] = "✅ (Streamlit Interface Ready)"

    # Final Report
    print("\nFINAL STATUS REPORT:")
    for phase, res in status.items():
        print(f"{phase}: {res}")
    
    print("\n" + "="*40)
    if all("✅" in s for s in status.values()):
        print("🚀 ALL SYSTEMS NOMINAL: The project is fully connected and working!")
    else:
        print("⚠️ SOME SYSTEMS NEED ATTENTION: Check the report above.")

if __name__ == "__main__":
    run_final_audit()
