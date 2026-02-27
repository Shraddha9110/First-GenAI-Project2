import os
import sys
import pandas as pd

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.vector_db.search import RestaurantSearch

searcher = RestaurantSearch()
print(f"Total metadata rows: {len(searcher.metadata)}")

# Try search without filters
print("\nSearch without filters:")
res = searcher.search("North Indian", top_k=5)
print(res[['name', 'location']])

# Try search with location filter
print("\nSearch with Banashankari filter:")
res_filtered = searcher.search("North Indian", top_k=5, location="Banashankari")
print(res_filtered[['name', 'location']])
