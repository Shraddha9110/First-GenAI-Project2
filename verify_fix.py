from src.vector_db.search import RestaurantSearch
import os
import sys

# Ensure we are in root
sys.path.append(os.getcwd())

searcher = RestaurantSearch()
# Testing with a very common query and location
print("Testing search for 'Pizza' in 'Banashankari' with max_price 1000...")
res = searcher.search("Pizza", location="Banashankari", max_price=1000)
print(f"Results found: {len(res)}")
if not res.empty:
    print(res[['name', 'location', 'rate_float', 'approx_cost_two']].to_string())
else:
    print("Zero results found.")

print("\nTesting search for 'Biryani' without location filter...")
res2 = searcher.search("Biryani", top_k=3)
print(f"Results found: {len(res2)}")
print(res2[['name', 'location', 'rate_float', 'approx_cost_two']].to_string())
