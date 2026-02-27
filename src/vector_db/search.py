from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd
import numpy as np
import os
import pickle

class RestaurantSearch:
    def __init__(self, vector_store_path='vector_store'):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = faiss.read_index(os.path.join(vector_store_path, 'restaurant_index.faiss'))
        self.metadata = pd.read_pickle(os.path.join(vector_store_path, 'metadata.pkl'))

    def search(self, query, top_k=5, location=None, max_price=None, min_rating=0.0):
        # 1. Start with full metadata
        filtered_df = self.metadata.copy()
        
        # 2. Apply hard filters first (Location, Price, Rating)
        # This ensures we only look at restaurants that actually meet the user's constraints
        if location and location != "Any":
            filtered_df = filtered_df[filtered_df['location'].str.contains(location, case=False, na=False)]
        
        if max_price:
            filtered_df = filtered_df[filtered_df['approx_cost_two'] <= max_price]
            
        if min_rating:
            filtered_df = filtered_df[filtered_df['rate_float'] >= min_rating]
            
        if filtered_df.empty:
            return filtered_df # Return empty if no restaurant matches hard constraints
            
        # 3. Rank the remaining restaurants by similarity to the query
        query_vector = self.model.encode([query]).astype('float32')
        
        # If the filtered set is small, we calculate similarity manually (very fast for < 50k rows)
        # If no filters were applied (full dataset), we use the FAISS index for speed
        if len(filtered_df) < len(self.metadata):
            # Fallback: Get all embeddings for the filtered subset and calculate distance
            # For simplicity and speed in this size, we can still use FAISS for the full set 
            # but we'll take a much larger pool (5000) to ensure we get results.
            # However, the most robust way is to just use the FAISS index to find global top 5000
            # and then intersect.
            search_k = min(5000, len(self.metadata))
            distances, indices = self.index.search(query_vector, search_k)
            
            # Get the top semantic matches
            top_semantic_indices = indices[0]
            
            # Intersection: Keep only those that passed our hard filters
            results = filtered_df.loc[filtered_df.index.isin(top_semantic_indices)].copy()
            
            # If intersection is empty, it means the user's specific craving isn't in the top 5000 
            # global matches for that query, but we still want to show the BEST of the filtered set.
            if results.empty:
                # In this case, just return the top restaurants by rating in that location
                return filtered_df.sort_values(by='rate_float', ascending=False).head(top_k)
            
            return results.head(top_k)
        else:
            # Standard fast FAISS search for no filters
            distances, indices = self.index.search(query_vector, top_k)
            return self.metadata.iloc[indices[0]].copy()

def test_search():
    if not os.path.exists('vector_store/restaurant_index.faiss'):
        print("Index not found. Running ingestion first...")
        from ingest import initialize_vector_db
        initialize_vector_db()
        
    searcher = RestaurantSearch()
    
    # Test queries
    print("\n--- Test 1: Spicy North Indian food in Banashankari ---")
    results = searcher.search("Spicy North Indian food", location="Banashankari", max_price=1000)
    print(results[['name', 'location', 'rate_float', 'approx_cost_two']])
    
    print("\n--- Test 2: Best rooftop pizza ---")
    results = searcher.search("Best rooftop pizza", top_k=3)
    print(results[['name', 'location', 'rate_float', 'approx_cost_two']])

if __name__ == "__main__":
    test_search()
