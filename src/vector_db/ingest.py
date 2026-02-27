from sentence_transformers import SentenceTransformer
import pandas as pd
import faiss
import numpy as np
import os
import pickle

def initialize_vector_db(csv_path='processed_data/restaurants_processed.csv', 
                         model_name='all-MiniLM-L6-v2',
                         output_dir='vector_store'):
    
    print(f"Loading processed data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading embedding model: {model_name}...")
    model = SentenceTransformer(model_name)
    
    print(f"Generating embeddings for {len(df)} restaurants...")
    # This might take a few minutes
    sentences = df['document_string'].tolist()
    embeddings = model.encode(sentences, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')
    
    # 3. Create FAISS Index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    print(f"Saving FAISS index and metadata to {output_dir}...")
    faiss.write_index(index, os.path.join(output_dir, 'restaurant_index.faiss'))
    
    # Save the dataframe metadata (needed for retrieval)
    # We only save necessary columns to save space
    meta_cols = ['name', 'location', 'rate_float', 'approx_cost_two', 'cuisines', 'document_string']
    df[meta_cols].to_pickle(os.path.join(output_dir, 'metadata.pkl'))
    
    print("Vector database initialization complete!")

if __name__ == "__main__":
    if os.path.exists('processed_data/restaurants_processed.csv'):
        initialize_vector_db()
    else:
        print("Error: processed_data/restaurants_processed.csv not found. Run Phase 1 first.")
