import pandas as pd
from datasets import load_dataset
import os
from tqdm import tqdm

def load_and_preprocess_data(dataset_name="ManikaSaini/zomato-restaurant-recommendation"):
    print(f"Loading dataset from Hugging Face: {dataset_name}...")
    try:
        # Load dataset
        dataset = load_dataset(dataset_name)
        df = pd.DataFrame(dataset['train'])
        
        print("Data loaded. Starting preprocessing...")
        
        # 1. Basic Cleaning
        # Fill missing values
        df['cuisines'] = df['cuisines'].fillna('Not Specified')
        df['rest_type'] = df['rest_type'].fillna('Not Specified')
        df['dish_liked'] = df['dish_liked'].fillna('')
        df['location'] = df['location'].fillna('Unknown')
        
        # Normalize Ratings
        def clean_rate(rate):
            if isinstance(rate, str):
                if '/' in rate:
                    rate = rate.split('/')[0].strip()
                if rate == 'NEW' or rate == '-':
                    return 0.0
                try:
                    return float(rate)
                except:
                    return 0.0
            return float(rate) if rate else 0.0

        df['rate_float'] = df['rate'].apply(clean_rate)
        
        # Normalize Cost
        def clean_cost(cost):
            if isinstance(cost, str):
                cost = cost.replace(',', '').strip()
                try:
                    return int(cost)
                except:
                    return 0
            return int(cost) if cost else 0

        df['approx_cost_two'] = df['approx_cost(for two people)'].apply(clean_cost)
        
        # 2. Feature Engineering: Create Document String
        def create_doc_string(row):
            name = row.get('name', 'Unknown')
            cuisines = row.get('cuisines', 'Various')
            location = row.get('location', 'Unknown')
            rate = row.get('rate_float', 0.0)
            cost = row.get('approx_cost_two', 0)
            rest_type = row.get('rest_type', 'Restaurant')
            liked = row.get('dish_liked', '')
            
            doc = f"{name} is a {rest_type} specializing in {cuisines}, located in {location}. "
            doc += f"It has a rating of {rate}/5.0. "
            if cost > 0:
                doc += f"The approximate cost for two people is ₹{cost}. "
            if liked:
                doc += f"Customers particularly liked: {liked}."
            return doc

        print("Generating document strings for embeddings...")
        df['document_string'] = df.apply(create_doc_string, axis=1)
        
        # 3. Create output directory if not exists
        os.makedirs('processed_data', exist_ok=True)
        
        # Save as CSV
        output_path = 'processed_data/restaurants_processed.csv'
        df.to_csv(output_path, index=False)
        print(f"Preprocessing complete! Data saved to {output_path}")
        return df

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    load_and_preprocess_data()
