import os
import pandas as pd
import sys

# Add src to path if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_preprocessing_output():
    processed_file = 'processed_data/restaurants_processed.csv'
    
    # Check if file exists
    if not os.path.exists(processed_file):
        print(f"❌ FAILED: {processed_file} does not exist.")
        return False
    
    # Load and check columns
    df = pd.read_csv(processed_file)
    required_cols = ['name', 'rate_float', 'approx_cost_two', 'document_string']
    
    for col in required_cols:
        if col not in df.columns:
            print(f"❌ FAILED: Missing column {col}")
            return False
            
    # Check for empty document strings
    if df['document_string'].isnull().any():
        print("❌ FAILED: Found null values in document_string")
        return False

    print(f"✅ SUCCESS: Preprocessing test passed! Found {len(df)} restaurants.")
    return True

if __name__ == "__main__":
    test_preprocessing_output()
