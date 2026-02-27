import pytest
import pandas as pd
import os

@pytest.fixture
def processed_df():
    processed_file = 'processed_data/restaurants_processed.csv'
    if not os.path.exists(processed_file):
        pytest.fail(f"Processed file {processed_file} not found. Run preprocessing first.")
    return pd.read_csv(processed_file)

def test_file_exists():
    assert os.path.exists('processed_data/restaurants_processed.csv')

def test_required_columns(processed_df):
    required_columns = [
        'name', 'location', 'rest_type', 'cuisines', 
        'rate_float', 'approx_cost_two', 'document_string'
    ]
    for col in required_columns:
        assert col in processed_df.columns, f"Missing column: {col}"

def test_rating_range(processed_df):
    # Ensure ratings are between 0 and 5
    assert processed_df['rate_float'].min() >= 0.0
    assert processed_df['rate_float'].max() <= 5.0

def test_cost_non_negative(processed_df):
    # Ensure costs are not negative
    assert (processed_df['approx_cost_two'] >= 0).all()

def test_document_string_content(processed_df):
    # Check a few samples to ensure document strings are actually descriptive
    sample = processed_df.iloc[0]['document_string']
    assert isinstance(sample, str)
    assert len(sample) > 50
    assert "rating" in sample.lower()
    assert "located in" in sample.lower()

def test_no_null_critical_fields(processed_df):
    # Name and location shouldn't be null in processed data
    assert processed_df['name'].isnull().sum() == 0
    assert processed_df['location'].isnull().sum() == 0
    assert processed_df['document_string'].isnull().sum() == 0
