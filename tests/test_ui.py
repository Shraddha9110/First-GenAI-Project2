import pytest
import os

def test_ui_file_exists():
    assert os.path.exists('src/ui/app.py')

def test_ui_components_defined():
    with open('src/ui/app.py', 'r') as f:
        content = f.read()
        assert "st.sidebar.selectbox" in content
        assert "st.sidebar.slider" in content
        assert "engine.get_recommendations" in content
        assert "st.markdown" in content

def test_ui_import_logic():
    # Check if it correctly imports from the parent directory structure
    with open('src/ui/app.py', 'r') as f:
        content = f.read()
        assert "sys.path.append" in content
        assert "src.llm.recommender" in content
