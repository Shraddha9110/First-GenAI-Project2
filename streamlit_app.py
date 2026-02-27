import streamlit as st
import os
import sys
import pandas as pd
from dotenv import load_dotenv
from PIL import Image

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.llm.recommender import RecommendationEngine

# Load environment variables
load_dotenv()

# Handle Groq API Key for Deployment
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

# Page Config
st.set_page_config(
    page_title="Zomato AI Guide",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for a Premium Look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp {
        background-color: #FAFAFA;
    }
    
    /* Hero Title Styling */
    .hero-title {
        font-size: 3.5rem;
        font-weight: 700;
        color: #1A1A1A;
        margin-bottom: 0.5rem;
        line-height: 1.2;
    }
    
    .hero-subtitle {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #EEE;
    }
    
    /* Recommendation Card Design */
    .recommendation-container {
        background-color: #FFFFFF;
        padding: 2.5rem;
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.05);
        border: 1px solid #F0F0F0;
        margin-top: 2rem;
    }
    
    .stButton>button {
        background: linear-gradient(90deg, #FF4B4B 0%, #FF2E2E 100%);
        color: white;
        font-weight: 600;
        border-radius: 12px;
        padding: 0.6rem 2rem;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255, 75, 75, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 75, 75, 0.4);
    }
    
    /* Input Styling */
    div[data-baseweb="input"] {
        border-radius: 12px !important;
        background-color: #F3F3F3 !important;
        border: 1px solid transparent !important;
    }
    
    div[data-baseweb="input"]:focus-within {
        border: 1px solid #FF4B4B !important;
        background-color: #FFF !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Engine
@st.cache_resource
def get_engine():
    return RecommendationEngine()

try:
    engine = get_engine()
    metadata = pd.read_pickle('vector_store/metadata.pkl')
except Exception as e:
    st.error(f"Initialization Error: {e}")
    st.stop()

# --- Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3448/3448609.png", width=60)
    st.title("Filters")
    st.markdown("Refine your taste.")
    
    st.divider()
    
    location_options = ["Any"] + sorted(metadata['location'].dropna().unique().tolist())
    place = st.selectbox("🌍 Select Location", options=location_options, index=0)
    
    max_price = st.slider("💰 Max Budget (for two)", 200, 5000, 1500, step=100)
    min_rating = st.slider("⭐ Min Rating", 0.0, 5.0, 3.5, step=0.1)
    
    st.divider()
    st.info("💡 Tip: Be specific! 'Rooftop Italian with cocktails' works better than just 'Pizza'.")

# --- Main Page ---
col1, col2 = st.columns([1.5, 1])

with col1:
    st.markdown('<div class="hero-title">Find Your Next <br><span style="color:#FF4B4B">Favorite Spot</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Smart AI-powered restaurant discovery for the foodies of Bangalore.</div>', unsafe_allow_html=True)
    
    query = st.text_input("What are you craving today?", 
                         placeholder="Describe the vibe, dish, or occasion...", 
                         label_visibility="collapsed")
    
    search_clicked = st.button("Get Recommendation")

with col2:
    # Use the local asset image
    hero_img_path = os.path.join(os.path.dirname(__file__), 'assets', 'hero_banner.png')
    if os.path.exists(hero_img_path):
        st.image(hero_img_path, use_column_width=True, caption="Experience Excellence")
    else:
        st.image("https://images.unsplash.com/photo-1514362545857-3bc16c4c7d1b?auto=format&fit=crop&q=80", use_column_width=True)

# --- Results Area ---
if search_clicked:
    if not query:
        st.warning("Please tell us what you're craving first!")
    else:
        with st.spinner("Searching the city for the best matches..."):
            loc_filter = None if place == "Any" else place
            
            recommendation = engine.get_recommendations(
                query=query,
                location=loc_filter,
                max_price=max_price,
                min_rating=min_rating
            )
        
        st.markdown('<div class="recommendation-container">', unsafe_allow_html=True)
        st.subheader("🍽️ Our Handpicked Suggestions")
        st.markdown(recommendation)
        st.markdown('</div>', unsafe_allow_html=True)

# Footer Styling
st.markdown("<br><hr><center><p style='color: #999;'>Powered by Vector Search & Groq Llama 3.1 • 2024</p></center>", unsafe_allow_html=True)
