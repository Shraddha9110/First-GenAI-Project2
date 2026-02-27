# 🚀 Deploying Zomato AI Guide to Streamlit Cloud

Follow these steps to deploy your AI Restaurant Recommendation service:

## 1. Prepare your GitHub Repository
Ensure all files are committed and pushed to your GitHub repository, including:
- `src/` (core logic and UI)
- `vector_store/` (FAISS index and metadata)
- `requirements.txt` (dependencies)
- `.streamlit/config.toml` (UI settings)

## 2. Connect to Streamlit Community Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io).
2. Click **"New app"**.
3. Select your repository, branch, and set the main file path to:
   `src/ui/app.py`

## 3. Set Up Secrets (Crucial)
In the Streamlit deployment dashboard, go to **Settings > Secrets** and paste your Groq API key:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

## 4. Helpful Deployment Info
- **Main File**: `src/ui/app.py`
- **Python Version**: 3.9+ recommended.
- **Large Files**: The vector store files are ~80MB, which is well within GitHub's limit. No Git LFS is strictly required, but recommended for larger datasets.

---
*Powered by Vector Search & Groq Llama 3.1*
