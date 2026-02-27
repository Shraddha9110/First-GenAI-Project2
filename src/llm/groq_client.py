import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class GroqService:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            # We'll allow initialization without key for structure, 
            # but methods will fail if key is missing when called.
            pass
        self.client = Groq(api_key=self.api_key) if self.api_key else None

    def generate_recommendation(self, user_query, restaurants_context):
        if not self.client:
            return "Error: GROQ_API_KEY not found in environment. Please set it in a .env file."

        system_prompt = """
        You are an expert local food guide for Zomato. Your goal is to provide clear, helpful, 
        and conversational restaurant recommendations based ONLY on the context provided.
        
        Instructions:
        1. Read the user's preferences and the list of matching restaurants.
        2. Select the best 1-3 options from the context.
        3. Explain WHY you chose each one based on their price, rating, or cuisine.
        4. Use a friendly, professional tone.
        5. Use Markdown for formatting (bold names, bullet points).
        6. If no restaurants in the context strictly match the criteria, suggest the closest ones and explain.
        7. DO NOT hallucinate restaurants not provided in the context.
        """

        prompt = f"""
        User Query: {user_query}

        Available Restaurants Data:
        {restaurants_context}

        Please provide your recommendations:
        """

        try:
            completion = self.client.chat.completions.create(
                model="llama-3.1-8b-instant", # Current supported Groq model
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1024
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"An error occurred while calling Groq: {str(e)}"
