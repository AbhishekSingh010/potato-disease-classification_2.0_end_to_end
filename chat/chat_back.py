import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import streamlit as st

# Load environment variables from .env file
load_dotenv()

def content_creation(user_query,disease):
    st.write(disease)
    # Create the tweet prompt template
    tweet_prompt = PromptTemplate.from_template(f"**Act as an potato disease and agriculture expert, and  how can I help you with the {disease}  ?** (e.g., cause, symptoms, treatment)")

    # Get the Google API key from environment variables
    google_api_key = os.getenv('GOOGLE_API_KEY')
    topic=f"{user_query} for potato plant disease the disease is {disease} give response"
    # Check if the Google API key is available
    if google_api_key is None:
        raise ValueError("Google API key not found in environment variables")

    # Initialize ChatGoogleGenerativeAI with Gemini Pro model and Google API key
    llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=google_api_key)

    # Initialize LLMChain with the ChatGoogleGenerativeAI instance and tweet prompt
    tweet_chain = LLMChain(llm=llm, prompt=tweet_prompt, verbose=True)

    # Run the LLMChain to generate content
    try:
        response = tweet_chain.run(topic=topic)
        return response
    except Exception as e:
        return f"Error: {str(e)}"
