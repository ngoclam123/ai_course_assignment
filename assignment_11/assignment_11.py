import io
from PIL import Image
from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI
import base64
import streamlit as st
import os
import json
import numpy as np
from pinecone import Pinecone, ServerlessSpec
from openai import AzureOpenAI, RateLimitError, APIError
from dotenv import load_dotenv
from scipy.spatial.distance import cosine
import csv
from datetime import datetime
import pandas as pd

# Load environment variables
load_dotenv("../.env")

# Initialize log file path
LOG_FILE_PATH = "classification_log.csv"

def initialize_log_file():
    """Initialize CSV log file with headers if it doesn't exist"""
    if not os.path.exists(LOG_FILE_PATH):
        with open(LOG_FILE_PATH, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['Timestamp', 'Filename', 'Prediction', 'Confidence'])

def log_classification(filename, prediction, confidence):
    """Log classification result to CSV file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Ensure log file exists
    initialize_log_file()
    
    # Append new classification result
    with open(LOG_FILE_PATH, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, filename, prediction, f"{confidence:.1f}%"])

def display_log_history():
    """Display classification history from log file"""
    if os.path.exists(LOG_FILE_PATH):
        try:
            df = pd.read_csv(LOG_FILE_PATH)
            if not df.empty:
                st.subheader("📊 Classification History")
                st.dataframe(df, use_container_width=True)
                
                # Add download button for the log file
                with open(LOG_FILE_PATH, 'rb') as file:
                    st.download_button(
                        label="📥 Download Complete Log (CSV)",
                        data=file.read(),
                        file_name=LOG_FILE_PATH,
                        mime='text/csv'
                    )
        except Exception as e:
            st.error(f"Error reading log file: {str(e)}")
    else:
        st.info("No classification history available yet. Upload and classify an image to start logging.")

# --- Azure OpenAI Config ---
st.set_page_config(
    page_title="Satellite Cloud Detection with Azure OpenAI", 
    page_icon="🛰️",
    layout="wide"
)

st.title("🛰️ Satellite Image Cloud Detection")
st.markdown("""
Upload a satellite image (.jpg, .png) and get instant cloud/clear detection using Azure OpenAI.
All classifications are automatically logged to a CSV file for analysis.
""")

# Initialize log file
initialize_log_file()


# User inputs for secrets/config (can use os.environ for deployment)
azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
azure_api_key = os.environ.get("AZURE_OPENAI_API_KEY")
azure_deployment = os.environ.get("AZURE_OPENAI_LLM_MODEL")
# --- Model Setup ---
llm = AzureChatOpenAI(
    azure_endpoint=azure_endpoint,
    azure_deployment=azure_deployment,
    api_key=azure_api_key,
    api_version="2024-02-15-preview",  # Adjust if necessary
)


class WeatherResponse(BaseModel):

    accuracy: float = Field(description="The accuracy of the result")
    result: str = Field(description="The result of the classification")


llm_with_structured_output = llm.with_structured_output(WeatherResponse)

# --- File Upload ---
st.subheader("📤 Upload Satellite Image")
uploaded_file = st.file_uploader(
    "Choose a satellite image", 
    type=["jpg", "jpeg", "png"],
    help="Upload a satellite image to classify as Clear or Cloudy"
)
if uploaded_file:
    # Display the uploaded image
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📸 Uploaded Image")
        st.image(uploaded_file, caption=f"Filename: {uploaded_file.name}", use_column_width=True)
        
        # Show file details
        st.info(f"**File:** {uploaded_file.name}")
        st.info(f"**Size:** {uploaded_file.size / 1024:.1f} KB")
    
    with col2:
        st.subheader("🔍 Classification")
        
        # Process the image
        image_bytes = uploaded_file.read()
        image_data = base64.b64encode(image_bytes).decode("utf-8")
        image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        # Prompt
        message = [
            {
                "role": "system",
                "content": """Based on the satellite image provided, classify the scene as either:
                        'Clear' (no clouds) or 'Cloudy' (with clouds).
                        Respond with only one word: either 'Clear' or 'Cloudy' and Accuracy. Do not
                        provide explanations.""",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Classify the scene as either: 'Clear' (no clouds) or 'Cloudy' (with clouds) and Accuracy.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data}",
                        },
                    },
                ],
            },
        ]

        # Classify button
        if st.button("🚀 Classify Image", type="primary"):
            with st.spinner("🔄 Analyzing satellite image..."):
                try:
                    response = llm_with_structured_output.invoke(message)
                    
                    # Display results
                    prediction = response.result
                    confidence = response.accuracy
                    
                    # Color-coded results
                    if prediction.lower() == "clear":
                        st.success(f"☀️ **Prediction:** {prediction}")
                    else:
                        st.info(f"☁️ **Prediction:** {prediction}")
                    
                    st.metric("🎯 Confidence", f"{confidence:.1f}%", delta=None)
                    
                    # Log the classification result
                    log_classification(uploaded_file.name, prediction, confidence)
                    st.success("✅ Result logged to CSV file!")
                    
                except Exception as e:
                    st.error(f"❌ Prediction failed: {e}")

# Separator
st.markdown("---")

# Display log history
display_log_history()

# Footer
st.markdown("---")
st.caption("🛰️ Built with Azure OpenAI, Streamlit, and LangChain | 📊 All results automatically logged to CSV")
