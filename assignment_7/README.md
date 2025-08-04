# Assignment 7: Text-to-Speech Web App with Streamlit

This project demonstrates how to build a web-based Text-to-Speech (TTS) application using Streamlit, Hugging Face Transformers, and BeautifulSoup. The app allows users to enter a URL of a web page, extracts the text content, and converts it into speech.

## Objective

- Create a web interface with Streamlit to get a URL from the user.
- Fetch and parse the HTML of the web page to extract readable text.
- Use a pre-trained TTS model from Hugging Face to synthesize audio from the extracted text.
- Play the generated audio directly in the web app and provide a download link.

## Environment Setup

### 1. Python Virtual Environment

It is recommended to use a Python virtual environment to manage dependencies.

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
source venv/bin/activate
```

### 2. Install Dependencies

Install the necessary Python libraries using pip:

```bash
pip install streamlit requests beautifulsoup4 transformers torch soundfile
```

## How to Run

1.  **Activate the virtual environment** (if not already active):
    ```bash
    source venv/bin/activate
    ```
2.  **Run the Streamlit application**:
    ```bash
    streamlit run assignment_7.py
    ```

This will start the Streamlit server and open the application in your web browser.

## What the App Does

1.  **Web Interface**: It provides a simple web interface using Streamlit where you can enter the URL of a web page.
2.  **Fetches and Parses Content**: When you click the "Read Aloud" button, the app fetches the content of the URL, and uses BeautifulSoup to parse the HTML and extract text from all paragraph (`<p>`) tags.
3.  **Loads a TTS Model**: It loads the `facebook/mms-tts-vie` model and its tokenizer from the Hugging Face Hub. The model is cached to avoid reloading on every run.
4.  **Generates Audio**: The extracted text is fed to the model to generate an audio waveform.
5.  **Plays and Downloads Audio**: The app displays an audio player to listen to the generated speech and provides a button to download the audio as a `.wav` file.

## Project Structure

```
assignment_7/
├── assignment_7.py   # The main Streamlit application script
└── README.md         # This file
```
