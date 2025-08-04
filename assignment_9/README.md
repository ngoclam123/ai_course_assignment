# AI Course Assignment 9 - Conversational Laptop Recommendation Bot

This project demonstrates a conversational laptop recommendation bot using Retrieval-Augmented Generation (RAG). It leverages Azure OpenAI for embeddings and language model capabilities, ChromaDB for efficient vector search, and Streamlit for the user interface. The bot can understand user needs in natural language, retrieve relevant laptop specifications, and provide context-aware recommendations by remembering the last 10 turns of the conversation.

## Features

1.  **Natural Language Queries**: Users can describe their laptop needs in plain English.
2.  **Semantic Search**: Utilizes Azure OpenAI's text embedding models to find laptops based on the meaning of the user's query, not just keywords.
3.  **Vector-Based Retrieval**: Employs ChromaDB as a vector store to quickly find the most relevant laptops from a generated dataset.
4.  **Conversational AI**: Uses an Azure OpenAI Large Language Model (LLM) to generate helpful, human-like responses.
5.  **Context-Aware Recommendations**: The bot maintains a short-term memory of the chat history (last 10 messages) to understand follow-up questions and provide more relevant answers.
6.  **Interactive UI**: A clean and simple web interface built with Streamlit.

## Environment Setup

### 1. Environment Variables

Before running the application, you need to set up your Azure OpenAI credentials. Create a `.env` file in the project's parent directory (`ai_course_assignment/`).

**File path:** `ai_course_assignment/.env`

```
AZURE_OPENAI_ENDPOINT="your_azure_openai_endpoint"
OPENAI_EMBEDDING_TEST_KEY="your_embedding_api_key"
AZURE_OPENAI_API_KEY="your_llm_api_key"
AZURE_OPENAI_EMBED_MODEL="your_embedding_deployment_name"
AZURE_OPENAI_LLM_MODEL="your_llm_deployment_name"
```

Replace the placeholder values with your actual Azure OpenAI endpoint, API keys, and deployment names.

### 2. Python Virtual Environment & Dependencies

It is recommended to use a Python virtual environment.

1.  **Create a virtual environment:**
    ```bash
    python3 -m venv venv
    ```

2.  **Activate the virtual environment:**
    ```bash
    source venv/bin/activate
    ```

3.  **Install the required packages:**
    You can create a `requirements.txt` file with the following content and install it.
    **Installation command:**
    ```bash
    pip install -r envPkg.txt
    ```

## Execution Guide

Follow these steps to run the recommendation bot.

1.  **Activate Virtual Environment** (if not already active):
    ```bash
    source venv/bin/activate
    ```

2.  **Run the Streamlit Application**:
    Start the Streamlit web server. You can run either `assignment_9.py` or `app.py`.
    ```bash
    streamlit run assignment_9.py
    ```

3.  **Interact with the Bot**:
    - The application will open in your default web browser.
    - Type your request into the input box (e.g., "I need a powerful laptop for video editing").
    - The bot will provide recommendations and you can ask follow-up questions.

## Project Structure

```
assignment_9/
├── README.md           # This file
├── assignment_9.py     # Main Streamlit application with conversational memory
├── laptops.json        # Generated laptop data (after running the script)
.env                # (Must be created by user in parent directory)
```