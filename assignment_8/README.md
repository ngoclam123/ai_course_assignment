# AI Course Assignment - Smart Product Search using OpenAI Embeddings

This project is part of the AI Application Engineer course, demonstrating the use of Azure OpenAI's embedding model to create a semantic product search engine with a Streamlit web interface.

## Features

1. **Semantic Search**
   - Uses Azure OpenAI's text-embedding-3-small model for semantic understanding
   - Finds products based on meaning, not just keywords
   - Real-time search results with similarity scores

2. **Interactive Web Interface**
   - Clean, modern UI built with Streamlit
   - Real-time search updates
   - Expandable product cards
   - Visual similarity score indicators
   - Responsive design

3. **Performance Optimizations**
   - Cached embeddings generation
   - Efficient similarity calculations
   - Optimized data handling

## Environment Setup

### 1. Environment Variables
Create a `.env` file in the root directory with the following template:
```
AZURE_OPENAI_ENDPOINT=your_openai_endpoint
OPENAI_EMBEDDING_TEST_KEY=your_openai_api_key
```

Replace `your_openai_endpoint` and `your_openai_api_key` with your actual Azure OpenAI API credentials.

### 2. Python Virtual Environment Setup (Linux)

1. Create a new virtual environment:
```bash
python -m venv venv
```

2. Activate the virtual environment:
```bash
source venv/bin/activate
```

3. Install the required packages:
```bash
pip install -r envPkg.txt
```

## Execution Guide

1. **Activate Virtual Environment** (if not already activated):
```bash
source venv/bin/activate
```

2. **Run the Application**:
```bash
streamlit run assignment_8.py
```

3. **Using the Search Engine**:
   - The application will open in your default web browser
   - Enter your search query in natural language (e.g., "warm cotton sweatshirt")
   - View real-time results with similarity scores
   - Click on product cards to expand/collapse details

## Project Structure
```
.
├── .env                # Environment variables configuration
├── envPkg.txt         # Python package requirements file
└── assignment_8.py    # Main application file with search implementation
```

## Security Note
Make sure to keep your `.env` file secure and never commit it to version control. Add it to your `.gitignore` file to prevent accidental commits.
