# AI Course Assignment - Chatbot using OpenAI API

This project is part of the AI Application Engineer course, focusing on creating a chatbot using the OpenAI API.

## Environment Setup

### 1. Environment Variables
Create a `.env` file in the root directory with the following template:
```
AZURE_OPENAI_ENDPOINT=your_openai_endpoint
AZURE_OPENAI_API_KEY=your_openai_api_key
```

Replace `your_openai_endpoint` and `your_openai_api_key` with your actual OpenAI API credentials.

### 2. Python Virtual Environment Setup

1. Create a new virtual environment:
```bash
python -m venv venv
```

2. Activate the virtual environment:
```bash
source venv/bin/activate
```

3. Install the required packages from envPkg.txt:
```bash
pip install -r envPkg.txt
```

## Project Structure
```
.
├── .env                # Environment variables configuration
└── envPkg.txt         # Python package requirements file
```

Make sure to keep your `.env` file secure and never commit it to version control. Add it to your `.gitignore` file to prevent accidental commits.
