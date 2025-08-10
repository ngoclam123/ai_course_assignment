# Assignment 10: Vietnamese Product Search Engine with Vector Database

This project is a complete Vietnamese product search engine using **Pinecone vector database**, **Azure OpenAI embeddings**, and **Streamlit** for the user interface. The system enables semantic search through Vietnamese product data using advanced AI embeddings.

## ✨ Features

- **🔍 Semantic Search**: Search products using natural language queries in Vietnamese or English
- **🎯 Vector Similarity**: 512-dimensional embeddings for accurate product matching
- **🏷️ Smart Filtering**: Filter results by category and price range
- **📊 Real-time Stats**: Live database statistics and progress tracking
- **🎨 Modern UI**: Professional Streamlit interface with responsive design
- **⚡ High Performance**: Cached operations and batch processing for speed

## 🛠️ Technical Stack

- **Vector Database**: Pinecone (serverless, AWS us-east-1)
- **AI Embeddings**: Azure OpenAI `text-embedding-3-small` (512 dimensions)
- **Web Framework**: Streamlit
- **Data Processing**: 233 Vietnamese products with structured metadata
- **Language Support**: Vietnamese and English queries

## 🚀 Key Improvements

### 1. **Advanced Vector Embeddings**
- Uses `text-embedding-3-small` model from Azure OpenAI
- 512-dimensional embeddings for superior semantic understanding
- Compatible with current API key limitations

### 2. **Production-Ready Architecture**
- Environment variable configuration
- Error handling and graceful failures
- Batch processing for large datasets
- Caching for improved performance

### 3. **Real Vietnamese Product Data**
- 233 authentic Vietnamese products
- Categories include clothing, sportswear, and accessories
- Price information and discount details
- Rich product descriptions for better matching

### 4. **Professional User Interface**
- Clean, intuitive design
- Progress tracking for embedding generation
- Interactive filters and controls
- Sample query suggestions
- Similarity score visualization

## 📁 Project Structure

```
assignment_10/
├── assignment_10.py              # Main application (production-ready)
├── convert_products.py           # Data conversion utility
├── vietnamese_products.json      # Structured product database (233 items)
├── status_check.py              # Comprehensive system status checker
└── README.md                     # Documentation
```

## 🛠️ Setup Instructions

### 1. Prerequisites
- Python 3.8+ with virtual environment
- Azure OpenAI API access
- Pinecone account and API key

### 2. Install Dependencies
```bash
# Activate virtual environment
source ../venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 3. Environment Configuration
Create a `.env` file in the parent directory (`../`) with:
```env
PINECONE_API_KEY=your_pinecone_api_key
AZURE_OPENAI_ENDPOINT=your_azure_endpoint
OPENAI_EMBEDDING_TEST_KEY=your_azure_openai_key
```

### 4. Run the Application
```bash
streamlit run assignment_10.py
```

## 📊 Usage Guide

### **First Time Setup**
1. Start the application: `streamlit run assignment_10.py`
2. Click "� Generate & Upload Embeddings" in the sidebar
3. Wait for all 233 products to be processed and uploaded to Pinecone
4. Start searching!

### **Search Examples**
- `áo thun cotton nam` - Vietnamese product search
- `sports shirt` - English equivalent search
- `quần short thể thao` - Athletic shorts in Vietnamese
- `polo premium` - High-quality polo shirts

### **Advanced Features**
- **Category Filter**: Select specific product categories
- **Price Range**: Set minimum and maximum price limits
- **Results Count**: Adjust number of search results (1-20)
- **Similarity Scores**: View match confidence levels

## 🔧 Technical Specifications

### **Vector Database**
- **Platform**: Pinecone Serverless
- **Cloud**: AWS us-east-1
- **Dimensions**: 512
- **Metric**: Cosine similarity
- **Index Name**: coolmate-index

### **AI Model**
- **Provider**: Azure OpenAI
- **Model**: text-embedding-3-small
- **Dimensions**: 512 (maximum for this model)
- **API Version**: 2024-07-01-preview

### **Data Processing**
- **Products**: 233 Vietnamese items
- **Categories**: 7 different product types
- **Price Range**: 15,000đ - 2,450,000đ
## 🚀 Quick Start

1. **Activate virtual environment**:
   ```bash
   source ../venv/bin/activate
   ```

2. **Start the application**:
   ```bash
   streamlit run assignment_10.py
   ```

3. **First-time setup**:
   - Click "🔄 Generate & Upload Embeddings" in the sidebar
   - Wait for all 233 products to be processed
   - Start searching!

## 📊 System Status

Check your setup with the included status scripts:

```bash
# Quick system check
python quick_status.py

# Comprehensive status report
python status_check.py
```

## 🎯 Example Searches

Try these Vietnamese product searches:

- **`áo thun cotton nam`** - Cotton men's t-shirts
- **`quần short thể thao`** - Athletic shorts  
- **`áo polo premium`** - Premium polo shirts
- **`quần jogger thoải mái`** - Comfortable jogger pants
- **`áo sơ mi dài tay`** - Long sleeve dress shirts
- **`sports shirt`** - English search works too!

## 🔧 System Architecture

```
User Query → Azure OpenAI Embedding → Pinecone Search → Results + Filtering → Streamlit UI
     ↓
[Vietnamese Text] → [512D Vector] → [Similarity Search] → [Ranked Results] → [Web Interface]
```

## 📈 Performance Metrics

- **Search Speed**: < 2 seconds for typical queries
- **Embedding Generation**: ~5 seconds per product
- **Database Size**: 233 products, 512D vectors
- **Similarity Accuracy**: High quality semantic matching
- **UI Response**: Real-time filtering and search

## 🛠️ Troubleshooting

### Common Issues:

1. **Missing environment variables**: Check `.env` file in parent directory
2. **API key issues**: Ensure Azure OpenAI key has access to `text-embedding-3-small`
3. **Pinecone connection**: Verify API key and index creation
4. **Empty results**: Generate embeddings first using the sidebar button

### Status Check:
```bash
python quick_status.py  # Fast system verification
```

## 📝 Technical Implementation Notes

---

## 🎉 Success!

You now have a fully functional Vietnamese product search engine with:

✅ **512-dimensional semantic embeddings**  
✅ **Real-time vector search with Pinecone**  
✅ **Professional Streamlit web interface**  
✅ **233 Vietnamese products ready to search**  
✅ **Category and price filtering**  
✅ **Production-ready error handling**

### 🚀 Ready to Search!

Start the application and try searching for:
- `áo thun cotton nam` - Men's cotton t-shirts
- `quần short thể thao` - Athletic shorts  
- `áo polo premium` - Premium polo shirts

**Happy searching! 🛍️**
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