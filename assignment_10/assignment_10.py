import streamlit as st
import os
import json
import numpy as np
from pinecone import Pinecone, ServerlessSpec
from openai import AzureOpenAI, RateLimitError, APIError
from dotenv import load_dotenv
from scipy.spatial.distance import cosine

# Load environment variables
load_dotenv("../.env")

# Check required environment variables
required_env_vars = [
    "PINECONE_API_KEY",
    "AZURE_OPENAI_ENDPOINT", 
    "OPENAI_EMBEDDING_TEST_KEY"
]

missing_vars = []
for var in required_env_vars:
    if not os.environ.get(var):
        missing_vars.append(var)

if missing_vars:
    st.error(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
    st.error("Please configure these variables in ../.env file")
    st.stop()
else:
    st.success(f"✅ All required environment variables are configured")

# Streamlit page config
st.set_page_config(
    page_title="Vietnamese Product Search with Pinecone",
    page_icon="🛍️",
    layout="wide"
)

# Initialize Azure OpenAI client (using settings from assignment_8)
@st.cache_resource
def get_openai_client():
    try:
        client = AzureOpenAI(
            api_version="2024-07-01-preview",
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["OPENAI_EMBEDDING_TEST_KEY"],
        )
        return client
    except Exception as e:
        st.error(f"❌ Failed to create Azure OpenAI client: {str(e)}")
        raise e

# Initialize Pinecone client
@st.cache_resource
def get_pinecone_client():
    try:
        client = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        return client
    except Exception as e:
        st.error(f"❌ Failed to create Pinecone client: {str(e)}")
        raise e

# Load product data
@st.cache_data
def load_products():
    try:
        with open('vietnamese_products.json', 'r', encoding='utf-8') as f:
            products = json.load(f)
        return products
    except FileNotFoundError:
        st.error("vietnamese_products.json file not found. Please run convert_products.py first.")
        return []
    except json.JSONDecodeError:
        st.error("Error reading vietnamese_products.json. The file might be corrupted.")
        return []

# Function to get embeddings using text-embedding-3-small with dimension 512
def get_embedding(text, client):
    """Get 512-dimensional embedding from Azure OpenAI (using available model)"""
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",  # Using small model (available with current API key)
            input=text,
            dimensions=512  # Small model supports up to 512 dimensions
        )
        
        embedding = response.data[0].embedding
        return embedding
        
    except Exception as e:
        st.error(f"❌ Error getting embedding: {str(e)}")
        raise e

# Setup Pinecone index
@st.cache_resource
def setup_pinecone_index():
    try:
        pc = get_pinecone_client()
        index_name = "coolmate-index"
        
        # Check if index exists, if not create it
        existing_indexes = [index["name"] for index in pc.list_indexes()]
        
        if index_name not in existing_indexes:
            pc.create_index(
                name=index_name,
                dimension=512,  # Updated dimension to 512 (matching text-embedding-3-small)
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            st.info(f"Created new Pinecone index: {index_name}")
        
        index = pc.Index(index_name)
        return index
        
    except Exception as e:
        st.error(f"❌ Error setting up Pinecone index: {str(e)}")
        raise e

# Generate and upsert product embeddings
@st.cache_data
def generate_and_upsert_embeddings(_products, _client, _index):
    """Generate embeddings for all products and upsert to Pinecone"""
    try:
        vectors_to_upsert = []
        
        # Check if index already has data
        stats = _index.describe_index_stats()
        
        if stats['total_vector_count'] > 0:
            st.info(f"Index already contains {stats['total_vector_count']} vectors. Skipping embedding generation.")
            return
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, product in enumerate(_products):
            try:
                # Create rich description for embedding
                rich_description = f"{product['title']} {product['category']} {product['description']}"
                
                # Get embedding
                embedding = get_embedding(rich_description, _client)
                
                # Prepare vector for upsert
                vector = {
                    "id": product["id"],
                    "values": embedding,
                    "metadata": {
                        "title": product["title"],
                        "category": product["category"],
                        "price": product["price"],
                        "description": product["description"],
                        "discount_percent": product.get("discount_percent", 0)
                    }
                }
                vectors_to_upsert.append(vector)
                
                # Update progress
                progress = (i + 1) / len(_products)
                progress_bar.progress(progress)
                status_text.text(f"Processing product {i + 1}/{len(_products)}: {product['title'][:50]}...")
                
                # Batch upsert every 50 vectors to avoid memory issues
                if len(vectors_to_upsert) >= 50:
                    _index.upsert(vectors=vectors_to_upsert)
                    vectors_to_upsert = []
                    
            except Exception as e:
                st.error(f"Error processing product {product['id']}: {str(e)}")
                continue
        
        # Upsert remaining vectors
        if vectors_to_upsert:
            _index.upsert(vectors=vectors_to_upsert)
        
        progress_bar.empty()
        status_text.empty()
        st.success(f"Successfully uploaded {len(_products)} product embeddings to Pinecone!")
        
        # Verify final stats
        final_stats = _index.describe_index_stats()
        st.info(f"Final index stats - Total vectors: {final_stats['total_vector_count']}")
        
    except Exception as e:
        st.error(f"❌ Error in embedding generation: {str(e)}")
        raise e

def search_similar_products(query, client, index, top_k=5):
    """Search for similar products using query embedding"""
    try:
        # Get embedding for the query
        query_embedding = get_embedding(query, client)
        
        # Query Pinecone
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        return results.matches
    except Exception as e:
        st.error(f"❌ Error during search: {str(e)}")
        return []

def main():
    # Initialize clients and data
    try:
        client = get_openai_client()
    except Exception as e:
        st.error(f"❌ Failed to initialize OpenAI client: {str(e)}")
        st.stop()
    
    try:
        products = load_products()
    except Exception as e:
        st.error(f"❌ Failed to load products: {str(e)}")
        st.stop()
    
    if not products:
        st.error("No products loaded. Cannot continue.")
        st.stop()
    
    try:
        # Setup Pinecone
        index = setup_pinecone_index()
    except Exception as e:
        st.error(f"❌ Failed to setup Pinecone: {str(e)}")
        st.stop()
    
    # Streamlit UI
    st.title("🛍️ Vietnamese Product Search Engine")
    st.markdown("*Powered by Pinecone Vector Database & Azure OpenAI Embeddings (512D)*")
    
    # Sidebar for statistics and controls
    st.sidebar.markdown("### 📊 Database Statistics")
    
    # Get Pinecone index stats
    try:
        stats = index.describe_index_stats()
        st.sidebar.metric("Products in Vector DB", stats['total_vector_count'])
        st.sidebar.metric("Vector Dimension", "512")
        st.sidebar.metric("Total Products Loaded", len(products))
    except Exception as e:
        st.sidebar.error(f"Error getting stats: {str(e)}")
    
    # Control panel
    st.sidebar.markdown("### ⚙️ Controls")
    
    # Button to generate embeddings
    if st.sidebar.button("🔄 Generate & Upload Embeddings"):
        with st.spinner("Generating embeddings and uploading to Pinecone..."):
            try:
                generate_and_upsert_embeddings(products, client, index)
            except Exception as e:
                st.error(f"❌ Failed to generate embeddings: {str(e)}")
    
    # Search parameters
    st.sidebar.markdown("### 🔍 Search Settings")
    top_k = st.sidebar.slider("Number of results", min_value=1, max_value=20, value=5)
    
    # Category filter
    categories = sorted(list(set(p['category'] for p in products)))
    selected_category = st.sidebar.selectbox("Filter by Category", ["All Categories"] + categories)
    
    # Price range filter
    all_prices = [p['price'] for p in products]
    min_price = min(all_prices) if all_prices else 0
    max_price = max(all_prices) if all_prices else 1000000
    
    price_range = st.sidebar.slider(
        "Price Range (VND)",
        min_value=int(min_price),
        max_value=int(max_price),
        value=(int(min_price), int(max_price)),
        step=10000
    )
    
    # Main search interface
    st.markdown("### 🔍 Search Products")
    query = st.text_input(
        "Enter your search query in Vietnamese or English:",
        placeholder="e.g., áo thun cotton, quần jean nam, sports shirt..."
    )
    
    if query:
        with st.spinner("Searching for similar products..."):
            matches = search_similar_products(query, client, index, top_k=top_k * 2)  # Get more results for filtering
            
            if matches:
                # Filter results based on category and price
                filtered_matches = []
                for match in matches:
                    metadata = match.metadata
                    if (selected_category == "All Categories" or metadata['category'] == selected_category) and \
                       (price_range[0] <= metadata['price'] <= price_range[1]):
                        filtered_matches.append(match)
                        if len(filtered_matches) >= top_k:
                            break
                
                if filtered_matches:
                    st.markdown(f"### 🎯 Found {len(filtered_matches)} similar products:")
                    
                    # Display results in columns
                    cols = st.columns(2)
                    
                    for i, match in enumerate(filtered_matches):
                        metadata = match.metadata
                        score = match.score
                        
                        with cols[i % 2]:
                            with st.expander(f"**{metadata['title']}** - {metadata['price']:,.0f}đ", expanded=True):
                                st.markdown(f"**Category:** {metadata['category']}")
                                st.markdown(f"**Description:** {metadata['description']}")
                                
                                if metadata.get('discount_percent', 0) > 0:
                                    st.markdown(f"🏷️ **Discount:** {metadata['discount_percent']}% off")
                                
                                # Similarity score with progress bar
                                st.markdown(f"**Similarity Score:** {score:.4f}")
                                st.progress(score)
                                
                                # Color-coded score
                                if score > 0.8:
                                    st.success("🎯 Excellent match!")
                                elif score > 0.6:
                                    st.info("✅ Good match")
                                else:
                                    st.warning("🔍 Partial match")
                
                else:
                    st.warning("No products found matching your filters. Try adjusting the category or price range.")
            else:
                st.error("No similar products found. Please try a different search query.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🔧 Enhanced with 512D embeddings from Azure OpenAI | 
        🗄️ Vector search powered by Pinecone | 
        🎨 Built with Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
