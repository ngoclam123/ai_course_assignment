import os
import streamlit as st
from openai import AzureOpenAI, RateLimitError, APIError
from scipy.spatial.distance import cosine
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="Product Search Engine",
    page_icon="🔍",
    layout="wide"
)

# Step 1: Setup AzureOpenAI
@st.cache_resource
def get_openai_client():
    return AzureOpenAI(
        api_version="2024-07-01-preview",
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["OPENAI_EMBEDDING_TEST_KEY"],
    )

client = get_openai_client()

# Step 2: Sample product data
products = [
    {
        "title": "Classic Blue Jeans",
        "short_description": "Comfortable blue denim jeans with a relaxed fit.",
        "price": 49.99,
        "category": "Jeans"
    },
    {
        "title": "Red Hoodie",
        "short_description": "Cozy red hoodie made from organic cotton.",
        "price": 39.99, "category": "Hoodies"
    },
    {
        "title": "Black Leather Jacket",
        "short_description": "Stylish black leather jacket with a slim fit design.",
        "price": 120.00,
        "category": "Jackets"
    },
    # Add more products as needed
]

# Step 3: Function to get embeddings from Azure OpenAI
@st.cache_data
def get_embedding(_text, _client):
    response = _client.embeddings.create(
        model="text-embedding-3-small",
        input=_text
    )
    return response.data[0].embedding

# Function to compute similarity score
def similarity_score(vec1, vec2):
    return 1 - cosine(vec1, vec2)

# Generate embeddings for all products (only once)
@st.cache_data
def generate_product_embeddings(_products, _client):
    products_with_embeddings = []
    for product in _products:
        product_copy = product.copy()
        product_copy["embedding"] = get_embedding(product_copy["short_description"], _client)
        products_with_embeddings.append(product_copy)
    return products_with_embeddings

products = generate_product_embeddings(products, client)

# Streamlit UI
st.title("🔍 Smart Product Search")
st.write("Search for products using natural language!")

# Step 5: Accept user input (query)
query = st.text_input("Enter your search query:", placeholder="e.g., warm cotton sweatshirt")
if query:
    # Get embedding for the query
    with st.spinner("Searching for products..."):
        try:
            query_embedding = get_embedding(query, client)
            
            # Compute similarity scores
            scores = []
            for product in products:
                score = similarity_score(query_embedding, product["embedding"])
                scores.append((score, product.copy()))
            
            # Sort products by similarity
            scores.sort(key=lambda x: x[0], reverse=True)
            
            # Display results
            st.subheader("Search Results")
            
            for score, product in scores[:3]:
                with st.expander(f"{product['title']} - ${product['price']}", expanded=True):
                    st.write(f"**Description:** {product['short_description']}")
                    st.write(f"**Category:** {product['category']}")
                    st.progress(score)  # Show similarity score as a progress bar
                    st.caption(f"Similarity Score: {score:.4f}")
                    
        except (RateLimitError, APIError) as e:
            st.error(f"Error occurred while processing your request: {str(e)}")
            
# Add some styling
st.markdown("""
    <style>
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    </style>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("Powered by Azure OpenAI Embeddings")