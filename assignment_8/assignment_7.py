import os
import json
import streamlit as st
from openai import AzureOpenAI, RateLimitError, APIError
from scipy.spatial.distance import cosine
from dotenv import load_dotenv

# Load environment variables
load_dotenv("../.env")

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

# Step 2: Load products from JSON file
try:
    with open('products.json', 'r', encoding='utf-8') as f:
        products = json.load(f)
    st.sidebar.success(f"Loaded {len(products)} products from products.json")
except FileNotFoundError:
    st.error("products.json file not found. Please run generate_products.py first.")
    products = []
except json.JSONDecodeError:
    st.error("Error reading products.json. The file might be corrupted.")
    products = []

# Step 3: Function to get embeddings from Azure OpenAI
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
        # Create rich text for embedding by combining title, category, and description
        rich_description = f"{product_copy['title']} {product_copy['category']} {product_copy['short_description']}"
        print(product_copy["short_description"])
        product_copy["embedding"] = get_embedding(product_copy["short_description"], _client)
        products_with_embeddings.append(product_copy)
    return products_with_embeddings

products = generate_product_embeddings(products, client)

# Streamlit UI
st.title("🔍 Smart Product Search")

# Add category filter in sidebar
categories = sorted(list(set(p['category'] for p in products)))
selected_category = st.sidebar.selectbox(
    "Filter by Category",
    ["All Categories"] + categories
)

# Add price range filter
all_prices = [p['price'] for p in products]
min_price = min(all_prices) if all_prices else 0
max_price = max(all_prices) if all_prices else 1000

price_range = st.sidebar.slider(
    "Price Range ($)",
    min_value=float(min_price),
    max_value=float(max_price),
    value=(float(min_price), float(max_price))
)

# Filter products based on category and price range
filtered_products = [
    p for p in products
    if (selected_category == "All Categories" or p['category'] == selected_category)
    and (price_range[0] <= p['price'] <= price_range[1])
]

st.write(f"Search through {len(filtered_products)} products using natural language!")

# Display some statistics
st.sidebar.markdown("---")
st.sidebar.markdown("### Product Statistics")
st.sidebar.markdown(f"- Total Products: {len(products)}")
st.sidebar.markdown(f"- Filtered Products: {len(filtered_products)}")
st.sidebar.markdown(f"- Categories: {len(categories)}")
st.sidebar.markdown(f"- Price Range: ${min_price:.2f} - ${max_price:.2f}")

# Step 5: Accept user input (query)
query = st.text_input("Enter your search query:", placeholder="e.g., warm cotton sweatshirt")

if query:
    # Get embedding for the query
    with st.spinner("Searching for products..."):
        try:
            query_embedding = get_embedding(query, client)
            
            # Compute similarity scores
            scores = []
            for product in filtered_products:
                score = similarity_score(query_embedding, product["embedding"])
                scores.append((score, product.copy()))
            
            # Sort products by similarity
            scores.sort(key=lambda x: x[0], reverse=True)
            
            # Display results
            st.subheader("Search Results")
            
            for score, product in scores[:3]:  # Show top 3 results
                with st.expander(f"{product['title']} - ${product['price']}", expanded=True):
                    st.write(f"**Description:** {product['short_description']}")
                    st.write(f"**Category:** {product['category']}")
                    st.progress(score)
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