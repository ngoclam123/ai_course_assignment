import streamlit as st
import chromadb
from openai import AzureOpenAI
import os
from dotenv import load_dotenv
import json

# Page config
st.set_page_config(
    page_title="Laptop Recommendation Bot",
    page_icon="💻",
    layout="wide"
)

# Load environment variables
load_dotenv("../.env")

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# ---- CLIENTS ----


@st.cache_resource
def get_openai_clients():
    embedding_client = AzureOpenAI(
        api_version="2024-07-01-preview",
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["OPENAI_EMBEDDING_TEST_KEY"],
    )
    llm_client = AzureOpenAI(
        api_version="2024-07-01-preview",
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
    )
    return embedding_client, llm_client


embedding_client, llm_client = get_openai_clients()


def get_embedding(_text):
    response = embedding_client.embeddings.create(
        input=_text,
        model=os.environ["AZURE_OPENAI_EMBED_MODEL"],
    )
    return response.data[0].embedding


def ask_llm(context, user_input):

    system_prompt = """
You are a helpful laptop recommendation assistant.
Use the provided context and conversation history to recommend the best laptop(s) for the user's needs.
Reply in a friendly, conversational style and short but be concise. Must provide laptop name and prices.
Return in hierachy format.
"""

    user_prompt = (
        f"User requirements: {user_input}\n\n"
        f"Context (top relevant laptops):\n{context}\n\n"
        "Based on the above, which laptop(s) would you recommend and why?"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    response = llm_client.chat.completions.create(
        model=os.environ["AZURE_OPENAI_LLM_MODEL"],
        messages=messages
    )
    return response.choices[0].message.content

# ---- CHROMADB ----


@st.cache_resource
def setup_chromadb():
    chroma_client = chromadb.Client()
    try:
        collection = chroma_client.get_collection("laptops")
        chroma_client.delete_collection("laptops")
    except:
        pass
    return chroma_client.create_collection(name="laptops")


collection = setup_chromadb()

# Load laptops from JSON


def load_laptops():
    try:
        with open('laptops.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("laptops.json not found. Please run generate_laptops.py first.")
        return []

# Add laptops to ChromaDB


@st.cache_resource
def initialize_laptop_data():
    laptops = load_laptops()
    for laptop in laptops:
        embedding = get_embedding(laptop["description"])
        collection.add(
            ids=[laptop["id"]],
            embeddings=[embedding],
            metadatas=[{
                "name": laptop["name"],
                "tags": laptop["tags"],
                "price": str(laptop["price"]),
                "processor": laptop["processor"],
                "ram": laptop["ram"],
                "storage": laptop["storage"],
                "graphics": laptop["graphics"],
            }],
            documents=[laptop["description"]],
        )
    return len(laptops)


def build_context(results):
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    context_str = ""
    for doc, meta in zip(docs, metas):
        context_str += (
            f"Name: {meta['name']}\n"
            f"Description: {doc}\n"
            f"Price: ${meta['price']}\n"
            f"Processor: {meta['processor']}\n"
            f"RAM: {meta['ram']}\n"
            f"Storage: {meta['storage']}\n"
            f"Graphics: {meta['graphics']}\n"
            f"Tags: {meta['tags']}\n\n"
        )
    return context_str.strip()


# ---- UI COMPONENTS ----
st.title("💻 Smart Laptop Recommendation System")

# Initialize data
num_laptops = initialize_laptop_data()
st.sidebar.success(f"Loaded {num_laptops} laptops into the system")

# Sample queries in sidebar
st.sidebar.markdown("### Sample Queries")
sample_queries = [
]

# Query input
query = st.text_input("What kind of laptop are you looking for?",
                      placeholder="Describe your needs...")

# Sample query buttons
st.sidebar.markdown("Click to try sample queries:")
for sample in sample_queries:
    if st.sidebar.button(f"🔄 {sample[:50]}...", key=sample):
        query = sample
        # st.experimental_rerun()

if query:
    # Add user query to chat history
    st.session_state.chat_history.append({"role": "user", "content": query})

    with st.spinner("Searching for recommendations..."):
        # Get recommendations
        query_embedding = get_embedding(query)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )

        # Build context and get LLM response
        context = build_context(results)
        llm_response = ask_llm(context, query)

        # Add response to chat history
        st.session_state.chat_history.append(
            {"role": "assistant", "content": llm_response})

# Display chat history
st.markdown("### Conversation History")
for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.markdown(f"👤 **You**: {message['content']}")
    else:
        st.markdown(f"🤖 **Assistant**: {message['content']}")
    st.markdown("---")

# Clear chat button
if st.button("Clear Chat History"):
    st.session_state.chat_history = []
    st.experimental_rerun()

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.markdown("""
This system uses:
- Azure OpenAI for embeddings and chat
- ChromaDB for vector search
- Streamlit for the interface
""")
