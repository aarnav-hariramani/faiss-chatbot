import streamlit as st
import json
import numpy as np
import faiss
import requests  
from sentence_transformers import SentenceTransformer
from together import Together 
from langchain.text_splitter import CharacterTextSplitter

TOGETHER_API_KEY = "13cd356ee8fdc4ef7c62bf5bd95519c9711bb8955ac8f61b8bc3ed92ed50597d"
TOGETHER_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"

client = Together(api_key=TOGETHER_API_KEY)

@st.cache_resource
def load_products():
    """Load eBay product data from a JSON file."""
    with open("/Users/aarnavhariramani/NLPResearchLab/eBayCB/products.json", "r") as f:
        data = json.load(f)
    return data

products = load_products()

def create_documents(products):
    """Convert eBay product data into structured text format for embedding."""
    docs = []
    for product in products:
        doc = (
            f"Product Name: {product.get('name', '')}\n"
            f"Price: {product.get('price', '')}\n"
            f"Brand: {product.get('brand', '')}\n"
            f"Size: {product.get('size', '')}\n"
            f"Description: {product.get('description', '')}\n"
            f"Link: {product.get('link', '')}"
        )
        docs.append(doc)
    return docs

documents = create_documents(products)


text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)


chunks = []
for doc in documents:
    chunks.extend(text_splitter.split_text(doc))

@st.cache_resource
def load_embedding_model():
    """Load the SentenceTransformer model for embedding text into vectors."""
    return SentenceTransformer("all-MiniLM-L6-v2")


embedding_model = load_embedding_model()

@st.cache_data
def embed_text(text):
    """Convert text into a numerical vector using the embedding model."""
    return embedding_model.encode(text)

@st.cache_resource
def create_faiss_index(chunks):
    """Create a FAISS index for fast similarity search of product descriptions."""
    embeddings = embedding_model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype("float32"))
    return index, embeddings, chunks

# Initialize FAISS index for vector search
faiss_index, chunk_embeddings, chunk_texts = create_faiss_index(chunks)

def retrieve_context(query, k=3):
    """
    Given a user query, embed it and retrieve the top-k most relevant product chunks.
    Returns the full product dictionaries (which include links).
    """
    query_embedding = embed_text(query)
    D, I = faiss_index.search(np.array([query_embedding]).astype("float32"), k)
    retrieved_products = []

    for i in I[0]:
        if i < len(products):
            retrieved_products.append(products[i])
    return retrieved_products

def generate_response(user_query):
    """
    Generate a chatbot response based on retrieved eBay product descriptions using Together AI API.
    The prompt instructs the LLM to use the provided context only if the query is product-related.
    """
    retrieved_products = retrieve_context(user_query, k=3)
    
    context = "\n\n".join(
        [
            f"Product Name: {p['name']}\n"
            f"Price: {p['price']}\n"
            f"Description: {p['description']}\n"
            f"Link: {p['link']}\n"
            for p in retrieved_products
        ]
    )
    
    prompt = (
        "You are an intelligent assistant. Below is some context about eBay products. "
        "If the user's query is about shopping or products, use the context to suggest relevant eBay products (include the product names and links). "
        "If the user's query is not about products, ignore the context and answer the question normally.\n\n"
        f"Context:\n{context}\n\n"
        f"User's query: {user_query}"
    )
    
    response = client.chat.completions.create(
        model=TOGETHER_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content

st.title("EBAY Chatbot || Aarnav Hariramani || NLP Research Lab")

with st.form("llm-form"):
    user_input = st.text_area("Enter your question or statement:")
    submit = st.form_submit_button("Submit")

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if submit and user_input:
    with st.spinner("Generating response..."):
        try:
            response_text = generate_response(user_input)
            st.session_state["chat_history"].append({"user": user_input, "bot": response_text})
            st.write(response_text)
        except Exception as e:
            st.error(f"Error generating response: {e}")

st.write("## Chat History")
for chat in reversed(st.session_state["chat_history"]):
    st.write(f"**User:** {chat['user']}")
    st.write(f"**Bot:** {chat['bot']}")
    st.write("---")
