
import streamlit as st
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain_ollama import ChatOllama
from langchain.text_splitter import CharacterTextSplitter

@st.cache_resource
def load_products():
    with open("/Users/aarnavhariramani/NLPResearchLab/eBayCB/products.json", "r") as f:
        data = json.load(f)
    return data

products = load_products()

def create_documents(products):
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
    return SentenceTransformer('all-MiniLM-L6-v2')

embedding_model = load_embedding_model()

@st.cache_data
def embed_text(text):
    return embedding_model.encode(text)

@st.cache_resource
def create_faiss_index(chunks):
    embeddings = embedding_model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype("float32"))
    return index, embeddings, chunks


faiss_index, chunk_embeddings, chunk_texts = create_faiss_index(chunks)

def retrieve_context(query, k=3):
    """
    Given a user query, embed it and retrieve the top-k most relevant product chunks.
    """
    query_embedding = embed_text(query)
    D, I = faiss_index.search(np.array([query_embedding]).astype("float32"), k)
    retrieved_chunks = [chunk_texts[i] for i in I[0] if i < len(chunk_texts)]
    context = "\n\n".join(retrieved_chunks)
    return context

def generate_response(user_query):
    context = retrieve_context(user_query, k=3)
    
    prompt = (
        f"Here is some context about eBay products:\n{context}\n\n"
        f"Answer the following question based on the above context:\n{user_query}"
    )
    
    model = ChatOllama(model="llama3.2:3b", base_url="http://localhost:11434/") 
    response = model.invoke(prompt)
    return response.content


st.title("EBAY Chatbot || Aarnav Hariramani || NLP Research Lab")

with st.form("llm-form"):
    user_input = st.text_area("Enter your question or statement:")
    submit = st.form_submit_button("Submit")

if "chat_history" not in st.session_state:
    st.session_state['chat_history'] = []

if submit and user_input:
    with st.spinner("Generating response..."):
        try:
            response_text = generate_response(user_input)
            st.session_state['chat_history'].append({"user": user_input, "bot": response_text})
            st.write(response_text)
        except Exception as e:
            st.error(f"Error generating response: {e}")

st.write("## Chat History")
for chat in reversed(st.session_state['chat_history']):
    st.write(f"**User:** {chat['user']}")
    st.write(f"**Bot:** {chat['bot']}")
    st.write("---")