from dotenv import load_dotenv
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
import os
import streamlit as st

load_dotenv()

Settings.llm = GoogleGenAI(model='gemini-2.5-flash')
Settings.embed_model = GoogleGenAIEmbedding(model_name='models/gemini-embedding-001')


# --- CORE LOGIC ---
@st.cache_resource
def get_query_engine():
    # assumes documents are in a directory called 'data'
    documents = SimpleDirectoryReader('data').load_data()
    # create a vector database
    index = VectorStoreIndex.from_documents(documents)
    # query the index
    query_engine = index.as_query_engine()
    return query_engine


# --- STREAMLIT UI ---
st.title("Babson Student Handbook Chatbot")
prompt = st.chat_input("Ask me a question...")

if prompt:
    st.write(f"User: {prompt}")
    query_engine = get_query_engine()
    response = query_engine.query(prompt)
    st.write(f"Response: {response}")