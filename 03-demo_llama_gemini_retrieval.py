"""
pip install llama-index-llms-google-genai
pip install llama-index-embeddings-google-genai

"""

import os
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.core.query_engine import RetrieverQueryEngine
# Google GenAI Imports
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
# Llama Cloud Import
from llama_index.indices.managed.llama_cloud import LlamaCloudRetriever

# 1. Load environment variables
load_dotenv()
llama_cloud_key = os.getenv("LLAMA_CLOUD_API_KEY")
org_id = os.getenv("ORGANIZATION_ID")
google_api_key = os.getenv("GEMINI_API_KEY") # Ensure this is in your .env

if not llama_cloud_key or not org_id or not google_api_key:
    raise ValueError("Missing required environment variables.")

# 2. Configure LlamaIndex to use Google GenAI
# Using Gemini 2.5 Flash for fast, low-cost reasoning
Settings.llm = GoogleGenAI(
    model="models/gemini-2.5-flash",
    api_key=google_api_key
)

# Using Google's modern text-embedding model
Settings.embed_model = GoogleGenAIEmbedding(
    model_name="models/text-embedding-004",
    api_key=google_api_key
)

# 3. Initialize the retriever to fetch nodes from your Llama Cloud index
retriever = LlamaCloudRetriever(
    name="Odyssey",
    project_name="Default",
    api_key=llama_cloud_key,
    organization_id=org_id
)

# 4. Create a query engine
query_engine = RetrieverQueryEngine(
    retriever=retriever,
)

# 5. Run the query
query = "Who is the president of france currently?"
response = query_engine.query(query)

# 6. Print the response
print("Query response:")
print(response)
