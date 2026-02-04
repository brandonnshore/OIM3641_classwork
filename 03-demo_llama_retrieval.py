from dotenv import load_dotenv
from llama_cloud_services import LlamaCloudIndex
from llama_index.llms.openai import OpenAI
import os

"""
pip install llama-index-llms-huggingface
pip install llama-index-llms-gemini
pip install llama-index-embeddings-huggingface
pip install sentence-transformers

"""

load_dotenv()
api_key = os.getenv("LLAMA_CLOUD_API_KEY")
# get this from llama-cloud
org_id = os.getenv("ORGANIZATION_ID")

index = LlamaCloudIndex(
    name="Odyssey",
    project_name="Default",
    api_key=api_key,
    organization_id=org_id)
query = "What is this document about?"
nodes = index.as_retriever()
response = nodes.retrieve(query)

print("Query response:")
print(response[4])
