import os
from openai import AzureOpenAI

client = AzureOpenAI(
  api_key = "voxelcloud",  
  api_version = "2023-05-15",
  azure_endpoint = "http://192.168.12.232:8880"
)

response = client.embeddings.create(
    input = "Your text string goes here",
    model= "embedding-ada-002"
)

print(response.model_dump_json(indent=2))