import os
import logging
import sys

from llama_index.embeddings.openai import OpenAIEmbedding

from llama_index.llms.azure_openai import AzureOpenAI 
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import (
    Settings, 
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    PromptTemplate,
    SimpleDirectoryReader
    )

logging.basicConfig(
    stream=sys.stdout, level=logging.INFO
)  # logging.DEBUG for more verbose output
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

api_key = "voxelcloud"
azure_endpoint = "http://192.168.12.232:8881"
api_version = "2023-12-01-preview"

llm = AzureOpenAI(
    engine="gpt4-1106-prevision",
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version,
)

# response = llm.complete("你好") 
# print(response)

embed_model = AzureOpenAIEmbedding(
    model="text-embedding-ada-002",
    deployment_name="embedding-ada-002-v2",
    api_key=api_key,
    azure_endpoint="http://192.168.12.232:8880",
    api_version="2023-05-15",
)

# embed_model = OpenAIEmbedding(
#     model="text-embedding-ada-002",  
#     api_key = "voxelcloud",  
#     api_version = "2023-05-15",
#     api_base = "http://192.168.12.232:8880"
#     )

# embeddings = embed_model.get_text_embedding("Hello World!")
# print(len(embeddings))
# print(embeddings[:5])

# embed_model = HuggingFaceEmbedding(model_name="/home/jyhu/xiaogpt/rag/bge-large-zh-v1.5")
Settings.embed_model = embed_model
Settings.llm = llm

# check if storage already exists
PERSIST_DIR = "./storage"
if not os.path.exists(PERSIST_DIR):
    # load the documents and create the index
    documents = SimpleDirectoryReader("./data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    # store it for later
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    # load the existing index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

# set Logging to DEBUG for more detailed outputs

text_qa_template_str = (
    "Context information is"
    " below.\n---------------------\n{context_str}\n---------------------\nUsing"
    " both the context information and also using your own knowledge, answer"
    " the question with less that 100 words: {query_str}\nIf the context isn't helpful, you can also"
    " answer the question on your own.\n"
)
text_qa_template = PromptTemplate(text_qa_template_str)

refine_template_str = (
    "The original question is as follows: {query_str}\nWe have provided an"
    " existing answer: {existing_answer}\nWe have the opportunity to refine"
    " the existing answer (only if needed) with some more context"
    " below.\n------------\n{context_msg}\n------------\nUsing both the new"
    " context and your own knowledge, update existing answer with less than 100 words. \n"
)
refine_template = PromptTemplate(refine_template_str)


query_engine = index.as_query_engine(
    text_qa_template=text_qa_template, 
    refine_template=refine_template,
    llm=llm
    )


response = query_engine.query("什么是光疗？")

print(response)

