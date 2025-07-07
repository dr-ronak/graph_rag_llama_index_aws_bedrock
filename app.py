# !pip install python-dotenv
# !pip install datasets
# !pip install langchain
# !pip install neo4j
# !pip install llama-index
# !pip install boto3 # Required for AWS interaction
# !pip install llama-index-llms-bedrock-converse
# !pip install llama-index-embeddings-bedrock
# !pip install llama-index-graph-stores-neo4j

import os
from dotenv import load_dotenv
import logging
import sys

# --- IMPORT: Use Bedrock from integrations ---
# Use the specific integration imports as recommended by LlamaIndex
from llama_index.llms.bedrock_converse import BedrockConverse
from llama_index.embeddings.bedrock import BedrockEmbedding
from llama_index.graph_stores.neo4j import Neo4jGraphStore

# Import necessary LlamaIndex core components
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex, # Although not used in the final KG index creation, good to keep if needed
    StorageContext,
    # ServiceContext is deprecated, use Settings instead
    KnowledgeGraphIndex,
    Settings
)


load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_DEFAULT_REGION = os.getenv('AWS_REGION')
BUCKET_NAME = os.getenv('BUCKET_NAME') # Not used in this script, but kept for context
LLM_MODEL_ID = os.getenv('BEDROCK_LLM_MODEL_ID')
EMBEDDING_MODEL_ID = os.getenv('EMBEDDING_MODEL_ID')

NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
NEO4J_DATABASE = os.getenv('NEO4J_DATABASE') # Not explicitly used in Neo4jGraphStore, defaults often used


# Configure logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout)) # Ensure logs go to stdout

# Initialize the Bedrock LLM
llm = BedrockConverse(
    model=LLM_MODEL_ID,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_DEFAULT_REGION
)

# Initialize the Bedrock Embedding model
embed_model = BedrockEmbedding(model=EMBEDDING_MODEL_ID)

# Configure LlamaIndex Settings (replaces ServiceContext)
Settings.llm = llm
Settings.chunk_size = 512
Settings.embed_model = embed_model

# Load documents from the data directory
try:
    documents = SimpleDirectoryReader("data/").load_data()
    print(f"Loaded {len(documents)} documents.")
    # Optional: print first document content snippet for verification
    # if documents:
    #     print(f"First document content snippet: {documents[0].get_content()[:200]}...")
except Exception as e:
    print(f"Error loading documents: {e}")
    sys.exit(1) # Exit if document loading fails

# Set up Neo4j Graph Store
url = os.environ['NEO4J_URI']
username = os.environ['NEO4J_USERNAME']
password = os.environ['NEO4J_PASSWORD']
database = "neo4j" # Default database name

try:
    graph_store = Neo4jGraphStore(
        username=username,
        password=password,
        url=url,
        database=database,
    )
    print("Neo4j Graph Store initialized successfully.")
except Exception as e:
    print(f"Error initializing Neo4j Graph Store: {e}")
    print("Please ensure your Neo4j instance is running and credentials are correct.")
    sys.exit(1) # Exit if graph store initialization fails


# Create StorageContext with the graph store
storage_context = StorageContext.from_defaults(graph_store=graph_store)
print("StorageContext created.")

# --- Knowledge Graph Indexing ---


# With embeddings (for hybrid search)
print("\n--- Creating Knowledge Graph Index (with embeddings) ---")
try:
    # Note: ServiceContext is removed, Settings is used implicitly
    index_with_embed = KnowledgeGraphIndex.from_documents(
        documents,
        storage_context=storage_context,
        max_triplets_per_chunk=2,
        # service_context is removed
        include_embeddings=True, # Explicitly set to True
    )
    print("Knowledge Graph Index (with embeddings) created.")

    # Query Engine with embeddings for hybrid search
    print("\n--- Querying Index (with embeddings/hybrid) ---")
    query_engine_with_embed = index_with_embed.as_query_engine(
        include_text=True,
        response_mode="tree_summarize",
        embedding_mode="hybrid", # Use hybrid mode for search
        similarity_top_k=5,
    )
    print("Question-1:")
    response_with_embed = query_engine_with_embed.query("What are the authors details?")
    print(f"Response (with embeddings/hybrid): {response_with_embed}")
    print("Question-2:")
    response_with_embed = query_engine_with_embed.query("list out all research paper title.")
    print(f"Response (with embeddings/hybrid): {response_with_embed}")

except Exception as e:
    print(f"Error during KG index creation or querying (with embeddings): {e}")