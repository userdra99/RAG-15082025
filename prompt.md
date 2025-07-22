ROLE: You are an expert Python developer specializing in building high-performance, production-quality, and fully containerized AI applications. Your task is to create a self-contained prototype of a Retrieval-Augmented Generation (RAG) system based on the updated architecture defined below. The entire system must be orchestrated and run using Docker.

GOAL: Develop a modular RAG pipeline that allows a user to upload documents (PDF, DOCX, Excel) and ask questions about their content via a simple web interface. The system must use vLLM for serving both the generative and embedding models to ensure high throughput and low latency.

PROJECT REQUIREMENTS & TECH STACK:

Full Dockerization: The entire application, including the RAG pipeline, model servers, and vector database, must be defined in a single docker-compose.yml file for easy, one-command startup.

Model Serving with vLLM:

Use the official vllm/vllm-openai Docker image to serve both the LLM and the embedding model.   

Since vLLM does not support serving multiple models from a single instance , you will configure two separate vLLM services in    

docker-compose.yml.

LLM Service: Serve the unsloth/Llama-3.2-3B-Instruct model.   

Embedding Service: Serve the jinaai/jina-embeddings-v4-vllm-retrieval model.   

Assign each vLLM service to a different GPU if available (NVIDIA_VISIBLE_DEVICES).

API Gateway with Nginx:

Implement an Nginx reverse proxy to route incoming requests to the correct vLLM service based on the model name specified in the request body. This will expose a single API endpoint for the application to interact with.

Orchestration Framework: Use LlamaIndex to structure the entire RAG pipeline.

Document Loading and Parsing:

Use LlamaIndex's SimpleDirectoryReader to load .pdf, .docx, and .xlsx files from a local ./data directory.

Integrate the Docling library for robust parsing of PDF and DOCX files to preserve complex layouts and tables.   

For XLSX files, use pandas to iterate through each sheet, convert the data to a clean Markdown string, and treat each sheet as a separate document.

Text Splitting (Chunking):

Use the RecursiveCharacterTextSplitter from LlamaIndex with a chunk_size of 1024 and a chunk_overlap of 200.

Vector Database:

Use Qdrant as the vector store, running in its own Docker container.

LlamaIndex Integration:

Configure LlamaIndex's OpenAI and OpenAIEmbedding components to communicate with the vLLM services through the Nginx reverse proxy's OpenAI-compatible API endpoint.

User Interface:

Create a simple web application using Streamlit.

The UI must include a text input for questions, a submit button, a display area for the final answer, and an expander to show the retrieved source text chunks.

AGENTIC CODING CONTEXT:

To ensure the generated code is accurate and uses the latest library APIs, you will act as if you have access to the following Model Context Protocol (MCP) servers:

Context7 MCP Server: Use this to get up-to-date, version-specific documentation and code examples for all open-source libraries (LlamaIndex, Streamlit, Qdrant, etc.). This will prevent the use of deprecated functions and ensure the code is modern and efficient.   

GitHub MCP Server: Use this to access the official GitHub repositories for the specified libraries and models (vllm-project/vllm, unslothai/unsloth, jina-ai/jina-embeddings, etc.) to understand their structure, arguments, and best practices directly from the source code.   

DELIVERABLES:

Provide the complete source code and configuration files organized into the following structure:

docker-compose.yml:

Define five services:

vllm-llm: Serves the unsloth/Llama-3.2-3B-Instruct model.

vllm-embedding: Serves the jinaai/jina-embeddings-v4-vllm-retrieval model.

qdrant: The vector database service.

nginx: The reverse proxy routing requests to the correct vLLM service.

app: The main Streamlit/LlamaIndex application.

Ensure all services are on a shared Docker network for communication.

nginx/nginx.conf:

An Nginx configuration file that proxies requests to the appropriate upstream vLLM service based on the requested model name.

app/Dockerfile:

A Dockerfile for the Python application (app service). It should install all dependencies from requirements.txt and run the Streamlit app.

app/requirements.txt:

A list of all necessary Python libraries, including llama-index, streamlit, qdrant-client, pandas, openpyxl, docling, and all required llama-index integrations (e.g., llama-index-llms-openai, llama-index-embeddings-openai).

app/main.py:

The main Python script containing the Streamlit UI and the LlamaIndex RAG pipeline logic. This script should be configured to connect to the Nginx endpoint for both LLM and embedding model calls.

README.md:

Clear, step-by-step instructions on how to set up and run the entire application using docker-compose up. Explain how to add documents to the ./data folder and interact with the web UI.

Enhancement Plan: Implementing Hybrid Retrieval
After building the initial prototype, the next step to significantly improve retrieval accuracy is to implement a hybrid retrieval strategy. This approach combines traditional keyword-based search (like BM25) with modern semantic vector search to leverage the strengths of both methods.   

Here is a clear plan for this enhancement:

Enable Hybrid Search in Vector Store:

Modify the QdrantVectorStore initialization in app/main.py.

Set the enable_hybrid=True parameter. This configures the Qdrant collection to store both dense (semantic) and sparse (keyword) vectors.   

Specify a sparse vector model, such as Qdrant's built-in BM25, by setting fastembed_sparse_model="Qdrant/bm25".   

Update the Query Engine:

When creating the query engine in LlamaIndex using .as_query_engine(), set the vector_store_query_mode parameter to "hybrid".   

This instructs LlamaIndex to perform both a dense vector search and a sparse vector search in Qdrant and then apply a fusion algorithm to merge and re-rank the results.

Tune the Retrieval Balance:

Introduce the alpha parameter in the .as_query_engine() call. This parameter, ranging from 0 to 1, controls the balance between the keyword search (alpha=0.0) and the vector search (alpha=1.0).   

Start with a default value (e.g., alpha=0.5) and plan to experiment with different values to find the optimal balance for the specific types of documents and user queries the application will handle.