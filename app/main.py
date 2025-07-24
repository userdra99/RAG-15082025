import os
import time
import logging
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import streamlit as st

from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    Settings, 
    StorageContext,
    Document
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.llms.openai import OpenAI
from llama_index.readers.file import PDFReader, DocxReader
from llama_index.core.postprocessor import SentenceTransformerRerank

from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

import qdrant_client

# Configure custom OpenAI-compatible model
CUSTOM_MODEL_CONFIG = {
    "unsloth/Llama-3.2-3B-Instruct": {
        "context_window": 8192,
        "is_chat_model": True,
        "is_function_calling_model": False,
        "is_vision_model": False,
    }
}

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
if 'index' not in st.session_state:
    st.session_state.index = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'last_document_hash' not in st.session_state:
    st.session_state.last_document_hash = None

class DoclingExcelReader:
    """Enhanced Excel reader using Docling with contextual chunking"""
    
    def load_data(self, file_path: str) -> List[Document]:
        """Load Excel file and create contextual chunks"""
        documents = []
        
        try:
            # Use LlamaIndex's SentenceSplitter for contextual chunking
            from llama_index.core.node_parser import SentenceSplitter
            text_splitter = SentenceSplitter(
                chunk_size=512,
                chunk_overlap=50,
                paragraph_separator="\n\n"
            )
            
            # Read all sheets
            excel_file = pd.ExcelFile(file_path)
            
            for sheet_name in excel_file.sheet_names:
                # Read sheet into DataFrame
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                
                # Convert to markdown
                markdown_content = f"# Sheet: {sheet_name}\n\n"
                markdown_content += df.to_markdown(index=False)
                
                # Create chunks with context
                chunks = text_splitter.split_text(markdown_content)
                
                for i, chunk_text in enumerate(chunks):
                    doc = Document(
                        text=chunk_text,
                        metadata={
                            "file_name": os.path.basename(file_path),
                            "sheet_name": sheet_name,
                            "source": file_path,
                            "chunk_id": i,
                            "chunk_type": "contextual",
                            "chunk_size": len(chunk_text)
                        }
                    )
                    documents.append(doc)
                
        except Exception as e:
            logger.error(f"Error reading Excel file {file_path}: {e}")
            
        return documents

class DoclingPDFReader:
    """Enhanced PDF reader using Docling for complex layouts with contextual chunking"""
    
    def __init__(self):
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        self.converter = DocumentConverter()
    
    def load_data(self, file_path: str) -> List[Document]:
        """Load PDF using Docling with contextual chunking"""
        try:
            conv_result = self.converter.convert(file_path)
            doc = conv_result.document
            
            # Use LlamaIndex's SentenceSplitter for contextual chunking
            from llama_index.core.node_parser import SentenceSplitter
            text_splitter = SentenceSplitter(
                chunk_size=512,
                chunk_overlap=50,
                paragraph_separator="\n\n"
            )
            
            # Convert docling doc to text and split
            full_text = doc.export_to_markdown()
            
            # Create chunks with context
            chunks = text_splitter.split_text(full_text)
            
            documents = []
            for i, chunk_text in enumerate(chunks):
                doc = Document(
                    text=chunk_text,
                    metadata={
                        "file_name": os.path.basename(file_path),
                        "source": file_path,
                        "type": "pdf",
                        "chunk_id": i,
                        "chunk_type": "contextual",
                        "chunk_size": len(chunk_text)
                    }
                )
                documents.append(doc)
            
            logger.info(f"Created {len(documents)} contextual chunks from {file_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Error processing PDF {file_path} with Docling: {e}")
            # Fallback to simple PDF reader
            reader = PDFReader()
            return reader.load_data(file_path)

class DoclingDocxReader:
    """Enhanced DOCX reader using Docling with contextual chunking"""
    
    def __init__(self):
        self.converter = DocumentConverter()
    
    def load_data(self, file_path: str) -> List[Document]:
        """Load DOCX using Docling with contextual chunking"""
        try:
            conv_result = self.converter.convert(file_path)
            doc = conv_result.document
            
            # Use LlamaIndex's SentenceSplitter for contextual chunking
            from llama_index.core.node_parser import SentenceSplitter
            text_splitter = SentenceSplitter(
                chunk_size=512,
                chunk_overlap=50,
                paragraph_separator="\n\n"
            )
            
            # Convert docling doc to text and split
            full_text = doc.export_to_markdown()
            
            # Create chunks with context
            chunks = text_splitter.split_text(full_text)
            
            documents = []
            for i, chunk_text in enumerate(chunks):
                doc = Document(
                    text=chunk_text,
                    metadata={
                        "file_name": os.path.basename(file_path),
                        "source": file_path,
                        "type": "docx",
                        "chunk_id": i,
                        "chunk_type": "contextual",
                        "chunk_size": len(chunk_text)
                    }
                )
                documents.append(doc)
            
            logger.info(f"Created {len(documents)} contextual chunks from {file_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Error processing DOCX {file_path} with Docling: {e}")
            # Fallback to simple DOCX reader
            reader = DocxReader()
            return reader.load_data(file_path)



@st.cache_resource
def initialize_system():
    """Initialize the RAG system with vLLM and Qdrant"""
    try:
        # Use environment variables for API base and key
        api_base = os.environ.get("OPENAI_API_BASE", "http://nginx:80/v1")
        api_key = os.environ.get("OPENAI_API_KEY", "sk-12345")
        # Configure LlamaIndex settings
        # Configure OpenAI client for vLLM with OpenAI-compatible API
        # Use a compatible model name that OpenAI client accepts
        Settings.llm = OpenAI(
            model="gpt-3.5-turbo",
            api_base=api_base,
            api_key=api_key,
            max_tokens=512
        )
        # Use jinaai/jina-embeddings-v4-vllm-retrieval model with vLLM OpenAI-compatible API
        # This connects to the vLLM embedding service running in Docker
        from llama_index.core.embeddings import BaseEmbedding
        import httpx
        from pydantic import Field
        class vLLMEmbedding(BaseEmbedding):
            """Custom embedding class for vLLM with jinaai/jina-embeddings-v4-vllm-retrieval"""
            model_name: str = Field(default="jina-embeddings-v4", description="Model name")
            api_base: str = Field(default_factory=lambda: os.environ.get("OPENAI_API_BASE", "http://nginx:80/v1"), description="API base URL")
            api_key: str = Field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", "sk-12345"), description="API key")
            embed_batch_size: int = Field(default=5, description="Batch size for embeddings")
            def __init__(
                self,
                model_name: str = "jina-embeddings-v4",
                api_base: str = None,
                api_key: str = None,
                embed_batch_size: int = 10,
                **kwargs
            ):
                if api_base is None:
                    api_base = os.environ.get("OPENAI_API_BASE", "http://nginx:80/v1")
                if api_key is None:
                    api_key = os.environ.get("OPENAI_API_KEY", "sk-12345")
                super().__init__(
                    model_name=model_name,
                    api_base=api_base,
                    api_key=api_key,
                    embed_batch_size=embed_batch_size,
                    **kwargs
                )
            
            @property
            def headers(self):
                """Get headers for API requests"""
                return {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
            
            def _get_query_embedding(self, query: str) -> List[float]:
                """Get embedding for a single query"""
                return self._get_text_embeddings([query])[0]
            
            def _get_text_embedding(self, text: str) -> List[float]:
                """Get embedding for a single text"""
                return self._get_text_embeddings([text])[0]
            
            async def _aget_query_embedding(self, query: str) -> List[float]:
                """Get embedding for a single query (async)"""
                embeddings = await self._aget_text_embeddings([query])
                return embeddings[0]
            
            async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
                """Get embeddings for a list of texts (async)"""
                # For now, use the sync version since we're not doing async HTTP calls
                return self._get_text_embeddings(texts)
            
            def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
                """Get embeddings for a list of texts"""
                url = f"{self.api_base}/embeddings"
                
                embeddings = []
                # Process in batches
                for i in range(0, len(texts), self.embed_batch_size):
                    batch = texts[i:i + self.embed_batch_size]
                    
                    payload = {
                        "model": self.model_name,
                        "input": batch,
                        "encoding_format": "float"
                    }
                    
                    try:
                        response = httpx.post(
                            url, 
                            json=payload, 
                            headers=self.headers,
                            timeout=60.0
                        )
                        response.raise_for_status()
                        
                        data = response.json()
                        batch_embeddings = [item["embedding"] for item in data["data"]]
                        embeddings.extend(batch_embeddings)
                        
                    except Exception as e:
                        logger.error(f"Error getting embeddings: {e}")
                        raise Exception(f"Error getting embeddings from vLLM: {e}")
                
                return embeddings
        
        # Use jina-embeddings-v4-vllm-retrieval as the embedding model
        model_name = os.environ.get("EMBEDDING_MODEL", "jina-embeddings-v4")
        Settings.embed_model = vLLMEmbedding(
            model_name=model_name,
            api_base=api_base,
            api_key=api_key,
            embed_batch_size=5,
            vector_type="dense"
        )
        
        # Configure text splitter - will be handled by docling contextual chunking
        Settings.text_splitter = SentenceSplitter(
            chunk_size=1024,
            chunk_overlap=200
        )
        
        # Initialize Qdrant client
        client = qdrant_client.QdrantClient(
            host="qdrant-2",
            port=6333
        )
        
        # Create vector store
        vector_store = QdrantVectorStore(
            client=client,
            collection_name="documents",
            enable_hybrid=True
        )
        logger.info("Created Qdrant vector store with hybrid search enabled")
        
        # Create storage context
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        logger.info("Created storage context")
        
        return client, storage_context
        
    except Exception as e:
        logger.error(f"Error initializing system: {e}")
        st.error(f"Failed to initialize system: {e}")
        return None, None

def check_collection_exists(client, collection_name: str = "documents") -> bool:
    """Check if a Qdrant collection exists"""
    try:
        # Try to get the specific collection
        collection_info = client.get_collection(collection_name)
        logger.info(f"Collection '{collection_name}' exists with status: {collection_info.status}")
        return True
    except Exception as e:
        logger.info(f"Collection '{collection_name}' does not exist: {e}")
        return False

def load_documents(data_dir: str) -> List[Document]:
    """Load documents from the data directory with file hashing to prevent unnecessary reprocessing"""
    documents = []
    data_path = Path(data_dir)
    
    if not data_path.exists():
        data_path.mkdir(parents=True, exist_ok=True)
        return documents
    
    # Create hash storage directory
    hash_dir = data_path / ".hashes"
    hash_dir.mkdir(exist_ok=True)
    hash_file = hash_dir / "file_hashes.json"
    
    # Load existing hashes
    existing_hashes = {}
    if hash_file.exists():
        try:
            with open(hash_file, 'r') as f:
                existing_hashes = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load existing hashes: {e}")
    
    # Calculate current file hashes
    current_hashes = {}
    processed_files = []
    
    # Supported file extensions
    supported_files = []
    supported_files.extend(list(data_path.glob("*.pdf")))
    supported_files.extend(list(data_path.glob("*.docx")))
    supported_files.extend(list(data_path.glob("*.xlsx")))
    
    # Calculate hashes for all supported files
    for file_path in supported_files:
        try:
            # Calculate MD5 hash of file content
            with open(file_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            current_hashes[str(file_path)] = file_hash
        except Exception as e:
            logger.error(f"Error calculating hash for {file_path}: {e}")
            current_hashes[str(file_path)] = "error"
    
    # Track which files need processing
    files_to_process = []
    for file_path, current_hash in current_hashes.items():
        file_path_obj = Path(file_path)
        if str(file_path_obj) not in existing_hashes or existing_hashes[str(file_path_obj)] != current_hash:
            files_to_process.append(file_path_obj)
        else:
            logger.info(f"Skipping unchanged file: {file_path_obj.name}")
    
    # Process only changed or new files
    if not files_to_process:
        logger.info("No new or changed files to process")
        # Load existing documents from vector store if available
        return documents
    
    # Separate files by type for processing
    pdf_files = [f for f in files_to_process if f.suffix.lower() == '.pdf']
    docx_files = [f for f in files_to_process if f.suffix.lower() == '.docx']
    xlsx_files = [f for f in files_to_process if f.suffix.lower() == '.xlsx']
    
    # Load PDFs
    if pdf_files:
        pdf_reader = DoclingPDFReader()
        for pdf_file in pdf_files:
            try:
                docs = pdf_reader.load_data(str(pdf_file))
                documents.extend(docs)
                logger.info(f"Loaded {len(docs)} documents from {pdf_file}")
                processed_files.append(str(pdf_file))
            except Exception as e:
                st.error(f"Error loading PDF {pdf_file}: {e}")
    
    # Load DOCX files
    if docx_files:
        docx_reader = DoclingDocxReader()
        for docx_file in docx_files:
            try:
                docs = docx_reader.load_data(str(docx_file))
                documents.extend(docs)
                logger.info(f"Loaded {len(docs)} documents from {docx_file}")
                processed_files.append(str(docx_file))
            except Exception as e:
                st.error(f"Error loading DOCX {docx_file}: {e}")
    
    # Load Excel files
    if xlsx_files:
        excel_reader = DoclingExcelReader()
        for xlsx_file in xlsx_files:
            try:
                docs = excel_reader.load_data(str(xlsx_file))
                documents.extend(docs)
                logger.info(f"Loaded {len(docs)} contextual chunks from {xlsx_file}")
                processed_files.append(str(xlsx_file))
            except Exception as e:
                st.error(f"Error loading Excel {xlsx_file}: {e}")
    
    # Update hash file with processed files
    if processed_files or existing_hashes != current_hashes:
        # Update existing hashes with current hashes for processed files
        for file_path, current_hash in current_hashes.items():
            if file_path in [str(p) for p in processed_files]:
                existing_hashes[file_path] = current_hash
            elif file_path not in existing_hashes:
                # Add new files that weren't processed (e.g., due to errors)
                existing_hashes[file_path] = current_hash
        
        # Remove hashes for files that no longer exist
        files_to_remove = [f for f in existing_hashes.keys() if not Path(f).exists()]
        for file_to_remove in files_to_remove:
            del existing_hashes[file_to_remove]
        
        try:
            with open(hash_file, 'w') as f:
                json.dump(existing_hashes, f, indent=2)
            logger.info(f"Updated file hashes for {len(processed_files)} processed files")
        except Exception as e:
            logger.error(f"Error saving file hashes: {e}")
    
    return documents

def create_or_load_index(storage_context, documents: List[Document] = None) -> VectorStoreIndex:
    """Create or load existing index"""
    try:
        # Check if collection exists first
        client = storage_context.vector_store.client
        try:
            client.get_collection("documents")
            collection_exists = True
            logger.info("Collection 'documents' exists, attempting to load index...")
        except Exception:
            collection_exists = False
            logger.info("Collection 'documents' does not exist, will create new index...")
        
        if collection_exists:
            # Try to load existing index
            logger.info("Attempting to load existing index...")
            index = VectorStoreIndex.from_vector_store(
                storage_context.vector_store
            )
            logger.info("Successfully loaded existing index")
            return index
        else:
            # Collection doesn't exist, create new index
            if documents and len(documents) > 0:
                logger.info(f"Creating new index with {len(documents)} documents")
                try:
                    logger.info("Starting index creation...")
                    index = VectorStoreIndex.from_documents(
                        documents,
                        storage_context=storage_context
                    )
                    logger.info("Successfully created new index")
                    return index
                except Exception as create_error:
                    logger.error(f"Error creating index: {create_error}")
                    return None
            else:
                logger.info("No documents available to create index")
                return None
                
    except Exception as e:
        logger.info(f"Could not load existing index: {e}")
        # Create new index if documents are provided
        if documents and len(documents) > 0:
            logger.info(f"Creating new index with {len(documents)} documents")
            try:
                logger.info("Starting index creation...")
                index = VectorStoreIndex.from_documents(
                    documents,
                    storage_context=storage_context
                )
                logger.info("Successfully created new index")
                return index
            except Exception as create_error:
                logger.error(f"Error creating index: {create_error}")
                return None
        else:
            logger.info("No documents available to create index")
            return None

def main():
    st.title("📚 RAG Document Assistant")
    st.markdown("Upload documents and ask questions about their content")
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Document Management")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=['pdf', 'docx', 'xlsx'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            if st.button("Process Documents"):
                with st.spinner("Processing documents..."):
                    # Save uploaded files
                    data_dir = "/app/data"
                    os.makedirs(data_dir, exist_ok=True)
                    
                    for uploaded_file in uploaded_files:
                        file_path = os.path.join(data_dir, uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                    
                    # Load and index documents
                    documents = load_documents(data_dir)
                    if documents:
                        client, storage_context = initialize_system()
                        if storage_context:
                            # Store client in session state for collection checks
                            st.session_state.qdrant_client = client
                            
                            index = create_or_load_index(storage_context, documents)
                            if index:
                                st.session_state.index = index
                                st.session_state.initialized = True
                                st.success(f"Successfully indexed {len(documents)} documents!")
                                st.rerun()
                    else:
                        st.error("No documents found to process")
    
    # Initialize system
    if not st.session_state.initialized:
        with st.spinner("Initializing system..."):
            client, storage_context = initialize_system()
            if storage_context:
                # Store client in session state for collection checks
                st.session_state.qdrant_client = client
                
                documents = load_documents("/app/data")
                index = create_or_load_index(storage_context, documents)
                
                if index:
                    st.session_state.index = index
                    st.session_state.initialized = True
                    if documents:
                        st.success(f"Loaded {len(documents)} documents")
                    else:
                        st.info("System ready. Please upload documents using the sidebar.")
                else:
                    st.session_state.initialized = True  # Mark as initialized even without index
                    st.info("No documents found. Please upload documents using the sidebar.")
            else:
                st.error("Failed to initialize system. Please check the logs.")
    
    # Main interface
    if st.session_state.initialized:
        # Check if collection exists before allowing queries
        collection_exists = False
        if hasattr(st.session_state, 'qdrant_client'):
            collection_exists = check_collection_exists(st.session_state.qdrant_client)
        
        # Create three columns: main content, knowledge base sidebar, and chat history
        col_main, col_kb, col_chat = st.columns([3, 1, 1])
        
        with col_kb:
            st.header("📊 Knowledge Base")
            
            # Display documents in knowledge base
            if collection_exists and hasattr(st.session_state, 'qdrant_client'):
                try:
                    client = st.session_state.qdrant_client
                    collection_info = client.get_collection("documents")
                    
                    st.metric("Total Documents", collection_info.points_count)
                    
                    # Get document metadata
                    try:
                        scroll_result = client.scroll(
                            collection_name="documents",
                            limit=100,
                            with_payload=True
                        )
                        
                        documents = {}
                        for point in scroll_result[0]:
                            file_name = point.payload.get('file_name', 'Unknown')
                            if file_name not in documents:
                                documents[file_name] = {
                                    'count': 0,
                                    'type': point.payload.get('type', 'document'),
                                    'source': point.payload.get('source', 'unknown')
                                }
                            documents[file_name]['count'] += 1
                        
                        if documents:
                            st.subheader("📄 Documents")
                            for file_name, info in documents.items():
                                with st.expander(f"📋 {file_name}"):
                                    st.write(f"**Type:** {info['type']}")
                                    st.write(f"**Chunks:** {info['count']}")
                        else:
                            st.info("No documents found in knowledge base")
                            
                    except Exception as e:
                        st.error(f"Error loading documents: {e}")
                
                except Exception as e:
                    st.error(f"Error accessing collection: {e}")
            else:
                st.info("📁 Upload documents to populate knowledge base")
        
        with col_chat:
            # Chat History Panel
            st.header("📝 Chat History")
            
            # Collapsible chat history
            if 'chat_history' in st.session_state and st.session_state.chat_history:
                with st.expander("💬 View Conversations", expanded=True):
                    for idx, chat in enumerate(reversed(st.session_state.chat_history)):
                        st.markdown(f"**Q{len(st.session_state.chat_history)-idx}:** {chat['question']}")
                        st.markdown(f"**A{len(st.session_state.chat_history)-idx}:** {chat['answer'][:200]}..." if len(chat['answer']) > 200 else f"**A{len(st.session_state.chat_history)-idx}:** {chat['answer']}")
                        st.divider()
                
                if st.button("🗑️ Clear History", key="clear_history"):
                    st.session_state.chat_history = []
                    st.rerun()
            else:
                st.info("No conversation history yet")
        
        with col_main:
            # Debug logging
            logger.info(f"Session state initialized: {st.session_state.initialized}")
            logger.info(f"Has index: {hasattr(st.session_state, 'index')}")
            logger.info(f"Index exists: {st.session_state.index if hasattr(st.session_state, 'index') else 'No index'}")
            logger.info(f"Collection exists: {collection_exists}")
            
            if hasattr(st.session_state, 'index') and st.session_state.index and collection_exists:
                logger.info("Showing query interface")
                st.header("💬 Ask Questions")
                
                # Query input
                query = st.text_input("Enter your question:", placeholder="Ask about the documents...")
                
                if query:
                    with st.spinner("Searching for answers..."):
                        try:
                            # Create query engine with hybrid search and reranking
                            reranker = SentenceTransformerRerank(
                                model="cross-encoder/ms-marco-MiniLM-L-6-v2",
                                top_n=3
                            )
                            
                            query_engine = st.session_state.index.as_query_engine(
                                vector_store_query_mode="hybrid",
                                alpha=0.5,  # Balance between keyword and vector search
                                similarity_top_k=10,  # Retrieve more initially for reranking
                                node_postprocessors=[reranker]
                            )
                            
                            # Execute query
                            response = query_engine.query(query)
                            
                            # Display answer
                            st.markdown("### Answer:")
                            st.markdown(response.response)
                            
                            # Display source documents
                            with st.expander("📄 View Source Documents"):
                                for node in response.source_nodes:
                                    col1, col2 = st.columns([3, 1])
                                    with col1:
                                        st.markdown(f"**{node.metadata.get('file_name', 'Unknown')}**")
                                        st.markdown(node.metadata.get('sheet_name', ''))
                                        st.text(node.text[:500] + "..." if len(node.text) > 500 else node.text)
                                    with col2:
                                        st.caption(f"Score: {node.score:.3f}")
                            
                            # Add to chat history
                            st.session_state.chat_history.append({
                                "question": query,
                                "answer": response.response,
                                "timestamp": time.strftime("%H:%M:%S")
                            })
                            
                        except Exception as e:
                            st.error(f"Error processing query: {e}")
            else:
                logger.info("Not showing query interface - conditions not met")
                st.info("📁 Please upload documents to get started.")
    
    elif not st.session_state.initialized:
        st.info("🚀 System is initializing... Please wait.")

if __name__ == "__main__":
    main()