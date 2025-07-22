import os
import time
import logging
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

class ExcelReader:
    """Custom Excel reader that converts each sheet to markdown"""
    
    def load_data(self, file_path: str) -> List[Document]:
        """Load Excel file and convert each sheet to a document"""
        documents = []
        
        try:
            # Read all sheets
            excel_file = pd.ExcelFile(file_path)
            
            for sheet_name in excel_file.sheet_names:
                # Read sheet into DataFrame
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                
                # Convert to markdown
                markdown_content = f"# Sheet: {sheet_name}\n\n"
                markdown_content += df.to_markdown(index=False)
                
                # Create document
                doc = Document(
                    text=markdown_content,
                    metadata={
                        "file_name": os.path.basename(file_path),
                        "sheet_name": sheet_name,
                        "source": file_path
                    }
                )
                documents.append(doc)
                
        except Exception as e:
            logger.error(f"Error reading Excel file {file_path}: {e}")
            
        return documents

class DoclingPDFReader:
    """Enhanced PDF reader using Docling for complex layouts"""
    
    def __init__(self):
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        self.converter = DocumentConverter()
    
    def load_data(self, file_path: str) -> List[Document]:
        """Load PDF using Docling for better text extraction"""
        try:
            conv_result = self.converter.convert(file_path)
            markdown_content = conv_result.document.export_to_markdown()
            
            doc = Document(
                text=markdown_content,
                metadata={
                    "file_name": os.path.basename(file_path),
                    "source": file_path,
                    "type": "pdf"
                }
            )
            return [doc]
            
        except Exception as e:
            logger.error(f"Error processing PDF {file_path} with Docling: {e}")
            # Fallback to simple PDF reader
            reader = PDFReader()
            return reader.load_data(file_path)

class DoclingDocxReader:
    """Enhanced DOCX reader using Docling"""
    
    def __init__(self):
        self.converter = DocumentConverter()
    
    def load_data(self, file_path: str) -> List[Document]:
        """Load DOCX using Docling"""
        try:
            conv_result = self.converter.convert(file_path)
            markdown_content = conv_result.document.export_to_markdown()
            
            doc = Document(
                text=markdown_content,
                metadata={
                    "file_name": os.path.basename(file_path),
                    "source": file_path,
                    "type": "docx"
                }
            )
            return [doc]
            
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
            vector_type: str = Field(default="dense", description="Vector type: dense or late")
            def __init__(
                self,
                model_name: str = "jina-embeddings-v4",
                api_base: str = None,
                api_key: str = None,
                embed_batch_size: int = 10,
                vector_type: str = "dense",
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
                    vector_type=vector_type,
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
                        "encoding_format": "float",
                        "vector_type": self.vector_type
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
        
        # Configure text splitter
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
    """Load documents from the data directory"""
    documents = []
    data_path = Path(data_dir)
    
    if not data_path.exists():
        data_path.mkdir(parents=True, exist_ok=True)
        return documents
    
    # Supported file extensions
    pdf_files = list(data_path.glob("*.pdf"))
    docx_files = list(data_path.glob("*.docx"))
    xlsx_files = list(data_path.glob("*.xlsx"))
    
    # Load PDFs
    if pdf_files:
        pdf_reader = DoclingPDFReader()
        for pdf_file in pdf_files:
            try:
                docs = pdf_reader.load_data(str(pdf_file))
                documents.extend(docs)
                logger.info(f"Loaded {len(docs)} documents from {pdf_file}")
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
            except Exception as e:
                st.error(f"Error loading DOCX {docx_file}: {e}")
    
    # Load Excel files
    if xlsx_files:
        excel_reader = ExcelReader()
        for xlsx_file in xlsx_files:
            try:
                docs = excel_reader.load_data(str(xlsx_file))
                documents.extend(docs)
                logger.info(f"Loaded {len(docs)} documents from {xlsx_file}")
            except Exception as e:
                st.error(f"Error loading Excel {xlsx_file}: {e}")
    
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
                        # Create query engine with hybrid search
                        query_engine = st.session_state.index.as_query_engine(
                            vector_store_query_mode="hybrid",
                            alpha=0.5,  # Balance between keyword and vector search
                            similarity_top_k=5
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
                        if 'chat_history' not in st.session_state:
                            st.session_state.chat_history = []
                        st.session_state.chat_history.append({
                            "question": query,
                            "answer": response.response
                        })
                        
                    except Exception as e:
                        st.error(f"Error processing query: {e}")
            
            # Display chat history
            if 'chat_history' in st.session_state and st.session_state.chat_history:
                st.header("📝 Chat History")
                for chat in reversed(st.session_state.chat_history):
                    with st.container():
                        st.markdown(f"**Q:** {chat['question']}")
                        st.markdown(f"**A:** {chat['answer']}")
                        st.divider()
        else:
            logger.info("Not showing query interface - conditions not met")
            st.info("📁 Please upload documents to get started.")
    
    elif not st.session_state.initialized:
        st.info("🚀 System is initializing... Please wait.")

if __name__ == "__main__":
    main()