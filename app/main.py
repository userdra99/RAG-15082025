import os
import time
import logging
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for, session
from werkzeug.utils import secure_filename

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

CUSTOM_MODEL_CONFIG = {
    "unsloth/Llama-3.2-3B-Instruct": {
        "context_window": 8192,
        "is_chat_model": True,
        "is_function_calling_model": False,
        "is_vision_model": False,
    }
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['UPLOAD_FOLDER'] = '/app/data'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'docx', 'xlsx'}

# Global variables for RAG system
rag_system = {
    'initialized': False,
    'index': None,
    'client': None,
    'storage_context': None
}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

class DoclingExcelReader:
    """Enhanced Excel reader using Docling with contextual chunking"""
    
    def load_data(self, file_path: str) -> List[Document]:
        """Load Excel file and create contextual chunks"""
        documents = []
        
        try:
            from llama_index.core.node_parser import SentenceSplitter
            text_splitter = SentenceSplitter(
                chunk_size=512,
                chunk_overlap=50,
                paragraph_separator="\n\n"
            )
            
            excel_file = pd.ExcelFile(file_path)
            
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                
                markdown_content = f"# Sheet: {sheet_name}\n\n"
                markdown_content += df.to_markdown(index=False)
                
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
            
            from llama_index.core.node_parser import SentenceSplitter
            text_splitter = SentenceSplitter(
                chunk_size=512,
                chunk_overlap=50,
                paragraph_separator="\n\n"
            )
            
            full_text = doc.export_to_markdown()
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
            
            from llama_index.core.node_parser import SentenceSplitter
            text_splitter = SentenceSplitter(
                chunk_size=512,
                chunk_overlap=50,
                paragraph_separator="\n\n"
            )
            
            full_text = doc.export_to_markdown()
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
            reader = DocxReader()
            return reader.load_data(file_path)

def get_embedding_dimensions(model_name: str) -> int:
    """Get embedding dimensions based on model name"""
    model_dimensions = {
        "nomic-embed-text-v1": 768,
        "nomic-ai/nomic-embed-text-v1": 768,
        "BAAI/bge-m3": 1024,
        "bge-m3": 1024
    }
    return model_dimensions.get(model_name, 768)  # Default to 768 if unknown

def initialize_system():
    """Initialize the RAG system with vLLM and Qdrant"""
    try:
        # Separate endpoints for LLM and embedding services
        llm_api_base = os.environ.get("LLM_API_BASE", "http://vllm-llm:8000/v1")
        embedding_api_base = os.environ.get("EMBEDDING_API_BASE", "http://vllm-embedding:8000/v1")
        api_key = os.environ.get("OPENAI_API_KEY", "sk-12345")
        
        # Use OpenAI-like client for vLLM endpoints
        from llama_index.llms.openai_like import OpenAILike
        
        # Support multiple LLM models
        llm_model = os.environ.get("LLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
        
        Settings.llm = OpenAILike(
            model=llm_model,
            api_base=llm_api_base,
            api_key=api_key,
            max_new_tokens=1024,  # Increased for better responses
            is_chat_model=True,
            temperature=0.7,
            top_p=0.8
        )
        
        # Use simple approach: just use the working OpenAI embedding and test directly
        from llama_index.embeddings.openai import OpenAIEmbedding
        
        model_name = os.environ.get("EMBEDDING_MODEL", "nomic-embed-text-v1")
        embedding_dimensions = get_embedding_dimensions(model_name)
        
        logger.info(f"Using embedding model: {model_name} with {embedding_dimensions} dimensions")
        
        # Test if we can connect directly to BGE-M3
        import requests
        test_payload = {
            "input": ["test"],
            "model": model_name
        }
        
        test_response = requests.post(
            f"{embedding_api_base}/embeddings",
            json=test_payload,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10
        )
        
        if test_response.status_code == 200:
            logger.info(f"✅ BGE-M3 endpoint test successful")
            
            # Create custom embedding class that works with BGE-M3
            from llama_index.core.embeddings import BaseEmbedding
            
            class BGE_M3_Embedding(BaseEmbedding):
                api_base: str
                api_key: str
                model_name: str
                
                def __init__(self, api_base: str, api_key: str, model_name: str):
                    super().__init__(
                        api_base=api_base,
                        api_key=api_key,
                        model_name=model_name
                    )
                    
                def _get_embedding(self, text: str) -> list:
                    """Get embedding for a single text"""
                    payload = {
                        "input": [text],
                        "model": self.model_name
                    }
                    
                    response = requests.post(
                        f"{self.api_base}/embeddings",
                        json=payload,
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        embeddings = data.get('data', [])
                        if embeddings:
                            return embeddings[0].get('embedding', [])
                    
                    raise Exception(f"Failed to get embedding: {response.status_code} - {response.text}")
                
                def _get_text_embedding(self, text: str) -> list:
                    """Interface method for single text embedding"""
                    return self._get_embedding(text)
                
                def _get_text_embeddings(self, texts: list) -> list:
                    """Interface method for batch text embeddings"""
                    embeddings = []
                    for text in texts:
                        embeddings.append(self._get_embedding(text))
                    return embeddings
                
                def _get_query_embedding(self, query: str) -> list:
                    """Interface method for query embedding"""
                    return self._get_embedding(query)
                
                async def _aget_query_embedding(self, query: str) -> list:
                    """Async query embedding - fallback to sync"""
                    return self._get_query_embedding(query)
                
                async def _aget_text_embedding(self, text: str) -> list:
                    """Async text embedding - fallback to sync"""
                    return self._get_text_embedding(text)
                
                async def _aget_text_embeddings(self, texts: list) -> list:
                    """Async batch text embeddings - fallback to sync"""
                    return self._get_text_embeddings(texts)
            
            # Use custom BGE-M3 embedding
            Settings.embed_model = BGE_M3_Embedding(
                api_base=embedding_api_base,
                api_key=api_key,
                model_name=model_name
            )
            logger.info(f"✅ Successfully configured custom BGE-M3 embedding")
            
        else:
            logger.error(f"❌ BGE-M3 endpoint test failed: {test_response.status_code} - {test_response.text}")
            
            # Fallback to OpenAI embedding with ada-002
            Settings.embed_model = OpenAIEmbedding(
                model="text-embedding-ada-002",
                api_key=api_key,
                api_base=embedding_api_base,
                embed_batch_size=10
            )
        
        Settings.text_splitter = SentenceSplitter(
            chunk_size=1024,
            chunk_overlap=200
        )
        
        client = qdrant_client.QdrantClient(
            host="qdrant",
            port=6333
        )
        
        # Get embedding model and dimensions
        model_name = os.environ.get("EMBEDDING_MODEL", "nomic-embed-text-v1")
        embedding_dimensions = get_embedding_dimensions(model_name)
        
        # Create collection if it doesn't exist with appropriate dimensions
        collection_name = "documents"
        try:
            collection_info = client.get_collection(collection_name)
            logger.info(f"Collection '{collection_name}' already exists")
            
            # Check if dimensions match current model
            current_vector_size = collection_info.config.params.vectors.size
            if current_vector_size != embedding_dimensions:
                logger.warning(f"Collection vector size ({current_vector_size}) doesn't match model dimensions ({embedding_dimensions})")
                logger.warning("Consider creating a new collection or migrating data")
        except Exception:
            from qdrant_client.models import Distance, VectorParams
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=embedding_dimensions, distance=Distance.COSINE)
            )
            logger.info(f"Created new collection '{collection_name}' with {embedding_dimensions}-dimensional vectors for model {model_name}")
        
        vector_store = QdrantVectorStore(
            client=client,
            collection_name="documents"
        )
        logger.info("Created Qdrant vector store")
        
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        logger.info("Created storage context")
        
        return client, storage_context
        
    except Exception as e:
        logger.error(f"Error initializing system: {e}")
        return None, None

def check_collection_exists(client, collection_name: str = "documents") -> bool:
    """Check if a Qdrant collection exists"""
    try:
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
    
    hash_dir = data_path / ".hashes"
    hash_dir.mkdir(exist_ok=True)
    hash_file = hash_dir / "file_hashes.json"
    
    existing_hashes = {}
    if hash_file.exists():
        try:
            with open(hash_file, 'r') as f:
                existing_hashes = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load existing hashes: {e}")
    
    current_hashes = {}
    processed_files = []
    
    supported_files = []
    supported_files.extend(list(data_path.glob("*.pdf")))
    supported_files.extend(list(data_path.glob("*.docx")))
    supported_files.extend(list(data_path.glob("*.xlsx")))
    
    for file_path in supported_files:
        try:
            with open(file_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            current_hashes[str(file_path)] = file_hash
        except Exception as e:
            logger.error(f"Error calculating hash for {file_path}: {e}")
            current_hashes[str(file_path)] = "error"
    
    files_to_process = []
    for file_path, current_hash in current_hashes.items():
        file_path_obj = Path(file_path)
        if str(file_path_obj) not in existing_hashes or existing_hashes[str(file_path_obj)] != current_hash:
            files_to_process.append(file_path_obj)
        else:
            logger.info(f"Skipping unchanged file: {file_path_obj.name}")
    
    if not files_to_process:
        logger.info("No new or changed files to process")
        return documents
    
    pdf_files = [f for f in files_to_process if f.suffix.lower() == '.pdf']
    docx_files = [f for f in files_to_process if f.suffix.lower() == '.docx']
    xlsx_files = [f for f in files_to_process if f.suffix.lower() == '.xlsx']
    
    if pdf_files:
        pdf_reader = DoclingPDFReader()
        for pdf_file in pdf_files:
            try:
                docs = pdf_reader.load_data(str(pdf_file))
                documents.extend(docs)
                logger.info(f"Loaded {len(docs)} documents from {pdf_file}")
                processed_files.append(str(pdf_file))
            except Exception as e:
                logger.error(f"Error loading PDF {pdf_file}: {e}")
    
    if docx_files:
        docx_reader = DoclingDocxReader()
        for docx_file in docx_files:
            try:
                docs = docx_reader.load_data(str(docx_file))
                documents.extend(docs)
                logger.info(f"Loaded {len(docs)} documents from {docx_file}")
                processed_files.append(str(docx_file))
            except Exception as e:
                logger.error(f"Error loading DOCX {docx_file}: {e}")
    
    if xlsx_files:
        excel_reader = DoclingExcelReader()
        for xlsx_file in xlsx_files:
            try:
                docs = excel_reader.load_data(str(xlsx_file))
                documents.extend(docs)
                logger.info(f"Loaded {len(docs)} contextual chunks from {xlsx_file}")
                processed_files.append(str(xlsx_file))
            except Exception as e:
                logger.error(f"Error loading Excel {xlsx_file}: {e}")
    
    if processed_files or existing_hashes != current_hashes:
        for file_path, current_hash in current_hashes.items():
            if file_path in [str(p) for p in processed_files]:
                existing_hashes[file_path] = current_hash
            elif file_path not in existing_hashes:
                existing_hashes[file_path] = current_hash
        
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

def create_or_load_index(storage_context, documents: List[Document] = None, mode: str = "incremental") -> VectorStoreIndex:
    """Create, load, or update existing index
    
    Args:
        storage_context: Storage context for the index
        documents: New documents to add (if any)
        mode: "incremental" to add to existing index, "replace" to create new index
    """
    try:
        # Check if collection exists and has documents
        collection_exists = check_collection_exists(rag_system['client'])
        
        if collection_exists and mode == "incremental":
            try:
                collection_info = rag_system['client'].get_collection("documents")
                existing_count = collection_info.points_count
                
                # Load existing index
                if existing_count > 0:
                    logger.info(f"Loading existing index with {existing_count} documents...")
                    vector_store = QdrantVectorStore(
                        client=rag_system['client'],
                        collection_name="documents"
                    )
                    index = VectorStoreIndex.from_vector_store(vector_store)
                    logger.info("Successfully loaded existing index")
                    
                    # Add new documents incrementally if provided
                    if documents and len(documents) > 0:
                        logger.info(f"Adding {len(documents)} new documents to existing index...")
                        for doc in documents:
                            index.insert(doc)
                        logger.info(f"Successfully added {len(documents)} documents. Total now: {existing_count + len(documents)}")
                    
                    return index
                else:
                    logger.info("Collection exists but is empty, will create new index")
            except Exception as e:
                logger.error(f"Could not load/update existing index: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Create new index if no collection exists, mode is "replace", or incremental failed
        if documents and len(documents) > 0:
            logger.info(f"Creating new index with {len(documents)} documents")
            try:
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
            # If no documents and no existing index, create empty index
            if not collection_exists:
                logger.info("Creating empty index for future document additions")
                index = VectorStoreIndex.from_documents(
                    [],
                    storage_context=storage_context
                )
                return index
            else:
                logger.info("No new documents to add")
                return None
                
    except Exception as e:
        logger.error(f"Unexpected error in create_or_load_index: {e}")
        return None

@app.route('/')
def index():
    """Main page"""
    collection_exists = False
    documents = {}
    total_docs = 0
    
    if rag_system['initialized'] and rag_system['client']:
        collection_exists = check_collection_exists(rag_system['client'])
        
        if collection_exists:
            try:
                collection_info = rag_system['client'].get_collection("documents")
                total_docs = collection_info.points_count
                
                scroll_result = rag_system['client'].scroll(
                    collection_name="documents",
                    limit=100,
                    with_payload=True
                )
                
                for point in scroll_result[0]:
                    file_name = point.payload.get('file_name', 'Unknown')
                    if file_name not in documents:
                        documents[file_name] = {
                            'count': 0,
                            'type': point.payload.get('type', 'document'),
                            'source': point.payload.get('source', 'unknown')
                        }
                    documents[file_name]['count'] += 1
                    
            except Exception as e:
                logger.error(f"Error accessing collection: {e}")
    
    return render_template('index.html', 
                         initialized=rag_system['initialized'],
                         collection_exists=collection_exists,
                         documents=documents,
                         total_docs=total_docs,
                         chat_history=session.get('chat_history', []))

@app.route('/initialize', methods=['POST'])
def initialize():
    """Initialize the RAG system"""
    if not rag_system['initialized']:
        client, storage_context = initialize_system()
        if storage_context:
            rag_system['client'] = client
            rag_system['storage_context'] = storage_context
            
            documents = load_documents(app.config['UPLOAD_FOLDER'])
            index = create_or_load_index(storage_context, documents)
            
            if index:
                rag_system['index'] = index
                rag_system['initialized'] = True
                message = f"System initialized with {len(documents)} documents" if documents else "System initialized"
                return jsonify({'success': True, 'message': message})
            else:
                rag_system['initialized'] = True
                return jsonify({'success': True, 'message': 'System initialized without index. Please upload documents.'})
        else:
            return jsonify({'success': False, 'error': 'Failed to initialize system'}), 500
    
    return jsonify({'success': True, 'message': 'System already initialized'})

@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle file uploads - saves files immediately without processing"""
    if 'files' not in request.files:
        return jsonify({'success': False, 'error': 'No files provided'}), 400
    
    files = request.files.getlist('files')
    uploaded_files = []
    
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            uploaded_files.append(filename)
        else:
            logger.warning(f"Skipped invalid file: {file.filename if file else 'unknown'}")
    
    if uploaded_files:
        return jsonify({
            'success': True, 
            'message': f'Successfully uploaded {len(uploaded_files)} files. Use the process endpoint to index them.',
            'uploaded_files': uploaded_files
        })
    
    return jsonify({'success': False, 'error': 'No valid files uploaded'}), 400

@app.route('/process', methods=['POST'])
def process_documents():
    """Process uploaded documents and create/update index"""
    if not rag_system['initialized']:
        return jsonify({'success': False, 'error': 'System not initialized'}), 400
    
    try:
        # Process documents in the upload folder
        documents = load_documents(app.config['UPLOAD_FOLDER'])
        
        if documents and rag_system['storage_context']:
            index = create_or_load_index(rag_system['storage_context'], documents)
            if index:
                rag_system['index'] = index
                return jsonify({
                    'success': True, 
                    'message': f'Successfully processed and indexed {len(documents)} document chunks'
                })
            else:
                return jsonify({'success': False, 'error': 'Failed to create index from documents'}), 500
        else:
            return jsonify({'success': False, 'error': 'No documents found to process'}), 400
            
    except Exception as e:
        logger.error(f"Error processing documents: {e}")
        return jsonify({'success': False, 'error': f'Processing failed: {str(e)}'}), 500

@app.route('/query', methods=['POST'])
def query():
    """Handle search queries"""
    logger.info(f"Request method: {request.method}")
    logger.info(f"Content-Type: {request.content_type}")
    logger.info(f"Raw data: {request.get_data()}")
    
    try:
        data = request.get_json()
        logger.info(f"Parsed JSON: {data}")
    except Exception as e:
        logger.error(f"JSON parsing error: {e}")
        return jsonify({'success': False, 'error': 'Invalid JSON data'}), 400
    
    if data is None:
        logger.error("No JSON data received")
        return jsonify({'success': False, 'error': 'No JSON data provided'}), 400
    
    query_text = data.get('query', '')
    logger.info(f"Query text: '{query_text}'")
    
    if not query_text:
        return jsonify({'success': False, 'error': 'No query provided'}), 400
    
    if not rag_system['initialized'] or not rag_system['index']:
        return jsonify({'success': False, 'error': 'System not initialized or no documents indexed'}), 400
    
    try:
        # Simplified query engine without reranker to test core functionality
        query_engine = rag_system['index'].as_query_engine(
            similarity_top_k=5
        )
        
        response = query_engine.query(query_text)
        
        sources = []
        for node in response.source_nodes:
            sources.append({
                'file_name': node.metadata.get('file_name', 'Unknown'),
                'sheet_name': node.metadata.get('sheet_name', ''),
                'text': node.text[:500] + "..." if len(node.text) > 500 else node.text,
                'score': float(node.score) if node.score else 0
            })
        
        # Update session chat history
        if 'chat_history' not in session:
            session['chat_history'] = []
        
        session['chat_history'].append({
            'question': query_text,
            'answer': response.response,
            'timestamp': time.strftime("%H:%M:%S")
        })
        session.modified = True
        
        return jsonify({
            'success': True,
            'answer': response.response,
            'sources': sources
        })
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/clear_history', methods=['POST'])
def clear_history():
    """Clear chat history"""
    session['chat_history'] = []
    session.modified = True
    return jsonify({'success': True})

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'initialized': rag_system['initialized'],
        'has_index': rag_system['index'] is not None
    })

if __name__ == '__main__':
    # Initialize system on startup without processing documents
    client, storage_context = initialize_system()
    if storage_context:
        rag_system['client'] = client
        rag_system['storage_context'] = storage_context
        
        # Try to load existing index without processing new documents
        try:
            index = create_or_load_index(storage_context, [])
            if index:
                rag_system['index'] = index
                rag_system['initialized'] = True
                logger.info("System initialized with existing index")
            else:
                rag_system['initialized'] = True
                logger.info("System initialized without index - ready for document upload")
        except Exception as e:
            logger.info(f"No existing index found: {e}")
            rag_system['initialized'] = True
            logger.info("System initialized without index - ready for document upload")
    
    app.run(host='0.0.0.0', port=5000, debug=False)