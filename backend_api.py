#!/usr/bin/env python3
"""
FastAPI Backend for DocuMind AI Web Interface
Bridges the modern web frontend with the existing document processing system
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import tempfile
import os
import shutil
import logging
import time
import asyncio
from pathlib import Path
import pandas as pd

# Import our existing system components
from phase1_data_processor import Phase1DataProcessor
from phase2_embedding_indexer import Phase2EmbeddingIndexer
from phase3_local_llm import Phase3LocalLLM
from phase4_evaluator import Phase4Evaluator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="DocuMind AI API",
    description="AI-powered document search and summarization system",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (the web interface)
app.mount("/static", StaticFiles(directory="web_interface"), name="static")

# Add favicon route to prevent 404 errors
@app.get("/favicon.ico")
async def favicon():
    from fastapi.responses import Response
    return Response(status_code=204)  # No content

# Request/Response Models
class SearchRequest(BaseModel):
    query: str
    search_type: str = "hybrid"
    top_k: int = 5
    summary_type: str = "medium"

class SearchResponse(BaseModel):
    search_results: List[Dict[str, Any]]
    summary: Dict[str, Any]
    performance: Dict[str, Any]
    error: Optional[str] = None

class StatusResponse(BaseModel):
    initialized: bool
    documents_count: int
    chunks_count: int
    llm_available: bool
    avg_response_time: Optional[str] = None
    avg_accuracy: Optional[str] = None
    processing_complete: bool = True

class SuggestionsResponse(BaseModel):
    suggestions: List[str]

# Global system components
class SystemState:
    def __init__(self):
        self.data_processor = Phase1DataProcessor()
        self.embedding_indexer = Phase2EmbeddingIndexer()
        self.local_llm = Phase3LocalLLM()
        self.evaluator = Phase4Evaluator()
        
        self.is_initialized = False
        self.documents_df = None
        self.query_suggestions = []
        self.performance_data = []
        self.processing_in_progress = False
        
        # Try to load existing indexes
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize system and try to load existing data"""
        try:
            # Check if embeddings exist and load them
            if self.embedding_indexer.indexes_exist():
                logger.info("Loading existing indexes...")
                if self.embedding_indexer.load_indexes():
                    self.is_initialized = True
                    # Try to load existing documents metadata
                    try:
                        import pickle
                        from config import EMBEDDINGS_DIR
                        metadata_file = EMBEDDINGS_DIR / "metadata.pkl"
                        if metadata_file.exists():
                            with open(metadata_file, 'rb') as f:
                                self.documents_df = pickle.load(f)
                            logger.info(f"Loaded {len(self.documents_df)} existing documents")
                            self._update_query_suggestions()
                    except Exception as e:
                        logger.warning(f"Could not load existing documents: {e}")
            else:
                logger.info("No existing indexes found. System ready for new uploads.")
                self.is_initialized = True
                
        except Exception as e:
            logger.error(f"System initialization error: {e}")
            self.is_initialized = False
    
    def _update_query_suggestions(self):
        """Generate query suggestions based on uploaded documents"""
        if self.documents_df is None or len(self.documents_df) == 0:
            self.query_suggestions = [
                "Please upload documents to get suggestions",
                "Upload PDF files to start analyzing",
                "System ready for document upload"
            ]
            return
            
        suggestions = []
        
        # Extract from file names
        unique_files = self.documents_df['file_name'].unique()
        for file_name in unique_files[:5]:
            clean_name = file_name.replace('_', ' ').replace('.pdf', '')
            suggestions.extend([
                f"What is {clean_name} about?",
                f"Summarize {clean_name}",
                f"Key points in {clean_name}"
            ])
        
        # Extract key terms from content
        sample_texts = self.documents_df['text'].head(10).tolist()
        common_terms = []
        
        for text in sample_texts:
            words = text.lower().split()
            # Find potentially interesting terms (capitalized words, technical terms)
            for word in words:
                if (len(word) > 5 and 
                    word.isalpha() and 
                    word not in ['abstract', 'introduction', 'conclusion', 'research', 'paper', 'study']):
                    common_terms.append(word.capitalize())
        
        # Get most common terms
        from collections import Counter
        common_terms = [term for term, count in Counter(common_terms).most_common(10)]
        
        for term in common_terms[:5]:
            suggestions.extend([
                f"What is {term}?",
                f"How does {term} work?",
                f"Applications of {term}"
            ])
        
        # Add generic research questions
        suggestions.extend([
            "What are the main findings?",
            "What methodology was used?",
            "What are the conclusions?",
            "What are the key contributions?",
            "What are the limitations?"
        ])
        
        # Remove duplicates and limit
        self.query_suggestions = list(dict.fromkeys(suggestions))[:25]
        logger.info(f"Generated {len(self.query_suggestions)} query suggestions")

# Initialize global system state
system = SystemState()

# API Endpoints

@app.get("/")
async def read_root():
    """Serve the main web interface"""
    return {"message": "DocuMind AI API", "docs": "/docs", "interface": "/static/index.html"}

@app.get("/api/status", response_model=StatusResponse)
async def get_status():
    """Get system status and statistics"""
    try:
        doc_count = len(system.documents_df) if system.documents_df is not None else 0
        chunk_count = len(system.documents_df) if system.documents_df is not None else 0
        
        # Calculate average metrics from performance data
        avg_response_time = None
        avg_accuracy = None
        
        if system.performance_data:
            avg_response_time = f"{sum(p.get('total_time', 0) for p in system.performance_data) / len(system.performance_data):.2f}s"
            avg_accuracy = f"{sum(p.get('avg_accuracy', 0) for p in system.performance_data) / len(system.performance_data):.1f}"
        
        return StatusResponse(
            initialized=system.is_initialized,
            documents_count=len(system.documents_df['file_name'].unique()) if system.documents_df is not None else 0,
            chunks_count=chunk_count,
            llm_available=system.local_llm.is_available,
            avg_response_time=avg_response_time,
            avg_accuracy=avg_accuracy,
            processing_complete=not system.processing_in_progress
        )
    except Exception as e:
        logger.error(f"Status check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/suggestions", response_model=SuggestionsResponse)
async def get_suggestions():
    """Get query suggestions based on uploaded documents"""
    try:
        return SuggestionsResponse(suggestions=system.query_suggestions)
    except Exception as e:
        logger.error(f"Suggestions error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/languages")
async def get_supported_languages():
    """Get supported languages for translation"""
    try:
        languages = system.local_llm.get_supported_languages()
        return {"languages": languages}
    except Exception as e:
        logger.error(f"Error getting supported languages: {e}")
        return {"languages": {"english": "en-IN"}}

class TranslateRequest(BaseModel):
    text: str
    target_language: str

@app.post("/api/translate")
async def translate_text(request: TranslateRequest):
    """Translate text to target language"""
    try:
        if not request.text.strip():
            return {
                "status": "error",
                "error": "Text is required",
                "original_text": request.text,
                "translated_text": request.text,
                "target_language": request.target_language
            }
        
        if request.target_language.lower() == "english":
            return {
                "status": "success",
                "original_text": request.text,
                "translated_text": request.text,
                "target_language": "english"
            }
        
        # Translate using the translation service directly
        translation_result = system.local_llm.translation_service.translate_text(request.text, request.target_language)
        
        return translation_result
        
    except Exception as e:
        logger.error(f"Translation error: {e}")
        return {
            "status": "error",
            "error": f"Translation service error: {str(e)}",
            "original_text": request.text,
            "translated_text": request.text,
            "target_language": request.target_language
        }

@app.post("/api/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload and process PDF documents"""
    try:
        if system.processing_in_progress:
            raise HTTPException(status_code=400, detail="Processing already in progress")
            
        system.processing_in_progress = True
        
        # Validate files
        pdf_files = [f for f in files if f.content_type == 'application/pdf']
        if not pdf_files:
            raise HTTPException(status_code=400, detail="No valid PDF files provided")
        
        logger.info(f"🔄 UPLOAD-ONLY MODE: Processing {len(pdf_files)} PDF files")
        logger.info("📋 Clearing existing documents and embeddings...")
        
        # CRITICAL: Clear existing documents and embeddings for upload-only mode
        system.documents_df = None
        system.query_suggestions = []
        system.performance_data = []
        
        # Clear embeddings and indexes
        try:
            system.embedding_indexer.clear_indexes()
            logger.info("✅ Cleared existing embeddings and indexes")
        except Exception as e:
            logger.warning(f"Could not clear existing indexes: {e}")
        
        # Process files
        all_chunks = []
        processed_files = []
        
        for file in pdf_files:
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    shutil.copyfileobj(file.file, tmp_file)
                    tmp_path = Path(tmp_file.name)
                
                # Process the PDF
                chunks = system.data_processor.process_document(tmp_path)
                
                # Update file names to original names and add upload metadata
                for chunk in chunks:
                    chunk['file_name'] = file.filename.replace('.pdf', '')
                    chunk['original_filename'] = file.filename
                    chunk['upload_session'] = int(time.time())  # Track upload session
                    chunk['source_type'] = 'uploaded'  # Mark as uploaded
                
                all_chunks.extend(chunks)
                processed_files.append(file.filename)
                
                # Clean up temp file
                os.unlink(tmp_path)
                
                logger.info(f"Processed {file.filename}: {len(chunks)} chunks")
                
            except Exception as e:
                logger.error(f"Error processing {file.filename}: {e}")
                # Continue with other files
                continue
        
        if not all_chunks:
            raise HTTPException(status_code=400, detail="No documents could be processed")
        
        # Create new DataFrame with only uploaded documents
        system.documents_df = pd.DataFrame(all_chunks)
        
        # Build embeddings and indexes (force rebuild for upload-only mode)
        logger.info("Building embeddings and indexes...")
        success = system.embedding_indexer.initialize_indexes(system.documents_df, force_rebuild=True)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to build embeddings")
        
        # Update query suggestions
        system._update_query_suggestions()
        
        system.processing_in_progress = False
        
        return {
            "status": "success",
            "message": "Upload-only mode: Previous documents cleared, new documents processed",
            "processed_files": processed_files,
            "total_chunks": len(all_chunks),
            "documents_count": len(system.documents_df['file_name'].unique()),
            "upload_session": int(time.time()),
            "mode": "upload_only"
        }
        
    except HTTPException:
        system.processing_in_progress = False
        raise
    except Exception as e:
        system.processing_in_progress = False
        logger.error(f"Upload processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """Search documents and generate summary"""
    try:
        if system.documents_df is None or len(system.documents_df) == 0:
            raise HTTPException(status_code=400, detail="No documents uploaded. Please upload PDF documents first.")
        
        start_time = time.time()
        
        # Perform search
        if request.search_type == "semantic":
            search_results = system.embedding_indexer.semantic_search(request.query, request.top_k)
        elif request.search_type == "keyword":
            search_results = system.embedding_indexer.keyword_search(request.query, request.top_k)
        else:  # hybrid
            search_results = system.embedding_indexer.hybrid_search(request.query, request.top_k)
        
        search_time = time.time() - start_time
        
        if not search_results:
            return SearchResponse(
                search_results=[],
                summary={"summary": "No relevant documents found for your query.", "status": "no_results"},
                performance={"search_time": search_time, "total_time": search_time},
                error="No relevant documents found"
            )
        
        # Add accuracy scores
        for i, result in enumerate(search_results):
            # Calculate accuracy score based on similarity and other factors
            base_score = result.get('score', result.get('hybrid_score', result.get('semantic_similarity', 0.5)))
            query_words = set(request.query.lower().split())
            doc_words = set(result.get('text', '').lower().split())
            keyword_overlap = len(query_words.intersection(doc_words)) / len(query_words) if query_words else 0.0
            
            # Combine scores
            accuracy_score = (base_score * 0.7 + keyword_overlap * 0.3)
            search_results[i]['accuracy_score'] = min(max(accuracy_score, 0.0), 1.0)
        
        # Generate summary using RAG (always in English first)
        summary_start = time.time()
        summary_result = system.local_llm.summarize_with_rag(
            request.query, 
            search_results, 
            request.summary_type,
            "english"  # Always generate in English first
        )
        summary_time = time.time() - summary_start
        
        total_time = time.time() - start_time
        
        # Calculate performance metrics
        performance = {
            "query": request.query,
            "search_time": search_time,
            "summary_time": summary_time,
            "total_time": total_time,
            "search_type": request.search_type,
            "summary_type": request.summary_type,
            "num_results": len(search_results),
            "avg_accuracy": sum(r.get('accuracy_score', 0) for r in search_results) / len(search_results) if search_results else 0,
            "max_accuracy": max(r.get('accuracy_score', 0) for r in search_results) if search_results else 0,
            "min_accuracy": min(r.get('accuracy_score', 0) for r in search_results) if search_results else 0
        }
        
        # Store performance data
        system.performance_data.append(performance)
        if len(system.performance_data) > 100:
            system.performance_data = system.performance_data[-100:]
        
        return SearchResponse(
            search_results=search_results,
            summary=summary_result,
            performance=performance
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "system_initialized": system.is_initialized,
        "llm_available": system.local_llm.is_available
    }

# Error handlers
from fastapi.responses import JSONResponse

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found", "path": str(request.url)}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )

if __name__ == "__main__":
    import uvicorn
    
    print("\n🚀 Starting DocuMind AI Backend Server...")
    print("📋 Features:")
    print("  • FastAPI backend with modern web interface")
    print("  • Real-time PDF upload and processing")
    print("  • Advanced search with multiple algorithms")
    print("  • Local LLM integration for summarization")
    print("  • Comprehensive performance analytics")
    print("\n🌐 Access the interface at: http://localhost:8080/static/index.html")
    print("📖 API documentation at: http://localhost:8080/docs")
    print("\n⚠️  Make sure Ollama is running: ollama serve")
    print("🤖 Required model: ollama pull deepseek-r1:8b\n")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8080,
        log_level="info",
        reload=False  # Set to True for development
    ) 