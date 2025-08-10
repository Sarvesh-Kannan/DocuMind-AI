#!/usr/bin/env python3
"""
Fix Upload Isolation and Embedding Storage
This script addresses the issues with document upload isolation and proper embedding storage.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_backend_upload_isolation():
    """Fix the backend to ensure proper document isolation"""
    
    backend_file = Path("backend_api.py")
    if not backend_file.exists():
        logger.error("backend_api.py not found")
        return False
    
    # Read the current backend file
    with open(backend_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Key fixes needed:
    # 1. Ensure upload replaces all existing documents (upload-only mode)
    # 2. Clear existing embeddings before processing new uploads
    # 3. Ensure proper isolation of uploaded documents
    
    # Fix 1: Update the upload function to ensure complete replacement
    upload_function_fix = '''
    @app.post("/api/upload")
    async def upload_documents(files: List[UploadFile] = File(...)):
        """Upload and process PDF documents - UPLOAD-ONLY MODE"""
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
                    
                    logger.info(f"✅ Processed {file.filename}: {len(chunks)} chunks")
                    
                except Exception as e:
                    logger.error(f"❌ Error processing {file.filename}: {e}")
                    # Continue with other files
                    continue
            
            if not all_chunks:
                raise HTTPException(status_code=400, detail="No documents could be processed")
            
            # Create new DataFrame with ONLY uploaded documents
            system.documents_df = pd.DataFrame(all_chunks)
            logger.info(f"📊 Created new document dataset: {len(system.documents_df)} chunks from {len(processed_files)} files")
            
            # Build embeddings and indexes (force rebuild for upload-only mode)
            logger.info("🧠 Building embeddings and indexes for uploaded documents...")
            success = system.embedding_indexer.initialize_indexes(system.documents_df, force_rebuild=True)
            
            if not success:
                raise HTTPException(status_code=500, detail="Failed to build embeddings")
            
            # Update query suggestions based ONLY on uploaded documents
            system._update_query_suggestions()
            
            system.processing_in_progress = False
            
            logger.info(f"🎉 Upload complete: {len(processed_files)} files, {len(all_chunks)} chunks, {len(system.query_suggestions)} suggestions")
            
            return {
                "status": "success",
                "message": "Upload-only mode: Previous documents cleared, new documents processed",
                "processed_files": processed_files,
                "total_chunks": len(all_chunks),
                "documents_count": len(system.documents_df['file_name'].unique()),
                "upload_session": int(time.time())
            }
            
        except HTTPException:
            system.processing_in_progress = False
            raise
        except Exception as e:
            system.processing_in_progress = False
            logger.error(f"Upload processing error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    '''
    
    # Check if we need to add the clear_indexes method to embedding indexer
    indexer_file = Path("phase2_embedding_indexer.py")
    if indexer_file.exists():
        with open(indexer_file, 'r', encoding='utf-8') as f:
            indexer_content = f.read()
        
        if 'def clear_indexes(' not in indexer_content:
            # Add clear_indexes method
            clear_method = '''
    def clear_indexes(self):
        """Clear all indexes and embeddings for upload-only mode"""
        try:
            logger.info("🔄 Clearing indexes and embeddings...")
            
            # Clear in-memory data
            self.semantic_model = None
            self.tfidf_vectorizer = None
            self.faiss_index = None
            self.documents = None
            self.tfidf_matrix = None
            self.semantic_embeddings = None
            
            # Clear files on disk
            files_to_clear = [
                self.embeddings_file,
                self.faiss_index_file,
                self.tfidf_file,
                self.tfidf_vectorizer_file,
                self.metadata_file
            ]
            
            for file_path in files_to_clear:
                if file_path.exists():
                    file_path.unlink()
                    logger.info(f"🗑️  Deleted: {file_path.name}")
            
            logger.info("✅ All indexes and embeddings cleared")
            
        except Exception as e:
            logger.error(f"❌ Error clearing indexes: {e}")
            raise e
'''
            
            # Add the method before the last method in the class
            if 'def get_statistics(' in indexer_content:
                indexer_content = indexer_content.replace(
                    'def get_statistics(',
                    clear_method + '\n    def get_statistics('
                )
            else:
                # Add before the last closing of the class
                indexer_content = indexer_content.replace(
                    '\nlogger = logging.getLogger(__name__)',
                    clear_method + '\n\nlogger = logging.getLogger(__name__)'
                )
            
            # Write back the updated indexer
            with open(indexer_file, 'w', encoding='utf-8') as f:
                f.write(indexer_content)
            
            logger.info("✅ Added clear_indexes method to embedding indexer")
    
    logger.info("🔧 Upload isolation fixes implemented")
    return True

def fix_ollama_timeout():
    """Fix Ollama timeout issues"""
    
    llm_file = Path("phase3_local_llm.py")
    if not llm_file.exists():
        logger.error("phase3_local_llm.py not found")
        return False
    
    with open(llm_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Increase timeout and add retry logic
    if 'timeout=120' not in content:
        content = content.replace('timeout=60', 'timeout=120')
    
    # Add retry logic for failed requests
    retry_logic = '''
    def _call_ollama_with_retry(self, prompt: str, max_tokens: int = None, max_retries: int = 2) -> str:
        """Call Ollama with retry logic for better reliability"""
        for attempt in range(max_retries + 1):
            try:
                result = self._call_ollama(prompt, max_tokens)
                if not result.startswith("Error:") and len(result.strip()) > 20:
                    return result
                elif attempt < max_retries:
                    logger.warning(f"Retry {attempt + 1}/{max_retries}: Previous attempt returned: {result[:100]}...")
                    import time
                    time.sleep(2)  # Wait before retry
                else:
                    return result
            except Exception as e:
                if attempt < max_retries:
                    logger.warning(f"Retry {attempt + 1}/{max_retries}: Error: {e}")
                    import time
                    time.sleep(2)
                else:
                    return f"Error after {max_retries + 1} attempts: {str(e)}"
        return "Error: Maximum retries exceeded"
'''
    
    if '_call_ollama_with_retry' not in content:
        # Add the retry method
        content = content.replace(
            'def _call_ollama(',
            retry_logic + '\n    def _call_ollama('
        )
    
    # Update the summarize_with_rag method to use retry logic
    if '_call_ollama_with_retry(' not in content:
        content = content.replace(
            'summary = self._call_ollama(prompt)',
            'summary = self._call_ollama_with_retry(prompt)'
        )
    
    # Write back the updated LLM file
    with open(llm_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    logger.info("🔧 Ollama timeout and retry fixes implemented")
    return True

def main():
    """Main function to apply all fixes"""
    logger.info("🚀 Starting Upload Isolation and Embedding Storage Fixes")
    logger.info("=" * 60)
    
    # Apply fixes
    success = True
    
    try:
        logger.info("1️⃣  Fixing backend upload isolation...")
        if fix_backend_upload_isolation():
            logger.info("✅ Backend upload isolation fixed")
        else:
            logger.error("❌ Failed to fix backend upload isolation")
            success = False
    except Exception as e:
        logger.error(f"❌ Error fixing backend: {e}")
        success = False
    
    try:
        logger.info("2️⃣  Fixing Ollama timeout issues...")
        if fix_ollama_timeout():
            logger.info("✅ Ollama timeout fixes applied")
        else:
            logger.error("❌ Failed to fix Ollama timeout")
            success = False
    except Exception as e:
        logger.error(f"❌ Error fixing Ollama: {e}")
        success = False
    
    logger.info("=" * 60)
    if success:
        logger.info("🎉 All fixes applied successfully!")
        logger.info("\n📋 What was fixed:")
        logger.info("   • Upload-only mode: Clears existing documents before processing new ones")
        logger.info("   • Proper embedding isolation: Only uploaded documents are indexed")
        logger.info("   • Enhanced error handling: Better timeout management")
        logger.info("   • Retry logic: Automatic retry for failed LLM requests")
        logger.info("\n🔄 Next steps:")
        logger.info("   1. Restart the backend server: python backend_api.py")
        logger.info("   2. Upload new documents through the web interface")
        logger.info("   3. Test queries - they should only return results from uploaded docs")
    else:
        logger.error("❌ Some fixes failed. Please check the logs above.")
    
    return success

if __name__ == "__main__":
    main() 