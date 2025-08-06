"""
Simple test for embeddings functionality
"""
import logging
from pathlib import Path
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_embeddings():
    """Test embeddings generation and storage"""
    logger.info("Testing embeddings functionality")
    
    try:
        from phase1_data_processor import Phase1DataProcessor
        from phase2_embedding_indexer import Phase2EmbeddingIndexer
        
        # Phase 1: Process a single document
        processor = Phase1DataProcessor()
        data_dir = Path("arxiv_papers")
        
        if not data_dir.exists():
            logger.error("arxiv_papers directory not found")
            return False
        
        files = list(data_dir.glob("*.pdf"))
        if not files:
            logger.error("No PDF files found")
            return False
        
        # Process first document
        test_file = files[0]
        logger.info(f"Testing with: {test_file.name}")
        
        chunks = processor.process_document(test_file)
        logger.info(f"Created {len(chunks)} chunks")
        
        # Create DataFrame
        df = pd.DataFrame(chunks)
        logger.info(f"DataFrame shape: {df.shape}")
        
        # Phase 2: Test embeddings
        indexer = Phase2EmbeddingIndexer()
        
        # Test model loading
        logger.info("Testing model loading...")
        indexer.load_semantic_model()
        logger.info("✅ Model loaded successfully")
        
        # Test embedding generation
        logger.info("Testing embedding generation...")
        texts = df['text'].tolist()[:5]  # Test with first 5 chunks
        embeddings = indexer.generate_semantic_embeddings(texts)
        logger.info(f"✅ Generated embeddings: {embeddings.shape}")
        
        # Test FAISS index
        logger.info("Testing FAISS index...")
        indexer.build_faiss_index(embeddings)
        logger.info(f"✅ FAISS index built: {indexer.faiss_index.ntotal} vectors")
        
        # Test TF-IDF
        logger.info("Testing TF-IDF...")
        indexer.build_tfidf_index(texts)
        logger.info(f"✅ TF-IDF built: {indexer.tfidf_matrix.shape}")
        
        # Test saving
        logger.info("Testing save functionality...")
        indexer.documents = df.head(5)  # Use first 5 documents
        indexer.semantic_embeddings = embeddings
        indexer.save_indexes()
        logger.info("✅ Indexes saved successfully")
        
        # Test loading
        logger.info("Testing load functionality...")
        new_indexer = Phase2EmbeddingIndexer()
        success = new_indexer.load_indexes()
        if success:
            logger.info("✅ Indexes loaded successfully")
            logger.info(f"Loaded: {new_indexer.faiss_index.ntotal} vectors")
        else:
            logger.error("❌ Failed to load indexes")
            return False
        
        # Test search
        logger.info("Testing search functionality...")
        query = "machine learning"
        results = new_indexer.hybrid_search(query, top_k=3)
        logger.info(f"✅ Search returned {len(results)} results")
        
        logger.info("🎉 All embedding tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Embedding test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_embeddings()
    if success:
        print("✅ Embeddings test completed successfully!")
    else:
        print("❌ Embeddings test failed!") 