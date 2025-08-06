"""
Test script to verify embeddings are properly loaded from disk
"""
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_embeddings_loading():
    """Test that embeddings are properly loaded from disk"""
    logger.info("Testing embeddings loading from disk")
    
    try:
        from phase2_embedding_indexer import Phase2EmbeddingIndexer
        
        # Create indexer
        indexer = Phase2EmbeddingIndexer()
        
        # Check if files exist
        logger.info("Checking if embedding files exist...")
        required_files = [
            indexer.embeddings_file,
            indexer.faiss_index_file,
            indexer.tfidf_file,
            indexer.tfidf_vectorizer_file,
            indexer.metadata_file
        ]
        
        for file_path in required_files:
            if file_path.exists():
                logger.info(f"✅ {file_path.name} exists")
            else:
                logger.error(f"❌ {file_path.name} missing")
                return False
        
        # Try to load indexes
        logger.info("Loading indexes from disk...")
        success = indexer.load_indexes()
        
        if success:
            logger.info("✅ Successfully loaded indexes from disk")
            logger.info(f"  - FAISS index: {indexer.faiss_index.ntotal} vectors")
            logger.info(f"  - TF-IDF matrix: {indexer.tfidf_matrix.shape}")
            logger.info(f"  - Documents: {len(indexer.documents)}")
            logger.info(f"  - Embeddings shape: {indexer.semantic_embeddings.shape}")
            
            # Test search functionality
            logger.info("Testing search functionality...")
            query = "machine learning"
            results = indexer.hybrid_search(query, top_k=3)
            logger.info(f"✅ Search returned {len(results)} results")
            
            if results:
                logger.info("Sample result:")
                logger.info(f"  - File: {results[0]['file_name']}")
                logger.info(f"  - Score: {results[0].get('hybrid_score', 'N/A')}")
                logger.info(f"  - Text preview: {results[0]['text'][:100]}...")
            
            return True
        else:
            logger.error("❌ Failed to load indexes")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error testing embeddings loading: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_embeddings_loading()
    if success:
        print("✅ Embeddings loading test passed!")
    else:
        print("❌ Embeddings loading test failed!") 