"""
Debug script to test search functionality
"""
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_search():
    """Debug the search functionality"""
    logger.info("Debugging search functionality")
    
    try:
        from phase2_embedding_indexer import Phase2EmbeddingIndexer
        
        # Create indexer
        indexer = Phase2EmbeddingIndexer()
        
        # Load indexes
        logger.info("Loading indexes...")
        success = indexer.load_indexes()
        
        if not success:
            logger.error("Failed to load indexes")
            return False
        
        logger.info("✅ Indexes loaded successfully")
        
        # Test query
        query = "What are the key challenges in deep reinforcement learning?"
        logger.info(f"Testing query: {query}")
        
        # Test different search types
        logger.info("\n=== Testing Semantic Search ===")
        semantic_results = indexer.semantic_search(query, top_k=3)
        logger.info(f"Semantic search returned {len(semantic_results)} results")
        
        if semantic_results:
            logger.info("Sample semantic result:")
            logger.info(f"  File: {semantic_results[0]['file_name']}")
            logger.info(f"  Score: {semantic_results[0]['semantic_similarity']}")
            logger.info(f"  Text: {semantic_results[0]['text'][:100]}...")
        
        logger.info("\n=== Testing Keyword Search ===")
        keyword_results = indexer.keyword_search(query, top_k=3)
        logger.info(f"Keyword search returned {len(keyword_results)} results")
        
        if keyword_results:
            logger.info("Sample keyword result:")
            logger.info(f"  File: {keyword_results[0]['file_name']}")
            logger.info(f"  Score: {keyword_results[0]['keyword_similarity']}")
            logger.info(f"  Text: {keyword_results[0]['text'][:100]}...")
        
        logger.info("\n=== Testing Hybrid Search ===")
        hybrid_results = indexer.hybrid_search(query, top_k=3)
        logger.info(f"Hybrid search returned {len(hybrid_results)} results")
        
        if hybrid_results:
            logger.info("Sample hybrid result:")
            logger.info(f"  File: {hybrid_results[0]['file_name']}")
            logger.info(f"  Hybrid Score: {hybrid_results[0].get('hybrid_score', 'N/A')}")
            logger.info(f"  Text: {hybrid_results[0]['text'][:100]}...")
        
        # Test with a simpler query
        logger.info("\n=== Testing Simple Query ===")
        simple_query = "machine learning"
        simple_results = indexer.hybrid_search(simple_query, top_k=2)
        logger.info(f"Simple query returned {len(simple_results)} results")
        
        if simple_results:
            logger.info("Sample simple result:")
            logger.info(f"  File: {simple_results[0]['file_name']}")
            logger.info(f"  Score: {simple_results[0].get('hybrid_score', 'N/A')}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error in debug search: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = debug_search()
    if success:
        print("✅ Search debug completed!")
    else:
        print("❌ Search debug failed!") 