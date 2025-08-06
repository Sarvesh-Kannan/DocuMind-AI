"""
Test the complete search and summarization pipeline
"""
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_complete_pipeline():
    """Test the complete search and summarization pipeline"""
    logger.info("Testing complete pipeline")
    
    try:
        from phase2_embedding_indexer import Phase2EmbeddingIndexer
        from phase3_local_llm import Phase3LocalLLM
        
        # Initialize components
        logger.info("Initializing components...")
        indexer = Phase2EmbeddingIndexer()
        llm = Phase3LocalLLM()
        
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
        
        # Step 1: Search
        logger.info("Step 1: Performing search...")
        search_start = time.time()
        search_results = indexer.hybrid_search(query, top_k=3)
        search_time = time.time() - search_start
        
        logger.info(f"✅ Search completed in {search_time:.2f}s")
        logger.info(f"Found {len(search_results)} results")
        
        if not search_results:
            logger.error("No search results found")
            return False
        
        # Display search results
        for i, result in enumerate(search_results):
            logger.info(f"Result {i+1}:")
            logger.info(f"  File: {result['file_name']}")
            logger.info(f"  Score: {result.get('hybrid_score', 'N/A')}")
            logger.info(f"  Text: {result['text'][:150]}...")
        
        # Step 2: Summarization
        logger.info("Step 2: Performing summarization...")
        summary_start = time.time()
        
        try:
            summary_result = llm.summarize_with_rag(query, search_results, "medium")
            summary_time = time.time() - summary_start
            
            logger.info(f"✅ Summarization completed in {summary_time:.2f}s")
            
            if 'error' in summary_result:
                logger.error(f"Summarization error: {summary_result['error']}")
                return False
            
            # Display summary
            logger.info("Generated Summary:")
            logger.info("=" * 50)
            logger.info(summary_result.get('summary', 'No summary generated'))
            logger.info("=" * 50)
            
            # Display statistics
            if 'statistics' in summary_result:
                stats = summary_result['statistics']
                logger.info(f"Summary statistics:")
                logger.info(f"  Words: {stats.get('word_count', 'N/A')}")
                logger.info(f"  Sentences: {stats.get('sentence_count', 'N/A')}")
                logger.info(f"  Characters: {stats.get('char_count', 'N/A')}")
            
            return True
            
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    except Exception as e:
        logger.error(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_complete_pipeline()
    if success:
        print("✅ Complete pipeline test passed!")
    else:
        print("❌ Complete pipeline test failed!") 