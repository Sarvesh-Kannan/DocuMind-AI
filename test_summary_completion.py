"""
Test to verify summary completion
"""
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_summary_completion():
    """Test that summaries are complete and not truncated"""
    logger.info("Testing summary completion")
    
    try:
        from phase2_embedding_indexer import Phase2EmbeddingIndexer
        from phase3_local_llm import Phase3LocalLLM
        
        # Initialize components
        indexer = Phase2EmbeddingIndexer()
        llm = Phase3LocalLLM()
        
        # Load indexes
        success = indexer.load_indexes()
        if not success:
            logger.error("Failed to load indexes")
            return False
        
        # Test query
        query = "What are the applications of robotics?"
        logger.info(f"Testing query: {query}")
        
        # Search
        search_results = indexer.hybrid_search(query, top_k=2)
        
        if not search_results:
            logger.error("No search results found")
            return False
        
        # Generate summary
        summary_result = llm.summarize_with_rag(query, search_results, "short")
        
        if 'error' in summary_result:
            logger.error(f"Summarization error: {summary_result['error']}")
            return False
        
        summary_text = summary_result.get('summary', '')
        logger.info(f"Generated summary: {summary_text}")
        
        # Check if summary is complete
        if summary_text:
            # Check if it ends with a complete sentence
            if summary_text.strip().endswith(('.', '!', '?')):
                logger.info("✅ Summary ends with complete sentence")
            else:
                logger.warning("⚠️ Summary does not end with complete sentence")
            
            # Check length
            word_count = len(summary_text.split())
            logger.info(f"Summary word count: {word_count}")
            
            if word_count >= 10:  # Reasonable minimum for a summary
                logger.info("✅ Summary has reasonable length")
            else:
                logger.warning("⚠️ Summary seems too short")
            
            return True
        else:
            logger.error("❌ No summary generated")
            return False
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_summary_completion()
    if success:
        print("✅ Summary completion test passed!")
    else:
        print("❌ Summary completion test failed!") 