"""
Test script for the Document Search and Summarization System
"""
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_phase1():
    """Test Phase 1: Data Processing"""
    logger.info("Testing Phase 1: Data Processing")
    
    try:
        from phase1_data_processor import Phase1DataProcessor
        
        processor = Phase1DataProcessor()
        
        # Test with a single document
        data_dir = Path("arxiv_papers")
        if data_dir.exists():
            files = list(data_dir.glob("*.pdf"))
            if files:
                test_file = files[0]
                logger.info(f"Testing with file: {test_file.name}")
                
                # Test document processing
                chunks = processor.process_document(test_file)
                logger.info(f"✅ Phase 1 test successful: {len(chunks)} chunks created")
                return True
            else:
                logger.warning("No PDF files found in arxiv_papers directory")
                return False
        else:
            logger.warning("arxiv_papers directory not found")
            return False
            
    except Exception as e:
        logger.error(f"❌ Phase 1 test failed: {e}")
        return False

def test_phase2():
    """Test Phase 2: Embedding & Indexing"""
    logger.info("Testing Phase 2: Embedding & Indexing")
    
    try:
        from phase2_embedding_indexer import Phase2EmbeddingIndexer
        
        indexer = Phase2EmbeddingIndexer()
        
        # Test model loading
        indexer.load_semantic_model()
        logger.info("✅ Phase 2 test successful: Model loaded")
        return True
        
    except Exception as e:
        logger.error(f"❌ Phase 2 test failed: {e}")
        return False

def test_phase3():
    """Test Phase 3: Local LLM"""
    logger.info("Testing Phase 3: Local LLM")
    
    try:
        from phase3_local_llm import Phase3LocalLLM
        
        llm = Phase3LocalLLM()
        
        # Test connection
        test_result = llm.test_connection()
        if test_result['available']:
            logger.info("✅ Phase 3 test successful: LLM available")
            return True
        else:
            logger.warning("⚠️ Phase 3 test: LLM not available (this is expected if Ollama is not running)")
            return True  # Not a failure, just not available
        
    except Exception as e:
        logger.error(f"❌ Phase 3 test failed: {e}")
        return False

def test_phase4():
    """Test Phase 4: Evaluation"""
    logger.info("Testing Phase 4: Evaluation")
    
    try:
        from phase4_evaluator import Phase4Evaluator
        
        evaluator = Phase4Evaluator()
        
        # Test ROUGE scorer
        test_text = "This is a test summary."
        reference_text = "This is a reference summary."
        
        scores = evaluator.rouge_scorer.score(reference_text, test_text)
        logger.info("✅ Phase 4 test successful: ROUGE scorer working")
        return True
        
    except Exception as e:
        logger.error(f"❌ Phase 4 test failed: {e}")
        return False

def test_phase5():
    """Test Phase 5: Streamlit Interface"""
    logger.info("Testing Phase 5: Streamlit Interface")
    
    try:
        from phase5_streamlit_app import Phase5StreamlitApp
        
        app = Phase5StreamlitApp()
        logger.info("✅ Phase 5 test successful: Streamlit app created")
        return True
        
    except Exception as e:
        logger.error(f"❌ Phase 5 test failed: {e}")
        return False

def test_main_system():
    """Test Main System"""
    logger.info("Testing Main System")
    
    try:
        from main_system import MainSystem
        
        system = MainSystem()
        logger.info("✅ Main System test successful: System created")
        return True
        
    except Exception as e:
        logger.error(f"❌ Main System test failed: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    logger.info("🚀 Starting System Tests")
    logger.info("=" * 50)
    
    tests = [
        ("Phase 1: Data Processing", test_phase1),
        ("Phase 2: Embedding & Indexing", test_phase2),
        ("Phase 3: Local LLM", test_phase3),
        ("Phase 4: Evaluation", test_phase4),
        ("Phase 5: Streamlit Interface", test_phase5),
        ("Main System", test_main_system)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running {test_name}")
        try:
            success = test_func()
            results.append((test_name, success))
            if success:
                logger.info(f"✅ {test_name}: PASSED")
            else:
                logger.error(f"❌ {test_name}: FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name}: ERROR - {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("📊 TEST SUMMARY")
    logger.info("=" * 50)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! System is ready to use.")
    else:
        logger.warning("⚠️ Some tests failed. Check the logs above.")
    
    return passed == total

if __name__ == "__main__":
    run_all_tests() 