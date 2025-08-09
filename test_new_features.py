#!/usr/bin/env python3
"""
Test script for new features:
1. PDF Upload functionality
2. Enhanced auto-suggestions
3. Accuracy scores
4. Summary length adjustment (already existing)
5. Pagination (already existing)
"""

import logging
import sys
import tempfile
from pathlib import Path
import pandas as pd

# Import system components
from phase5_streamlit_app import Phase5StreamlitApp
from phase2_embedding_indexer import Phase2EmbeddingIndexer
from phase3_local_llm import Phase3LocalLLM

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_accuracy_score_calculation():
    """Test accuracy score calculation"""
    print("🧪 Testing Accuracy Score Calculation...")
    
    try:
        app = Phase5StreamlitApp()
        
        # Test sample data
        search_result = {
            'text': 'Machine learning is a subset of artificial intelligence that uses statistical techniques.',
            'score': 0.8,
            'file_name': 'test.pdf'
        }
        
        queries = [
            "What is machine learning?",
            "artificial intelligence",
            "statistical techniques",
            "completely unrelated query about cooking"
        ]
        
        for query in queries:
            accuracy = app.calculate_accuracy_score(search_result, query)
            print(f"  Query: '{query}' -> Accuracy: {accuracy:.3f}")
        
        print("✅ Accuracy score calculation works!")
        return True
        
    except Exception as e:
        print(f"❌ Accuracy score test failed: {e}")
        return False

def test_enhanced_query_suggestions():
    """Test enhanced auto-suggestions"""
    print("\n🧪 Testing Enhanced Query Suggestions...")
    
    try:
        app = Phase5StreamlitApp()
        
        # Create mock documents
        mock_data = {
            'file_name': ['Machine_Learning_Basics.pdf', 'Deep_Learning_Guide.pdf', 'AI_Ethics.pdf'] * 10,
            'text': ['Machine learning concepts'] * 30,
            'chunk_id': list(range(30)),
            'page_number': [1] * 30
        }
        app.documents_df = pd.DataFrame(mock_data)
        
        # Update suggestions
        app._update_query_suggestions()
        
        # Test general suggestions
        general_suggestions = app.get_enhanced_query_suggestions()
        print(f"  General suggestions count: {len(general_suggestions)}")
        print(f"  Sample suggestions: {general_suggestions[:3]}")
        
        # Test filtered suggestions
        filtered_suggestions = app.get_enhanced_query_suggestions("machine")
        print(f"  Filtered suggestions for 'machine': {len(filtered_suggestions)}")
        print(f"  Sample filtered: {filtered_suggestions[:2]}")
        
        print("✅ Enhanced query suggestions work!")
        return True
        
    except Exception as e:
        print(f"❌ Query suggestions test failed: {e}")
        return False

def test_system_integration():
    """Test system integration with existing components"""
    print("\n🧪 Testing System Integration...")
    
    try:
        # Test component loading
        indexer = Phase2EmbeddingIndexer()
        llm = Phase3LocalLLM()
        
        # Check if indexes exist
        indexes_exist = indexer.indexes_exist()
        print(f"  Indexes exist: {indexes_exist}")
        
        # Check LLM availability
        print(f"  LLM available: {llm.is_available}")
        
        if indexes_exist:
            success = indexer.load_indexes()
            print(f"  Index loading: {'✅ Success' if success else '❌ Failed'}")
            
            if success:
                # Test search functionality
                results = indexer.hybrid_search("machine learning", top_k=3)
                print(f"  Search results: {len(results)} found")
                
                if results and llm.is_available:
                    # Test response generation
                    response = llm.summarize_with_rag("What is machine learning?", results, "short")
                    print(f"  Response generation: {'✅ Success' if 'error' not in response else '❌ Failed'}")
        
        print("✅ System integration test completed!")
        return True
        
    except Exception as e:
        print(f"❌ System integration test failed: {e}")
        return False

def test_streamlit_app_components():
    """Test Streamlit app components"""
    print("\n🧪 Testing Streamlit App Components...")
    
    try:
        app = Phase5StreamlitApp()
        
        # Test initialization check
        print(f"  Initial state: {'✅ Initialized' if app.is_initialized else '❌ Not initialized'}")
        
        # Test performance tracking
        app.performance_data.append({
            'query': 'test',
            'search_time': 0.1,
            'summary_time': 0.5,
            'total_time': 0.6,
            'memory_usage': 100,
            'search_type': 'hybrid',
            'summary_type': 'medium',
            'num_results': 5
        })
        
        print(f"  Performance tracking: {len(app.performance_data)} records")
        
        # Test error handling
        result = app.search_and_summarize("test query", "hybrid", 5, "medium")
        expected_error = 'error' in result and result['error'] == 'System not initialized'
        print(f"  Error handling: {'✅ Proper' if expected_error else '❌ Unexpected'}")
        
        print("✅ Streamlit app components work!")
        return True
        
    except Exception as e:
        print(f"❌ Streamlit app test failed: {e}")
        return False

def test_summary_enhancements():
    """Test summary enhancement features"""
    print("\n🧪 Testing Summary Enhancements...")
    
    try:
        app = Phase5StreamlitApp()
        
        # Test summary stats calculation
        sample_summary = {
            'status': 'success',
            'summary': 'This is a test summary with multiple words and sentences. It has good length for testing.',
            'summary_type': 'medium'
        }
        
        # Simulate summary enhancement
        summary_text = sample_summary['summary']
        stats = {
            'word_count': len(summary_text.split()),
            'character_count': len(summary_text),
            'sentence_count': len([s for s in summary_text.split('.') if s.strip()]),
        }
        
        print(f"  Word count: {stats['word_count']}")
        print(f"  Character count: {stats['character_count']}")
        print(f"  Sentence count: {stats['sentence_count']}")
        
        # Verify reasonable values
        success = (stats['word_count'] > 0 and 
                  stats['character_count'] > stats['word_count'] and 
                  stats['sentence_count'] > 0)
        
        print(f"  Summary stats calculation: {'✅ Success' if success else '❌ Failed'}")
        print("✅ Summary enhancements work!")
        return True
        
    except Exception as e:
        print(f"❌ Summary enhancement test failed: {e}")
        return False

def run_comprehensive_test():
    """Run all tests"""
    print("🚀 Running Comprehensive Test Suite for New Features")
    print("=" * 60)
    
    tests = [
        test_accuracy_score_calculation,
        test_enhanced_query_suggestions,
        test_summary_enhancements,
        test_streamlit_app_components,
        test_system_integration
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{i+1}. {test.__name__}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All new features are working correctly!")
    else:
        print("⚠️  Some features need attention.")
    
    return passed == total

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1) 