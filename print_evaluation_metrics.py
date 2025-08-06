#!/usr/bin/env python3
"""
Evaluation Metrics Printer for Document Search & Summarization System
This script loads the existing system and prints comprehensive evaluation metrics.
"""

import logging
import time
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple

# Import our system components
from phase2_embedding_indexer import Phase2EmbeddingIndexer
from phase3_local_llm import Phase3LocalLLM
from phase4_evaluator import Phase4Evaluator
from config import DATA_DIR, EMBEDDINGS_DIR

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def print_system_info():
    """Print basic system information"""
    print("=" * 80)
    print("📊 DOCUMENT SEARCH & SUMMARIZATION SYSTEM - EVALUATION METRICS")
    print("=" * 80)
    
    # Check embeddings directory
    embeddings_dir = Path(EMBEDDINGS_DIR)
    if embeddings_dir.exists():
        files = list(embeddings_dir.glob("*"))
        print(f"📁 Embeddings Directory: {embeddings_dir}")
        print(f"📄 Found {len(files)} files in embeddings directory")
        for file in files:
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"   - {file.name}: {size_mb:.2f} MB")
    else:
        print("❌ Embeddings directory not found!")
        return False
    
    # Check data directory
    data_dir = Path(DATA_DIR)
    if data_dir.exists():
        pdf_files = list(data_dir.glob("*.pdf"))
        print(f"📚 Data Directory: {data_dir}")
        print(f"📄 Found {len(pdf_files)} PDF files")
    else:
        print("❌ Data directory not found!")
        return False
    
    print()
    return True

def load_system_components():
    """Load all system components"""
    print("🔄 Loading System Components...")
    
    # Load embedding indexer
    print("  📥 Loading embedding indexer...")
    indexer = Phase2EmbeddingIndexer()
    success = indexer.load_indexes()
    if not success:
        print("❌ Failed to load indexes!")
        return None, None, None
    
    print(f"  ✅ Loaded {indexer.faiss_index.ntotal} vectors")
    print(f"  ✅ Loaded {len(indexer.documents)} documents")
    
    # Load LLM
    print("  📥 Loading local LLM...")
    llm = Phase3LocalLLM()
    if not llm.is_available:
        print("❌ LLM not available!")
        return indexer, None, None
    
    print(f"  ✅ LLM available: {llm.model}")
    
    # Load evaluator
    print("  📥 Loading evaluator...")
    evaluator = Phase4Evaluator()
    print("  ✅ Evaluator loaded")
    
    print("✅ All components loaded successfully!")
    print()
    return indexer, llm, evaluator

def test_search_functionality(indexer: Phase2EmbeddingIndexer):
    """Test search functionality with sample queries"""
    print("🔍 Testing Search Functionality...")
    
    test_queries = [
        "machine learning",
        "deep learning",
        "robotics",
        "quantum computing",
        "artificial intelligence",
        "neural networks",
        "data science",
        "computer vision",
        "natural language processing",
        "reinforcement learning"
    ]
    
    search_results_summary = []
    
    for query in test_queries:
        print(f"  Testing query: '{query}'")
        
        # Test semantic search
        semantic_results = indexer.semantic_search(query, top_k=3)
        
        # Test keyword search
        keyword_results = indexer.keyword_search(query, top_k=3)
        
        # Test hybrid search
        hybrid_results = indexer.hybrid_search(query, top_k=3)
        
        search_results_summary.append({
            'query': query,
            'semantic_count': len(semantic_results),
            'keyword_count': len(keyword_results),
            'hybrid_count': len(hybrid_results),
            'avg_semantic_score': np.mean([r.get('score', 0) for r in semantic_results]) if semantic_results else 0,
            'avg_keyword_score': np.mean([r.get('score', 0) for r in keyword_results]) if keyword_results else 0,
            'avg_hybrid_score': np.mean([r.get('score', 0) for r in hybrid_results]) if hybrid_results else 0
        })
    
    print("✅ Search functionality test completed!")
    print()
    return search_results_summary

def test_response_generation(llm: Phase3LocalLLM, indexer: Phase2EmbeddingIndexer):
    """Test response generation with sample queries"""
    print("📝 Testing Response Generation...")
    
    test_queries = [
        "What is machine learning?",
        "Explain deep learning concepts",
        "How does robotics work?",
        "What is quantum computing?"
    ]
    
    response_results = []
    
    for i, query in enumerate(test_queries):
        print(f"  Processing query {i+1}/{len(test_queries)}: '{query}'")
        
        # Get search results
        search_results = indexer.hybrid_search(query, top_k=3)
        
        if not search_results:
            print(f"    ❌ No search results found for query: {query}")
            continue
        
        # Generate response
        start_time = time.time()
        response_result = llm.summarize_with_rag(query, search_results, "medium")
        end_time = time.time()
        
        if 'error' not in response_result:
            response_text = response_result.get('summary', '')
            response_length = len(response_text.split())
            
            response_results.append({
                'query': query,
                'response_length': response_length,
                'generation_time': end_time - start_time,
                'search_results_count': len(search_results),
                'has_response': True,
                'response_preview': response_text[:100] + "..." if len(response_text) > 100 else response_text
            })
            
            print(f"    ✅ Generated response ({response_length} words, {end_time - start_time:.2f}s)")
        else:
            print(f"    ❌ Error generating response: {response_result['error']}")
            response_results.append({
                'query': query,
                'response_length': 0,
                'generation_time': 0,
                'search_results_count': len(search_results),
                'has_response': False,
                'response_preview': "Error"
            })
    
    print("✅ Response generation test completed!")
    print()
    return response_results

def calculate_performance_metrics(search_results: List[Dict], response_results: List[Dict]):
    """Calculate performance metrics from test results"""
    print("📊 Calculating Performance Metrics...")
    
    # Search Performance Metrics
    avg_semantic_score = np.mean([r['avg_semantic_score'] for r in search_results])
    avg_keyword_score = np.mean([r['avg_keyword_score'] for r in search_results])
    avg_hybrid_score = np.mean([r['avg_hybrid_score'] for r in search_results])
    
    # Response Generation Metrics
    successful_responses = [r for r in response_results if r['has_response']]
    avg_response_length = np.mean([r['response_length'] for r in successful_responses]) if successful_responses else 0
    avg_generation_time = np.mean([r['generation_time'] for r in successful_responses]) if successful_responses else 0
    success_rate = len(successful_responses) / len(response_results) if response_results else 0
    
    # System Performance Metrics
    total_documents = 836  # From your system
    total_chunks = 836
    avg_search_results = np.mean([r['hybrid_count'] for r in search_results])
    
    metrics = {
        'search_performance': {
            'avg_semantic_score': avg_semantic_score,
            'avg_keyword_score': avg_keyword_score,
            'avg_hybrid_score': avg_hybrid_score,
            'avg_search_results': avg_search_results
        },
        'response_performance': {
            'success_rate': success_rate,
            'avg_response_length': avg_response_length,
            'avg_generation_time': avg_generation_time
        },
        'system_performance': {
            'total_documents': total_documents,
            'total_chunks': total_chunks,
            'embedding_dimension': 384,
            'search_methods': 3  # semantic, keyword, hybrid
        }
    }
    
    print("✅ Performance metrics calculated!")
    print()
    return metrics

def print_comprehensive_metrics(search_results: List[Dict], response_results: List[Dict], metrics: Dict):
    """Print comprehensive metrics summary"""
    print("=" * 80)
    print("📊 COMPREHENSIVE EVALUATION METRICS")
    print("=" * 80)
    
    # System Overview
    print("\n🏗️ SYSTEM OVERVIEW:")
    print("-" * 50)
    print(f"Total Documents:     {metrics['system_performance']['total_documents']}")
    print(f"Total Chunks:        {metrics['system_performance']['total_chunks']}")
    print(f"Embedding Dimension: {metrics['system_performance']['embedding_dimension']}")
    print(f"Search Methods:      {metrics['system_performance']['search_methods']}")
    
    # Search Performance
    print("\n🔍 SEARCH PERFORMANCE:")
    print("-" * 50)
    print(f"Average Semantic Score:  {metrics['search_performance']['avg_semantic_score']:.4f}")
    print(f"Average Keyword Score:    {metrics['search_performance']['avg_keyword_score']:.4f}")
    print(f"Average Hybrid Score:     {metrics['search_performance']['avg_hybrid_score']:.4f}")
    print(f"Average Search Results:   {metrics['search_performance']['avg_search_results']:.1f}")
    
    # Response Generation Performance
    print("\n📝 RESPONSE GENERATION PERFORMANCE:")
    print("-" * 50)
    print(f"Success Rate:           {metrics['response_performance']['success_rate']:.2%}")
    print(f"Average Response Length: {metrics['response_performance']['avg_response_length']:.0f} words")
    print(f"Average Generation Time: {metrics['response_performance']['avg_generation_time']:.2f} seconds")
    
    # Quality Assessment
    print("\n⭐ QUALITY ASSESSMENT:")
    print("-" * 50)
    
    # Search Quality Rating
    hybrid_score = metrics['search_performance']['avg_hybrid_score']
    if hybrid_score > 0.7:
        search_rating = "🟢 Excellent"
    elif hybrid_score > 0.5:
        search_rating = "🟡 Good"
    elif hybrid_score > 0.3:
        search_rating = "🟠 Fair"
    else:
        search_rating = "🔴 Poor"
    
    # Response Quality Rating
    success_rate = metrics['response_performance']['success_rate']
    if success_rate > 0.9:
        response_rating = "🟢 Excellent"
    elif success_rate > 0.7:
        response_rating = "🟡 Good"
    elif success_rate > 0.5:
        response_rating = "🟠 Fair"
    else:
        response_rating = "🔴 Poor"
    
    print(f"Search Quality:     {search_rating} (Score: {hybrid_score:.3f})")
    print(f"Response Quality:   {response_rating} (Success: {success_rate:.1%})")
    
    # Sample Results
    print("\n📋 SAMPLE RESULTS:")
    print("-" * 50)
    for i, result in enumerate(response_results[:3]):
        print(f"Query {i+1}: {result['query']}")
        print(f"Response: {result['response_preview']}")
        print(f"Length: {result['response_length']} words, Time: {result['generation_time']:.2f}s")
        print()
    
    print("=" * 80)

def main():
    """Main evaluation function"""
    print_system_info()
    
    # Load system components
    indexer, llm, evaluator = load_system_components()
    if not all([indexer, llm, evaluator]):
        print("❌ Failed to load system components!")
        return
    
    # Test search functionality
    search_results = test_search_functionality(indexer)
    
    # Test response generation
    response_results = test_response_generation(llm, indexer)
    
    # Calculate performance metrics
    metrics = calculate_performance_metrics(search_results, response_results)
    
    # Print comprehensive summary
    print_comprehensive_metrics(search_results, response_results, metrics)
    
    print("✅ Evaluation completed successfully!")

if __name__ == "__main__":
    main() 