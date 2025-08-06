"""
Phase 5: Interface & Deployment
- Streamlit Interface: Local web interface
- Query Processing: Auto-suggestion, dynamic summary length
- Result Display: Pagination, source attribution
"""
import streamlit as st
import logging
import time
import psutil
from typing import List, Dict, Optional
import pandas as pd

from phase1_data_processor import Phase1DataProcessor
from phase2_embedding_indexer import Phase2EmbeddingIndexer
from phase3_local_llm import Phase3LocalLLM
from phase4_evaluator import Phase4Evaluator
from config import SUMMARY_TYPES, SUMMARY_LENGTHS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Phase5StreamlitApp:
    """Phase 5: Streamlit Interface & Deployment"""
    
    def __init__(self):
        """Initialize the Streamlit application"""
        self.data_processor = Phase1DataProcessor()
        self.embedding_indexer = Phase2EmbeddingIndexer()
        self.local_llm = Phase3LocalLLM()
        self.evaluator = Phase4Evaluator()
        
        self.is_initialized = False
        self.documents_df = None
        
        # Performance tracking
        self.performance_data = []
    
    def initialize_system(self, force_rebuild: bool = False) -> bool:
        """
        Initialize the complete system
        
        Args:
            force_rebuild: Force rebuild all indexes
            
        Returns:
            True if initialization successful
        """
        try:
            with st.spinner("Initializing system..."):
                # First, try to load existing indexes
                if not force_rebuild and self.embedding_indexer.load_indexes():
                    st.success("✅ Loaded existing indexes from disk")
                    self.is_initialized = True
                    return True
                
                # If indexes don't exist or force rebuild, process documents
                st.info("📄 Phase 1: Processing documents...")
                self.documents_df = self.data_processor.process_corpus()
                
                if self.documents_df.empty:
                    st.error("❌ No documents processed successfully")
                    return False
                
                # Phase 2: Embedding & Indexing
                st.info("🔍 Phase 2: Building indexes...")
                success = self.embedding_indexer.initialize_indexes(self.documents_df, force_rebuild=force_rebuild)
                
                if not success:
                    st.error("❌ Failed to build indexes")
                    return False
                
                self.is_initialized = True
                st.success("✅ System initialized successfully")
                return True
                
        except Exception as e:
            st.error(f"❌ Error initializing system: {e}")
            return False
    
    def search_and_summarize(self, query: str, search_type: str = "hybrid", 
                           top_k: int = 5, summary_type: str = "medium") -> Dict:
        """
        Perform search and summarization with performance tracking
        
        Args:
            query: Search query
            search_type: Type of search (semantic/keyword/hybrid)
            top_k: Number of results
            summary_type: Type of summary
            
        Returns:
            Dictionary with results and performance metrics
        """
        if not self.is_initialized:
            return {'error': 'System not initialized'}
        
        start_time = time.time()
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        try:
            # Search
            search_start = time.time()
            if search_type == "semantic":
                search_results = self.embedding_indexer.semantic_search(query, top_k)
            elif search_type == "keyword":
                search_results = self.embedding_indexer.keyword_search(query, top_k)
            else:  # hybrid
                search_results = self.embedding_indexer.hybrid_search(query, top_k)
            
            search_time = time.time() - search_start
            
            # Summarize
            summary_start = time.time()
            summary_result = self.local_llm.summarize_with_rag(query, search_results, summary_type)
            summary_time = time.time() - summary_start
            
            total_time = time.time() - start_time
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            # Record performance
            performance = {
                'query': query,
                'search_time': search_time,
                'summary_time': summary_time,
                'total_time': total_time,
                'memory_usage': final_memory - initial_memory,
                'search_type': search_type,
                'summary_type': summary_type,
                'num_results': len(search_results)
            }
            self.performance_data.append(performance)
            
            return {
                'search_results': search_results,
                'summary': summary_result,
                'performance': performance
            }
            
        except Exception as e:
            st.error(f"Error in search and summarization: {e}")
            return {'error': str(e)}
    
    def get_suggested_queries(self) -> List[str]:
        """Get suggested queries based on document content"""
        if not self.is_initialized or self.documents_df is None:
            return []
        
        suggestions = []
        
        # Use document titles as suggestions
        unique_files = self.documents_df['file_name'].unique()
        for file_name in unique_files[:5]:
            topic = file_name.replace('_', ' ').replace('-', ' ')
            suggestions.extend([
                f"What is {topic}?",
                f"Explain {topic}",
                f"What are the applications of {topic}?"
            ])
        
        # Add general queries
        suggestions.extend([
            "What are the main topics in machine learning?",
            "How do neural networks work?",
            "What are the challenges in deep learning?",
            "Explain quantum computing algorithms",
            "What is the future of AI?"
        ])
        
        return suggestions[:10]  # Limit to 10 suggestions
    
    def run_streamlit_app(self):
        """Run the Streamlit application"""
        st.set_page_config(
            page_title="Document Search & Summarization System",
            page_icon="🔍",
            layout="wide"
        )
        
        st.title("🔍 Document Search & Summarization System")
        st.markdown("Advanced RAG system with local LLM integration")
        
        # Sidebar for system status and controls
        with st.sidebar:
            st.header("System Status")
            
            if not self.is_initialized:
                if st.button("Initialize System", type="primary"):
                    self.initialize_system()
            else:
                st.success("✅ System Initialized")
                
                if st.button("Reinitialize System"):
                    self.initialize_system(force_rebuild=True)
            
            # System statistics
            if self.is_initialized and self.documents_df is not None:
                st.subheader("System Statistics")
                st.metric("Documents", self.documents_df['file_name'].nunique())
                st.metric("Chunks", len(self.documents_df))
                st.metric("Pages", self.documents_df['page_number'].sum())
            
            # LLM status
            if self.is_initialized:
                st.subheader("LLM Status")
                llm_status = self.local_llm.test_connection()
                if llm_status['available']:
                    st.success("✅ Local LLM Available")
                    st.text(f"Model: {llm_status['model']}")
                else:
                    st.error("❌ Local LLM Not Available")
                    st.text("Please ensure Ollama is running")
        
        # Main interface - try to auto-initialize if not already done
        if not self.is_initialized:
            # Try to auto-initialize by loading existing indexes
            if self.embedding_indexer.load_indexes():
                self.is_initialized = True
                st.success("✅ Auto-loaded existing indexes")
            else:
                st.info("Please initialize the system first using the sidebar.")
                return
        
        # Query input section
        st.header("Search & Summarize")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            query = st.text_area(
                "Enter your query:",
                placeholder="Ask a question about the research papers...",
                height=100
            )
        
        with col2:
            search_type = st.selectbox(
                "Search Type:",
                ["hybrid", "semantic", "keyword"],
                help="Hybrid combines semantic and keyword search"
            )
            
            top_k = st.slider("Number of Results:", 1, 10, 5)
            
            summary_type = st.selectbox(
                "Summary Length:",
                SUMMARY_TYPES,
                help="Choose summary length"
            )
        
        # Auto-suggestions
        if st.button("Get Suggestions"):
            suggestions = self.get_suggested_queries()
            st.subheader("Suggested Queries:")
            for i, suggestion in enumerate(suggestions, 1):
                st.write(f"{i}. {suggestion}")
        
        # Search and summarize
        if st.button("Search & Summarize", type="primary") and query.strip():
            with st.spinner("Processing..."):
                result = self.search_and_summarize(query, search_type, top_k, summary_type)
                
                if 'error' in result:
                    st.error(result['error'])
                else:
                    # Display results
                    self._display_results(result)
    
    def _display_results(self, result: Dict):
        """Display search results and summary with pagination"""
        
        # Summary section
        st.header("Generated Summary")
        summary_data = result['summary']
        
        if summary_data['status'] == 'success':
            st.markdown(f"**Summary ({summary_data['summary_type']}):**")
            st.write(summary_data['summary'])
            
            # Summary statistics
            stats = summary_data.get('summary_stats', {})
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Words", stats.get('word_count', 0))
            with col2:
                st.metric("Sentences", stats.get('sentence_count', 0))
            with col3:
                st.metric("Characters", stats.get('character_count', 0))
            with col4:
                st.metric("Generation Time", f"{summary_data.get('generation_time', 0):.2f}s")
        else:
            st.error(f"Summary generation failed: {summary_data.get('error', 'Unknown error')}")
        
        # Search results section
        st.header("Search Results")
        search_results = result['search_results']
        
        if not search_results:
            st.warning("No relevant documents found.")
            return
        
        # Pagination
        results_per_page = 3
        total_pages = (len(search_results) + results_per_page - 1) // results_per_page
        
        if total_pages > 1:
            page = st.selectbox("Page:", range(1, total_pages + 1)) - 1
        else:
            page = 0
        
        start_idx = page * results_per_page
        end_idx = min(start_idx + results_per_page, len(search_results))
        
        # Display results with source attribution
        for i, doc_result in enumerate(search_results[start_idx:end_idx], start_idx + 1):
            with st.expander(f"Result {i}: {doc_result['file_name']} (Page {doc_result.get('page_number', 'N/A')})"):
                # Source attribution
                st.markdown(f"**Source:** {doc_result['file_name']}")
                st.markdown(f"**Page:** {doc_result.get('page_number', 'N/A')}")
                st.markdown(f"**Chunk ID:** {doc_result.get('chunk_id', 'N/A')}")
                
                # Similarity scores
                if 'hybrid_score' in doc_result:
                    st.markdown(f"**Hybrid Score:** {doc_result['hybrid_score']:.3f}")
                elif 'semantic_similarity' in doc_result:
                    st.markdown(f"**Semantic Score:** {doc_result['semantic_similarity']:.3f}")
                elif 'keyword_similarity' in doc_result:
                    st.markdown(f"**Keyword Score:** {doc_result['keyword_similarity']:.3f}")
                
                # Document text
                st.markdown("**Content:**")
                st.text(doc_result['text'][:500] + "..." if len(doc_result['text']) > 500 else doc_result['text'])
        
        # Performance metrics
        if 'performance' in result:
            st.header("Performance Metrics")
            perf = result['performance']
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Search Time", f"{perf['search_time']:.3f}s")
            with col2:
                st.metric("Summary Time", f"{perf['summary_time']:.3f}s")
            with col3:
                st.metric("Total Time", f"{perf['total_time']:.3f}s")
            with col4:
                st.metric("Memory Usage", f"{perf['memory_usage']:.1f}MB")
    
    def run_evaluation(self):
        """Run system evaluation"""
        if not self.is_initialized:
            st.error("System not initialized")
            return
        
        st.header("System Evaluation")
        
        if st.button("Run Evaluation"):
            with st.spinner("Running evaluation..."):
                # Generate synthetic queries
                synthetic_queries = self.evaluator.generate_synthetic_queries(self.documents_df)
                
                # Run evaluation
                search_results = []
                summaries = []
                reference_summaries = []
                
                for query_info in synthetic_queries[:10]:  # Limit for performance
                    # Search
                    result = self.search_and_summarize(query_info['query'])
                    if 'error' not in result:
                        search_results.append(result['search_results'])
                        summaries.append(result['summary']['summary'])
                        
                        # Create reference summary from expected document
                        expected_doc = query_info['expected_document']
                        doc_text = self.documents_df[self.documents_df['file_name'] == expected_doc]['text'].iloc[0]
                        reference_summaries.append(doc_text[:200] + "...")
                
                # Calculate metrics
                retrieval_metrics = self.evaluator.evaluate_retrieval_metrics(
                    search_results, [q['expected_chunks'] for q in synthetic_queries[:10]]
                )
                
                summary_metrics = self.evaluator.evaluate_summary_quality(
                    summaries, reference_summaries
                )
                
                performance_metrics = self.evaluator.evaluate_performance_metrics(
                    self.performance_data
                )
                
                # Generate report
                report = self.evaluator.create_evaluation_report(
                    retrieval_metrics, summary_metrics, performance_metrics
                )
                
                st.subheader("Evaluation Report")
                st.text(report)

def main():
    """Main function to run the Streamlit app"""
    app = Phase5StreamlitApp()
    app.run_streamlit_app()

if __name__ == "__main__":
    main() 