"""
Phase 5: Interface & Deployment
- Streamlit Interface: Local web interface
- Query Processing: Auto-suggestion, dynamic summary length
- Result Display: Pagination, source attribution
- PDF Upload: Upload and process new PDFs
"""
import streamlit as st
import logging
import time
import psutil
from typing import List, Dict, Optional
import pandas as pd
import tempfile
import os
from pathlib import Path

from phase1_data_processor import Phase1DataProcessor
from phase2_embedding_indexer import Phase2EmbeddingIndexer
from phase3_local_llm import Phase3LocalLLM
from phase4_evaluator import Phase4Evaluator
from config import SUMMARY_TYPES, SUMMARY_LENGTHS, DATA_DIR

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Phase5StreamlitApp:
    """Phase 5: Streamlit Interface with Enhanced Features"""
    
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
        
        # Query suggestions cache
        self.query_suggestions = []
    
    def upload_and_process_pdfs(self, uploaded_files) -> bool:
        """
        Upload and process new PDF files
        
        Args:
            uploaded_files: List of uploaded files from Streamlit
            
        Returns:
            True if processing successful
        """
        if not uploaded_files:
            return False
        
        try:
            processed_files = []
            
            with st.spinner(f"Processing {len(uploaded_files)} uploaded PDF(s)..."):
                for uploaded_file in uploaded_files:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = Path(tmp_file.name)
                    
                    try:
                        # Process the uploaded PDF
                        chunks = self.data_processor.process_document(tmp_path)
                        if chunks:
                            # Create DataFrame for new chunks
                            new_df = pd.DataFrame(chunks)
                            
                            # Update file metadata to use original filename
                            new_df['file_name'] = uploaded_file.name
                            new_df['file_path'] = uploaded_file.name
                            
                            # Add to existing documents
                            if self.documents_df is not None:
                                self.documents_df = pd.concat([self.documents_df, new_df], ignore_index=True)
                            else:
                                self.documents_df = new_df
                            
                            processed_files.append(uploaded_file.name)
                            logger.info(f"Processed uploaded file: {uploaded_file.name} ({len(chunks)} chunks)")
                        
                    finally:
                        # Clean up temporary file
                        if tmp_path.exists():
                            os.unlink(tmp_path)
                
                if processed_files:
                    # Rebuild embeddings with new documents
                    st.info("Rebuilding embeddings with new documents...")
                    success = self.embedding_indexer.initialize_indexes(self.documents_df, force_rebuild=True)
                    
                    if success:
                        # Update query suggestions
                        self._update_query_suggestions()
                        st.success(f"✅ Successfully processed {len(processed_files)} PDF(s)")
                        st.info(f"Total documents: {len(self.documents_df)} chunks")
                        return True
                    else:
                        st.error("Failed to rebuild embeddings")
                        return False
                else:
                    st.warning("No files were successfully processed")
                    return False
                    
        except Exception as e:
            st.error(f"Error processing uploaded files: {e}")
            logger.error(f"Error in upload_and_process_pdfs: {e}")
            return False
    
    def _update_query_suggestions(self):
        """Update query suggestions based on current documents"""
        try:
            if self.documents_df is not None and not self.documents_df.empty:
                # Extract key topics from document titles and content
                file_names = self.documents_df['file_name'].unique()
                
                suggestions = []
                
                # Generate suggestions from file names
                for file_name in file_names[:10]:  # Limit to 10 files
                    clean_name = file_name.replace('.pdf', '').replace('_', ' ').replace('-', ' ')
                    suggestions.extend([
                        f"What is {clean_name}?",
                        f"Explain {clean_name}",
                        f"How does {clean_name} work?",
                        f"Applications of {clean_name}"
                    ])
                
                # Add general research-oriented queries
                general_queries = [
                    "What are the key findings?",
                    "What methods are used?",
                    "What are the main challenges?",
                    "What are the applications?",
                    "What is the methodology?",
                    "What are the results?",
                    "What is the conclusion?",
                    "What are the limitations?",
                    "What is the contribution?",
                    "What is the innovation?",
                    "machine learning",
                    "deep learning",
                    "artificial intelligence",
                    "data science",
                    "neural networks",
                    "computer vision",
                    "natural language processing",
                    "robotics",
                    "quantum computing",
                    "reinforcement learning"
                ]
                
                suggestions.extend(general_queries)
                
                # Remove duplicates and limit
                self.query_suggestions = list(dict.fromkeys(suggestions))[:30]
                
        except Exception as e:
            logger.error(f"Error updating query suggestions: {e}")
            self.query_suggestions = []

    def get_enhanced_query_suggestions(self, current_query: str = "") -> List[str]:
        """
        Get enhanced auto-suggestions based on current query
        
        Args:
            current_query: Current user input
            
        Returns:
            List of suggested queries
        """
        if not current_query.strip():
            return self.query_suggestions[:10]
        
        current_lower = current_query.lower()
        
        # Filter suggestions based on current input
        filtered_suggestions = [
            suggestion for suggestion in self.query_suggestions
            if current_lower in suggestion.lower()
        ]
        
        # If no matches, return general suggestions
        if not filtered_suggestions:
            return self.query_suggestions[:5]
        
        return filtered_suggestions[:10]

    def calculate_accuracy_score(self, search_result: Dict, query: str) -> float:
        """
        Calculate accuracy score for a search result
        
        Args:
            search_result: Individual search result
            query: User query
            
        Returns:
            Accuracy score between 0 and 1
        """
        try:
            # Base score from similarity
            base_score = search_result.get('score', 0.5)
            
            # Text relevance (keyword overlap)
            query_words = set(query.lower().split())
            doc_text = search_result.get('text', '').lower()
            doc_words = set(doc_text.split())
            
            if doc_words:
                keyword_overlap = len(query_words.intersection(doc_words)) / len(query_words)
            else:
                keyword_overlap = 0.0
            
            # Length penalty (too short or too long chunks get lower scores)
            text_length = len(search_result.get('text', ''))
            length_score = 1.0
            if text_length < 100:
                length_score = 0.7
            elif text_length > 2000:
                length_score = 0.8
            
            # Combine scores
            final_score = (base_score * 0.6 + keyword_overlap * 0.3 + length_score * 0.1)
            
            # Normalize to 0-1 range
            return min(max(final_score, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating accuracy score: {e}")
            return 0.5

    def initialize_system(self, force_rebuild: bool = False) -> bool:
        """Initialize system with existing documents"""
        try:
            with st.spinner("Initializing system..."):
                # First, try to load existing indexes
                if not force_rebuild and self.embedding_indexer.load_indexes():
                    self.documents_df = self.embedding_indexer.documents
                    self._update_query_suggestions()
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
                
                # Update query suggestions
                self._update_query_suggestions()
                
                self.is_initialized = True
                st.success("✅ System initialized successfully")
                return True
                
        except Exception as e:
            st.error(f"❌ Error initializing system: {e}")
            return False
    
    def search_and_summarize(self, query: str, search_type: str = "hybrid", 
                           top_k: int = 5, summary_type: str = "medium") -> Dict:
        """
        Perform search and summarization with enhanced metrics
        
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
            
            if not search_results:
                return {'error': 'No relevant documents found'}
            
            # Add accuracy scores to search results
            for i, result in enumerate(search_results):
                search_results[i]['accuracy_score'] = self.calculate_accuracy_score(result, query)
            
            # Summarize
            summary_start = time.time()
            summary_result = self.local_llm.summarize_with_rag(query, search_results, summary_type)
            summary_time = time.time() - summary_start
            
            # Enhance summary result with additional stats
            if summary_result.get('status') == 'success':
                summary_text = summary_result.get('summary', '')
                summary_result['summary_stats'] = {
                    'word_count': len(summary_text.split()),
                    'character_count': len(summary_text),
                    'sentence_count': len([s for s in summary_text.split('.') if s.strip()]),
                }
                summary_result['generation_time'] = summary_time
            
            total_time = time.time() - start_time
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            # Record enhanced performance metrics
            performance = {
                'query': query,
                'search_time': search_time,
                'summary_time': summary_time,
                'total_time': total_time,
                'memory_usage': final_memory - initial_memory,
                'search_type': search_type,
                'summary_type': summary_type,
                'num_results': len(search_results),
                'avg_accuracy': sum(r.get('accuracy_score', 0) for r in search_results) / len(search_results),
                'max_accuracy': max(r.get('accuracy_score', 0) for r in search_results),
                'min_accuracy': min(r.get('accuracy_score', 0) for r in search_results)
            }
            self.performance_data.append(performance)
            
            # Keep only last 100 performance records
            if len(self.performance_data) > 100:
                self.performance_data = self.performance_data[-100:]
            
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
            page_title="Document Search & Response Generation System",
            page_icon="🔍",
            layout="wide"
        )
        
        st.title("🔍 Document Search & Response Generation System")
        st.markdown("Advanced RAG system with local LLM integration")
        
        # Sidebar
        with st.sidebar:
            st.header("System Status")
            
            # System initialization
            if not self.is_initialized:
                # Try to auto-initialize by loading existing indexes
                if self.embedding_indexer.load_indexes():
                    self.documents_df = self.embedding_indexer.documents
                    self._update_query_suggestions()
                    self.is_initialized = True
                    st.success("✅ Auto-loaded existing indexes")
                else:
                    st.info("Please initialize the system first.")
                    if st.button("Initialize System", type="primary"):
                        self.initialize_system()
            else:
                st.success("✅ System Initialized")
                if st.button("Reinitialize System"):
                    self.initialize_system(force_rebuild=True)
            
            # Document statistics
            if self.is_initialized and self.documents_df is not None:
                st.subheader("📊 Document Statistics")
                total_docs = self.documents_df['file_name'].nunique()
                total_chunks = len(self.documents_df)
                st.metric("Documents", total_docs)
                st.metric("Text Chunks", total_chunks)
            
            # PDF Upload Section
            st.subheader("📁 Upload PDFs")
            uploaded_files = st.file_uploader(
                "Upload PDF files",
                type="pdf",
                accept_multiple_files=True,
                help="Upload one or more PDF files to add to the knowledge base"
            )
            
            if uploaded_files:
                if st.button("Process Uploaded PDFs", type="secondary"):
                    if self.upload_and_process_pdfs(uploaded_files):
                        st.rerun()  # Refresh the app after successful upload
            
            # LLM Status
            st.subheader("🤖 LLM Status")
            if self.local_llm.is_available:
                st.success(f"✅ {self.local_llm.model}")
            else:
                st.error("❌ LLM not available")
                st.info("Please start Ollama: `ollama serve`")
            
            # Performance monitoring
            if self.performance_data:
                st.subheader("⚡ Performance")
                avg_time = sum(p.get('total_time', 0) for p in self.performance_data) / len(self.performance_data)
                st.metric("Avg Response Time", f"{avg_time:.2f}s")
        
        # Main interface
        if not self.is_initialized:
            st.info("Please initialize the system first using the sidebar.")
            return
        
        # Query input with enhanced auto-suggestions
        st.header("🔍 Search & Summarize")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            query = st.text_input(
                "Enter your query:",
                placeholder="What would you like to know about the documents?",
                help="Type your question or search term"
            )
            
            # Enhanced auto-suggestions
            if query.strip():
                suggestions = self.get_enhanced_query_suggestions(query)
                if suggestions:
                    st.write("💡 **Suggestions based on your input:**")
                    suggestion_cols = st.columns(2)
                    for i, suggestion in enumerate(suggestions[:6]):  # Show top 6
                        with suggestion_cols[i % 2]:
                            if st.button(f"💬 {suggestion}", key=f"suggest_{i}"):
                                query = suggestion
                                st.rerun()
            else:
                # Show general suggestions when no query
                suggestions = self.get_enhanced_query_suggestions()
                if suggestions:
                    with st.expander("💡 Query Suggestions", expanded=False):
                        st.write("**Popular queries:**")
                        for i, suggestion in enumerate(suggestions[:10]):
                            if st.button(f"💬 {suggestion}", key=f"general_suggest_{i}"):
                                query = suggestion
                                st.rerun()
        
        with col2:
            st.write("")  # Spacing
        
        # Search options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            search_type = st.selectbox(
                "Search Type:",
                ["hybrid", "semantic", "keyword"],
                help="Hybrid combines semantic and keyword search"
            )
        
        with col2:
            top_k = st.selectbox(
                "Results to retrieve:",
                [3, 5, 7, 10],
                index=1,
                help="Number of relevant documents to retrieve"
            )
        
        with col3:
            summary_type = st.selectbox(
                "Summary Length:",
                ["short", "medium", "long"],
                index=1,
                help="Short: ~100 words, Medium: ~200 words, Long: ~300 words"
            )
        
        # Search and summarize
        if st.button("🚀 Search & Summarize", type="primary") and query.strip():
            with st.spinner("Processing your query..."):
                result = self.search_and_summarize(query, search_type, top_k, summary_type)
                
                if 'error' in result:
                    st.error(result['error'])
                else:
                    # Display results with enhanced formatting
                    self._display_enhanced_results(result, query)

    def _display_enhanced_results(self, result: Dict, query: str):
        """Display search results and summary with enhanced formatting and accuracy scores"""
        
        # Summary section
        st.header("📝 Generated Summary")
        summary_data = result['summary']
        
        if summary_data['status'] == 'success':
            # Summary content
            st.markdown("### Summary")
            st.markdown(f"**Length**: {summary_data['summary_type'].title()}")
            st.write(summary_data['summary'])
            
            # Summary statistics in columns
            stats = summary_data.get('summary_stats', {})
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📊 Words", stats.get('word_count', 0))
            with col2:
                st.metric("📄 Sentences", stats.get('sentence_count', 0))
            with col3:
                st.metric("🔤 Characters", stats.get('character_count', 0))
            with col4:
                st.metric("⏱️ Generation Time", f"{summary_data.get('generation_time', 0):.2f}s")
        else:
            st.error(f"Summary generation failed: {summary_data.get('error', 'Unknown error')}")
        
        # Search results section with enhanced display
        st.header("🔍 Search Results")
        search_results = result['search_results']
        
        if not search_results:
            st.warning("No relevant documents found.")
            return
        
        # Add accuracy scores to results
        for i, search_result in enumerate(search_results):
            search_results[i]['accuracy_score'] = self.calculate_accuracy_score(search_result, query)
        
        # Sort by accuracy score
        search_results.sort(key=lambda x: x.get('accuracy_score', 0), reverse=True)
        
        # Display total results found
        st.info(f"Found {len(search_results)} relevant documents (sorted by accuracy)")
        
        # Enhanced pagination
        results_per_page = 3
        total_pages = (len(search_results) + results_per_page - 1) // results_per_page
        
        if total_pages > 1:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                page = st.selectbox(
                    "Page:",
                    range(1, total_pages + 1),
                    help=f"Showing {results_per_page} results per page"
                )
        else:
            page = 1
        
        # Calculate page range
        start_idx = (page - 1) * results_per_page
        end_idx = min(start_idx + results_per_page, len(search_results))
        
        # Display results for current page
        for i in range(start_idx, end_idx):
            result_item = search_results[i]
            
            # Enhanced result card
            with st.container():
                st.markdown("---")
                
                # Header with accuracy score and source
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.markdown(f"**📄 Document:** {result_item.get('file_name', 'Unknown')}")
                with col2:
                    accuracy = result_item.get('accuracy_score', 0)
                    color = "🟢" if accuracy > 0.7 else "🟡" if accuracy > 0.4 else "🔴"
                    st.markdown(f"**🎯 Accuracy:** {color} {accuracy:.1%}")
                with col3:
                    similarity = result_item.get('score', 0)
                    st.markdown(f"**🔗 Similarity:** {similarity:.3f}")
                
                # Document details
                col1, col2 = st.columns([1, 1])
                with col1:
                    if 'page_number' in result_item:
                        st.markdown(f"**📖 Page:** {result_item['page_number']}")
                with col2:
                    if 'chunk_id' in result_item:
                        st.markdown(f"**🔢 Chunk:** {result_item['chunk_id']}")
                
                # Content preview
                content = result_item.get('text', 'No content available')
                if len(content) > 500:
                    with st.expander(f"📋 Content Preview (Click to expand)"):
                        st.write(content)
                else:
                    st.markdown("**📋 Content:**")
                    st.write(content)
                
                # Relevance indicators
                if accuracy > 0.8:
                    st.success("🎯 Highly relevant result")
                elif accuracy > 0.6:
                    st.info("📌 Relevant result")
                elif accuracy < 0.3:
                    st.warning("⚠️ Low relevance - consider refining your query")
        
        # Page navigation info
        if total_pages > 1:
            st.markdown(f"*Showing results {start_idx + 1}-{end_idx} of {len(search_results)}*")
    
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