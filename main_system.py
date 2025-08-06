"""
Main System Orchestrator
Integrates all phases of the Document Search and Summarization System
"""
import logging
import time
from typing import Dict, List, Optional
import streamlit as st

from phase1_data_processor import Phase1DataProcessor
from phase2_embedding_indexer import Phase2EmbeddingIndexer
from phase3_local_llm import Phase3LocalLLM
from phase4_evaluator import Phase4Evaluator
from phase5_streamlit_app import Phase5StreamlitApp

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MainSystem:
    """Main system orchestrator integrating all phases"""
    
    def __init__(self):
        """Initialize the main system"""
        self.phase1 = Phase1DataProcessor()
        self.phase2 = Phase2EmbeddingIndexer()
        self.phase3 = Phase3LocalLLM()
        self.phase4 = Phase4Evaluator()
        self.phase5 = Phase5StreamlitApp()
        
        self.is_initialized = False
        self.documents_df = None
        
    def run_phase1(self) -> bool:
        """Run Phase 1: Data Preparation & Processing"""
        logger.info("Starting Phase 1: Data Preparation & Processing")
        
        try:
            self.documents_df = self.phase1.process_corpus()
            
            if self.documents_df.empty:
                logger.error("Phase 1 failed: No documents processed")
                return False
            
            logger.info(f"Phase 1 completed: {len(self.documents_df)} chunks from {self.documents_df['file_name'].nunique()} documents")
            return True
            
        except Exception as e:
            logger.error(f"Phase 1 failed: {e}")
            return False
    
    def run_phase2(self) -> bool:
        """Run Phase 2: Local Embedding & Indexing"""
        logger.info("Starting Phase 2: Local Embedding & Indexing")
        
        try:
            if self.documents_df is None:
                logger.error("Phase 2 failed: No documents available")
                return False
            
            # Use the new initialize_indexes method
            success = self.phase2.initialize_indexes(self.documents_df, force_rebuild=False)
            
            if success:
                stats = self.phase2.get_statistics()
                logger.info(f"Phase 2 completed: {stats.get('semantic_vectors', 0)} vectors indexed")
                return True
            else:
                logger.error("Phase 2 failed: Could not initialize indexes")
                return False
            
        except Exception as e:
            logger.error(f"Phase 2 failed: {e}")
            return False
    
    def test_phase3(self) -> bool:
        """Test Phase 3: Local LLM Integration"""
        logger.info("Testing Phase 3: Local LLM Integration")
        
        try:
            test_result = self.phase3.test_connection()
            
            if test_result['available']:
                logger.info(f"Phase 3 test successful: {test_result['model']} available")
                return True
            else:
                logger.warning(f"Phase 3 test failed: {test_result.get('error', 'Unknown error')}")
                return False
                
        except Exception as e:
            logger.error(f"Phase 3 test failed: {e}")
            return False
    
    def run_phase4_evaluation(self) -> Dict:
        """Run Phase 4: Evaluation Framework"""
        logger.info("Starting Phase 4: Evaluation Framework")
        
        try:
            # Generate synthetic queries
            synthetic_queries = self.phase4.generate_synthetic_queries(self.documents_df)
            
            # Run evaluation
            search_results = []
            summaries = []
            reference_summaries = []
            
            for query_info in synthetic_queries[:10]:  # Limit for performance
                # Search
                search_result = self.phase2.hybrid_search(query_info['query'])
                search_results.append(search_result)
                
                # Summarize
                summary_result = self.phase3.summarize_with_rag(query_info['query'], search_result)
                summaries.append(summary_result['summary'])
                
                # Create reference summary
                expected_doc = query_info['expected_document']
                doc_text = self.documents_df[self.documents_df['file_name'] == expected_doc]['text'].iloc[0]
                reference_summaries.append(doc_text[:200] + "...")
            
            # Calculate metrics
            retrieval_metrics = self.phase4.evaluate_retrieval_metrics(
                search_results, [q['expected_chunks'] for q in synthetic_queries[:10]]
            )
            
            summary_metrics = self.phase4.evaluate_summary_quality(
                summaries, reference_summaries
            )
            
            # Generate report
            report = self.phase4.create_evaluation_report(
                retrieval_metrics, summary_metrics, {}
            )
            
            logger.info("Phase 4 evaluation completed")
            return {
                'retrieval_metrics': retrieval_metrics,
                'summary_metrics': summary_metrics,
                'report': report
            }
            
        except Exception as e:
            logger.error(f"Phase 4 evaluation failed: {e}")
            return {'error': str(e)}
    
    def initialize_complete_system(self, force_rebuild: bool = False) -> bool:
        """
        Initialize the complete system
        
        Args:
            force_rebuild: Force rebuild all components
            
        Returns:
            True if initialization successful
        """
        logger.info("Initializing complete Document Search and Summarization System")
        
        try:
            # Phase 1: Data Processing
            if not self.run_phase1():
                return False
            
            # Phase 2: Embedding & Indexing
            if not self.run_phase2():
                return False
            
            # Phase 3: Test LLM
            if not self.test_phase3():
                logger.warning("LLM not available, but continuing with other phases")
            
            self.is_initialized = True
            logger.info("Complete system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            return False
    
    def search_and_summarize(self, query: str, search_type: str = "hybrid", 
                           top_k: int = 5, summary_type: str = "medium") -> Dict:
        """
        Perform complete search and summarization
        
        Args:
            query: Search query
            search_type: Type of search (semantic/keyword/hybrid)
            top_k: Number of results
            summary_type: Type of summary
            
        Returns:
            Dictionary with results
        """
        if not self.is_initialized:
            return {'error': 'System not initialized'}
        
        try:
            # Search
            if search_type == "semantic":
                search_results = self.phase2.semantic_search(query, top_k)
            elif search_type == "keyword":
                search_results = self.phase2.keyword_search(query, top_k)
            else:  # hybrid
                search_results = self.phase2.hybrid_search(query, top_k)
            
            # Summarize
            summary_result = self.phase3.summarize_with_rag(query, search_results, summary_type)
            
            return {
                'search_results': search_results,
                'summary': summary_result,
                'query': query,
                'search_type': search_type,
                'summary_type': summary_type
            }
            
        except Exception as e:
            logger.error(f"Search and summarization failed: {e}")
            return {'error': str(e)}
    
    def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        status = {
            'initialized': self.is_initialized,
            'documents_processed': len(self.documents_df) if self.documents_df is not None else 0,
            'unique_documents': self.documents_df['file_name'].nunique() if self.documents_df is not None else 0,
            'llm_available': self.phase3.is_available,
            'llm_model': self.phase3.model
        }
        
        if self.is_initialized:
            status.update(self.phase2.get_statistics())
        
        return status
    
    def run_streamlit_interface(self):
        """Run the Streamlit interface"""
        self.phase5.run_streamlit_app()

def main():
    """Main function to run the system"""
    system = MainSystem()
    
    # Initialize system
    if system.initialize_complete_system():
        print("✅ System initialized successfully")
        
        # Get system status
        status = system.get_system_status()
        print(f"📊 System Status:")
        print(f"  - Documents: {status['unique_documents']}")
        print(f"  - Chunks: {status['documents_processed']}")
        print(f"  - LLM Available: {status['llm_available']}")
        print(f"  - LLM Model: {status['llm_model']}")
        
        # Run Streamlit interface
        system.run_streamlit_interface()
    else:
        print("❌ System initialization failed")

if __name__ == "__main__":
    main() 