"""
Phase 4: Evaluation Framework
- Test Set Creation: Generate synthetic queries for each document
- Retrieval Metrics: Precision@K, Recall@K, MRR
- Summary Quality: ROUGE scores, semantic similarity
- Performance Metrics: Response time, memory usage
"""
import logging
import time
import psutil
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from config import TEST_SET_RATIO, EVALUATION_METRICS

logger = logging.getLogger(__name__)

class Phase4Evaluator:
    """Phase 4: Comprehensive Evaluation Framework"""
    
    def __init__(self):
        """Initialize the evaluator"""
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    def generate_synthetic_queries(self, documents: pd.DataFrame, 
                                 num_queries_per_doc: int = 3) -> List[Dict]:
        """
        Generate synthetic queries for each document
        
        Args:
            documents: DataFrame with processed documents
            num_queries_per_doc: Number of queries to generate per document
            
        Returns:
            List of synthetic query dictionaries
        """
        synthetic_queries = []
        
        # Group by file to generate queries per document
        file_groups = documents.groupby('file_name')
        
        for file_name, group in file_groups:
            # Get text from first few chunks
            text_samples = group['text'].head(3).tolist()
            combined_text = ' '.join(text_samples)
            
            # Generate different types of queries
            queries = self._generate_query_variations(file_name, combined_text, num_queries_per_doc)
            
            for i, query in enumerate(queries):
                synthetic_queries.append({
                    'query_id': f"{file_name}_query_{i}",
                    'query': query,
                    'expected_document': file_name,
                    'expected_chunks': group['chunk_id'].tolist(),
                    'query_type': self._classify_query_type(query)
                })
        
        logger.info(f"Generated {len(synthetic_queries)} synthetic queries")
        return synthetic_queries
    
    def _generate_query_variations(self, file_name: str, text: str, num_queries: int) -> List[str]:
        """
        Generate different types of queries for a document
        
        Args:
            file_name: Document name
            text: Document text
            num_queries: Number of queries to generate
            
        Returns:
            List of generated queries
        """
        queries = []
        
        # Convert file name to readable topic
        topic = file_name.replace('_', ' ').replace('-', ' ')
        
        # Type 1: Topic-based queries
        queries.append(f"What is {topic}?")
        queries.append(f"Explain {topic}")
        
        # Type 2: Method-based queries
        if num_queries > 2:
            queries.append(f"What methods are used in {topic}?")
        
        # Type 3: Application-based queries
        if num_queries > 3:
            queries.append(f"What are the applications of {topic}?")
        
        # Type 4: Challenge-based queries
        if num_queries > 4:
            queries.append(f"What are the challenges in {topic}?")
        
        return queries[:num_queries]
    
    def _classify_query_type(self, query: str) -> str:
        """Classify query type for analysis"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['what is', 'explain', 'define']):
            return 'definition'
        elif any(word in query_lower for word in ['method', 'approach', 'technique']):
            return 'methodology'
        elif any(word in query_lower for word in ['application', 'use', 'implement']):
            return 'application'
        elif any(word in query_lower for word in ['challenge', 'problem', 'issue']):
            return 'challenge'
        else:
            return 'general'
    
    def evaluate_retrieval_metrics(self, search_results: List[List[Dict]], 
                                 ground_truth: List[List[int]], 
                                 k_values: List[int] = [1, 3, 5]) -> Dict:
        """
        Evaluate retrieval performance using multiple metrics
        
        Args:
            search_results: List of search results for each query
            ground_truth: List of ground truth document indices
            k_values: K values for Precision@K and Recall@K
            
        Returns:
            Dictionary with retrieval metrics
        """
        metrics = {}
        
        for k in k_values:
            precision_at_k = []
            recall_at_k = []
            mrr_scores = []
            
            for results, truth in zip(search_results, ground_truth):
                if not results or not truth:
                    continue
                
                # Get retrieved document indices
                retrieved_indices = [doc['document'].get('chunk_id', 0) for doc in results[:k]]
                
                # Calculate Precision@K
                relevant_retrieved = len(set(retrieved_indices) & set(truth))
                precision = relevant_retrieved / len(retrieved_indices) if retrieved_indices else 0
                precision_at_k.append(precision)
                
                # Calculate Recall@K
                recall = relevant_retrieved / len(truth) if truth else 0
                recall_at_k.append(recall)
                
                # Calculate MRR
                mrr = self._calculate_mrr(retrieved_indices, truth)
                mrr_scores.append(mrr)
            
            metrics[f'precision_at_{k}'] = np.mean(precision_at_k) if precision_at_k else 0
            metrics[f'recall_at_{k}'] = np.mean(recall_at_k) if recall_at_k else 0
            metrics[f'mrr_at_{k}'] = np.mean(mrr_scores) if mrr_scores else 0
        
        # Overall metrics
        metrics['avg_precision'] = np.mean([metrics[f'precision_at_{k}'] for k in k_values])
        metrics['avg_recall'] = np.mean([metrics[f'recall_at_{k}'] for k in k_values])
        metrics['avg_mrr'] = np.mean([metrics[f'mrr_at_{k}'] for k in k_values])
        
        return metrics
    
    def _calculate_mrr(self, retrieved_indices: List[int], truth: List[int]) -> float:
        """Calculate Mean Reciprocal Rank"""
        for rank, idx in enumerate(retrieved_indices, 1):
            if idx in truth:
                return 1.0 / rank
        return 0.0
    
    def evaluate_summary_quality(self, generated_summaries: List[str], 
                               reference_summaries: List[str]) -> Dict:
        """
        Evaluate summary quality using ROUGE scores and semantic similarity
        
        Args:
            generated_summaries: List of generated summaries
            reference_summaries: List of reference summaries
            
        Returns:
            Dictionary with summary quality metrics
        """
        if not generated_summaries or not reference_summaries:
            return {}
        
        rouge_scores = {
            'rouge1': [],
            'rouge2': [],
            'rougeL': []
        }
        
        semantic_similarities = []
        
        for gen_sum, ref_sum in zip(generated_summaries, reference_summaries):
            if not gen_sum.strip() or not ref_sum.strip():
                continue
            
            # Calculate ROUGE scores
            scores = self.rouge_scorer.score(ref_sum, gen_sum)
            rouge_scores['rouge1'].append(scores['rouge1'].fmeasure)
            rouge_scores['rouge2'].append(scores['rouge2'].fmeasure)
            rouge_scores['rougeL'].append(scores['rougeL'].fmeasure)
            
            # Calculate semantic similarity (simplified)
            gen_words = set(gen_sum.lower().split())
            ref_words = set(ref_sum.lower().split())
            if ref_words:
                similarity = len(gen_words.intersection(ref_words)) / len(ref_words)
                semantic_similarities.append(similarity)
        
        # Calculate averages
        metrics = {}
        for metric, scores in rouge_scores.items():
            metrics[f'avg_{metric}'] = np.mean(scores) if scores else 0
            metrics[f'std_{metric}'] = np.std(scores) if scores else 0
        
        metrics['avg_semantic_similarity'] = np.mean(semantic_similarities) if semantic_similarities else 0
        metrics['total_evaluated'] = len(rouge_scores['rouge1'])
        
        return metrics
    
    def evaluate_performance_metrics(self, performance_data: List[Dict]) -> Dict:
        """
        Evaluate system performance metrics
        
        Args:
            performance_data: List of performance measurements
            
        Returns:
            Dictionary with performance metrics
        """
        if not performance_data:
            return {}
        
        # Extract timing data
        search_times = [p.get('search_time', 0) for p in performance_data]
        summary_times = [p.get('summary_time', 0) for p in performance_data]
        total_times = [p.get('total_time', 0) for p in performance_data]
        
        # Memory usage
        memory_usage = [p.get('memory_usage', 0) for p in performance_data]
        
        metrics = {
            'avg_search_time': np.mean(search_times) if search_times else 0,
            'avg_summary_time': np.mean(summary_times) if summary_times else 0,
            'avg_total_time': np.mean(total_times) if total_times else 0,
            'avg_memory_usage': np.mean(memory_usage) if memory_usage else 0,
            'max_search_time': np.max(search_times) if search_times else 0,
            'max_summary_time': np.max(summary_times) if summary_times else 0,
            'total_queries': len(performance_data)
        }
        
        return metrics
    
    def create_evaluation_report(self, retrieval_metrics: Dict, 
                               summary_metrics: Dict,
                               performance_metrics: Dict) -> str:
        """
        Create comprehensive evaluation report
        
        Args:
            retrieval_metrics: Retrieval performance metrics
            summary_metrics: Summary quality metrics
            performance_metrics: System performance metrics
            
        Returns:
            Formatted evaluation report
        """
        report = "=== COMPREHENSIVE EVALUATION REPORT ===\n\n"
        
        # Retrieval Evaluation
        report += "1. RETRIEVAL PERFORMANCE EVALUATION\n"
        report += "=" * 40 + "\n"
        if retrieval_metrics:
            for k in [1, 3, 5]:
                report += f"Precision@{k}: {retrieval_metrics.get(f'precision_at_{k}', 0):.4f}\n"
                report += f"Recall@{k}: {retrieval_metrics.get(f'recall_at_{k}', 0):.4f}\n"
                report += f"MRR@{k}: {retrieval_metrics.get(f'mrr_at_{k}', 0):.4f}\n"
                report += "\n"
            
            report += f"Average Precision: {retrieval_metrics.get('avg_precision', 0):.4f}\n"
            report += f"Average Recall: {retrieval_metrics.get('avg_recall', 0):.4f}\n"
            report += f"Average MRR: {retrieval_metrics.get('avg_mrr', 0):.4f}\n"
        else:
            report += "No retrieval metrics available\n"
        
        # Summary Quality Evaluation
        report += "\n2. SUMMARY QUALITY EVALUATION\n"
        report += "=" * 40 + "\n"
        if summary_metrics:
            report += f"ROUGE-1: {summary_metrics.get('avg_rouge1', 0):.4f} (±{summary_metrics.get('std_rouge1', 0):.4f})\n"
            report += f"ROUGE-2: {summary_metrics.get('avg_rouge2', 0):.4f} (±{summary_metrics.get('std_rouge2', 0):.4f})\n"
            report += f"ROUGE-L: {summary_metrics.get('avg_rougeL', 0):.4f} (±{summary_metrics.get('std_rougeL', 0):.4f})\n"
            report += f"Semantic Similarity: {summary_metrics.get('avg_semantic_similarity', 0):.4f}\n"
            report += f"Total Evaluated: {summary_metrics.get('total_evaluated', 0)}\n"
        else:
            report += "No summary metrics available\n"
        
        # Performance Evaluation
        report += "\n3. SYSTEM PERFORMANCE EVALUATION\n"
        report += "=" * 40 + "\n"
        if performance_metrics:
            report += f"Average Search Time: {performance_metrics.get('avg_search_time', 0):.3f}s\n"
            report += f"Average Summary Time: {performance_metrics.get('avg_summary_time', 0):.3f}s\n"
            report += f"Average Total Time: {performance_metrics.get('avg_total_time', 0):.3f}s\n"
            report += f"Average Memory Usage: {performance_metrics.get('avg_memory_usage', 0):.1f}MB\n"
            report += f"Max Search Time: {performance_metrics.get('max_search_time', 0):.3f}s\n"
            report += f"Max Summary Time: {performance_metrics.get('max_summary_time', 0):.3f}s\n"
            report += f"Total Queries: {performance_metrics.get('total_queries', 0)}\n"
        else:
            report += "No performance metrics available\n"
        
        # Overall Assessment
        report += "\n4. OVERALL ASSESSMENT\n"
        report += "=" * 40 + "\n"
        
        if retrieval_metrics and summary_metrics and performance_metrics:
            # Calculate overall score
            retrieval_score = retrieval_metrics.get('avg_mrr', 0)
            summary_score = summary_metrics.get('avg_rougeL', 0)
            performance_score = 1.0 / (1.0 + performance_metrics.get('avg_total_time', 1.0))
            
            overall_score = (retrieval_score * 0.4 + summary_score * 0.4 + performance_score * 0.2)
            
            report += f"Overall System Score: {overall_score:.4f}\n\n"
            
            if overall_score > 0.7:
                report += "Assessment: EXCELLENT\n"
            elif overall_score > 0.5:
                report += "Assessment: GOOD\n"
            elif overall_score > 0.3:
                report += "Assessment: FAIR\n"
            else:
                report += "Assessment: NEEDS IMPROVEMENT\n"
            
            report += f"\nBreakdown:\n"
            report += f"- Retrieval Performance: {retrieval_score:.4f}\n"
            report += f"- Summary Quality: {summary_score:.4f}\n"
            report += f"- System Performance: {performance_score:.4f}\n"
        else:
            report += "Insufficient data for overall assessment\n"
        
        return report
    
    def save_evaluation_results(self, results: Dict, file_path: str):
        """
        Save evaluation results to file
        
        Args:
            results: Evaluation results dictionary
            file_path: Path to save results
        """
        try:
            with open(file_path, 'w') as f:
                for key, value in results.items():
                    if isinstance(value, dict):
                        f.write(f"\n{key}:\n")
                        for sub_key, sub_value in value.items():
                            f.write(f"  {sub_key}: {sub_value}\n")
                    else:
                        f.write(f"{key}: {value}\n")
            
            logger.info(f"Evaluation results saved to {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving evaluation results: {e}") 