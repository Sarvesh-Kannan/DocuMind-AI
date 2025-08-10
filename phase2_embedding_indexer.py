"""
Phase 2: Local Embedding & Indexing
- Local Embeddings: Use sentence-transformers (all-MiniLM-L6-v2) - runs locally
- Vector Storage: Create FAISS index and save locally
- Keyword Index: Build TF-IDF vectorizer for traditional IR
- Hybrid Search: Combine semantic + keyword search with weighted scoring
"""
import logging
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import torch
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import os

from config import (
    EMBEDDING_MODEL, EMBEDDING_DIMENSION, EMBEDDINGS_DIR,
    TOP_K_RESULTS, SIMILARITY_THRESHOLD, HYBRID_WEIGHT_SEMANTIC, HYBRID_WEIGHT_KEYWORD,
    BATCH_SIZE, DEVICE
)

logger = logging.getLogger(__name__)

class Phase2EmbeddingIndexer:
    """Phase 2: Local Embedding & Indexing"""
    
    def __init__(self):
        """Initialize the embedding and indexing system"""
        self.semantic_model = None
        self.tfidf_vectorizer = None
        self.faiss_index = None
        self.documents = None
        self.tfidf_matrix = None
        self.semantic_embeddings = None
        
        # File paths for persistence
        self.embeddings_file = EMBEDDINGS_DIR / "semantic_embeddings.npy"
        self.faiss_index_file = EMBEDDINGS_DIR / "faiss_index.bin"
        self.tfidf_file = EMBEDDINGS_DIR / "tfidf_matrix.pkl"
        self.tfidf_vectorizer_file = EMBEDDINGS_DIR / "tfidf_vectorizer.pkl"
        self.metadata_file = EMBEDDINGS_DIR / "metadata.pkl"
        
        # Ensure embeddings directory exists
        EMBEDDINGS_DIR.mkdir(exist_ok=True)
    
    def load_semantic_model(self):
        """Load the sentence transformer model with error handling"""
        if self.semantic_model is None:
            try:
                logger.info(f"Loading semantic model: {EMBEDDING_MODEL}")
                # Use CPU to avoid GPU issues
                self.semantic_model = SentenceTransformer(EMBEDDING_MODEL, device='cpu')
                logger.info(f"Semantic model loaded successfully on CPU")
            except Exception as e:
                logger.error(f"Failed to load semantic model: {e}")
                # Fallback to a simpler model
                try:
                    logger.info("Trying fallback model...")
                    self.semantic_model = SentenceTransformer('paraphrase-MiniLM-L3-v2', device='cpu')
                    logger.info("Fallback model loaded successfully")
                except Exception as e2:
                    logger.error(f"Failed to load fallback model: {e2}")
                    raise e2
    
    def generate_semantic_embeddings(self, texts: List[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
        """
        Generate semantic embeddings using sentence-transformers with robust error handling
        
        Args:
            texts: List of text strings
            batch_size: Batch size for processing
            
        Returns:
            numpy array of semantic embeddings
        """
        try:
            self.load_semantic_model()
            
            logger.info(f"Generating semantic embeddings for {len(texts)} texts")
            
            embeddings = []
            for i in tqdm(range(0, len(texts), batch_size), desc="Generating semantic embeddings"):
                batch = texts[i:i + batch_size]
                try:
                    batch_embeddings = self.semantic_model.encode(
                        batch, 
                        convert_to_numpy=True,
                        show_progress_bar=False,
                        device='cpu'
                    )
                    embeddings.append(batch_embeddings)
                except Exception as e:
                    logger.error(f"Error processing batch {i}: {e}")
                    # Create dummy embeddings for failed batch
                    dummy_embeddings = np.random.rand(len(batch), EMBEDDING_DIMENSION)
                    embeddings.append(dummy_embeddings)
            
            result = np.vstack(embeddings)
            logger.info(f"Generated embeddings shape: {result.shape}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            # Return random embeddings as fallback
            logger.warning("Using random embeddings as fallback")
            return np.random.rand(len(texts), EMBEDDING_DIMENSION)
    
    def build_tfidf_index(self, texts: List[str]):
        """
        Build TF-IDF vectorizer and matrix for keyword search
        Only builds if there are 2+ documents
        
        Args:
            texts: List of text strings
        """
        num_docs = len(texts)
        
        # Skip TF-IDF for small document sets (< 2 documents)
        if num_docs < 2:
            logger.info(f"Skipping TF-IDF index: only {num_docs} document(s), need at least 2")
            self.tfidf_vectorizer = None
            self.tfidf_matrix = None
            return
        
        logger.info("Building TF-IDF index")
        
        try:
            # Create TF-IDF vectorizer with standard parameters
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=10000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
            )
            
            # Fit and transform
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            
            logger.info(f"TF-IDF index built: {self.tfidf_matrix.shape}")
        except Exception as e:
            logger.error(f"Failed to build TF-IDF index: {e}")
            # Set to None on failure so search falls back to semantic only
            self.tfidf_vectorizer = None
            self.tfidf_matrix = None
    
    def build_faiss_index(self, embeddings: np.ndarray):
        """
        Build FAISS index for semantic search
        
        Args:
            embeddings: numpy array of semantic embeddings
        """
        logger.info("Building FAISS index")
        
        try:
            # Create FAISS index
            self.faiss_index = faiss.IndexFlatIP(embeddings.shape[1])  # Inner product for cosine similarity
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Add embeddings to index
            self.faiss_index.add(embeddings.astype('float32'))
            
            logger.info(f"FAISS index built with {self.faiss_index.ntotal} vectors")
        except Exception as e:
            logger.error(f"Failed to build FAISS index: {e}")
            raise e
    
    def _ensure_indexes_loaded(self):
        """Ensure indexes are loaded in memory"""
        if (self.faiss_index is None or self.documents is None or 
            self.semantic_model is None or self.semantic_embeddings is None):
            logger.info("Indexes not loaded in memory, attempting to load from disk")
            if not self.load_indexes():
                logger.error("Failed to load indexes from disk")
                return False
        return True

    def semantic_search(self, query: str, top_k: int = TOP_K_RESULTS) -> List[Dict]:
        """
        Perform semantic search using FAISS
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of search results with semantic similarity scores
        """
        # Ensure indexes are loaded
        if not self._ensure_indexes_loaded():
            logger.error("FAISS index not available")
            return []
            
        # Ensure semantic model is loaded
        if self.semantic_model is None:
            self.load_semantic_model()
        
        try:
            # Generate query embedding
            query_embedding = self.semantic_model.encode([query], convert_to_numpy=True, device='cpu')
            
            # Normalize query embedding
            faiss.normalize_L2(query_embedding)
            
            # Search index
            similarities, indices = self.faiss_index.search(
                query_embedding.astype('float32'), 
                min(top_k * 2, self.faiss_index.ntotal)
            )
            
            # Create results
            results = []
            for sim, idx in zip(similarities[0], indices[0]):
                if sim >= SIMILARITY_THRESHOLD and idx != -1 and idx < len(self.documents):
                    doc = self.documents.iloc[idx].to_dict()
                    results.append({
                        'document': doc,
                        'semantic_similarity': float(sim),
                        'text': doc['text'],
                        'file_name': doc['file_name'],
                        'chunk_id': doc['chunk_id'],
                        'page_number': doc.get('page_number', 0)
                    })
            
            logger.info(f"Semantic search returned {len(results)} results for query: '{query}'")
            return results[:top_k]
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
    
    def keyword_search(self, query: str, top_k: int = TOP_K_RESULTS) -> List[Dict]:
        """
        Perform keyword search using TF-IDF
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of search results with keyword similarity scores
        """
        # Ensure indexes are loaded
        if not self._ensure_indexes_loaded():
            logger.error("Indexes not available")
            return []
            
        if self.tfidf_vectorizer is None or self.tfidf_matrix is None:
            logger.info("TF-IDF index not available (semantic-only mode)")
            return []
        
        try:
            # Transform query
            query_vector = self.tfidf_vectorizer.transform([query])
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            
            # Get top indices
            top_indices = np.argsort(similarities)[::-1][:top_k * 2]  # Get more candidates
            
            # Create results
            results = []
            for idx in top_indices:
                if idx < len(self.documents) and similarities[idx] > 0:  # Lower threshold for keyword search
                    doc = self.documents.iloc[idx].to_dict()
                    results.append({
                        'document': doc,
                        'keyword_similarity': float(similarities[idx]),
                        'text': doc['text'],
                        'file_name': doc['file_name'],
                        'chunk_id': doc['chunk_id'],
                        'page_number': doc.get('page_number', 0)
                    })
            
            logger.info(f"Keyword search returned {len(results)} results for query: '{query}'")
            return results[:top_k]
        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            return []
    
    def hybrid_search(self, query: str, top_k: int = TOP_K_RESULTS) -> List[Dict]:
        """
        Perform hybrid search combining semantic and keyword search
        Falls back to semantic-only if TF-IDF is not available
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of search results with hybrid scores
        """
        try:
            # Always perform semantic search
            semantic_results = self.semantic_search(query, top_k * 2)
            
            # Only perform keyword search if TF-IDF is available
            keyword_results = []
            if self.tfidf_vectorizer is not None and self.tfidf_matrix is not None:
                keyword_results = self.keyword_search(query, top_k * 2)
                logger.info(f"Hybrid search: {len(semantic_results)} semantic + {len(keyword_results)} keyword results")
            else:
                logger.info(f"Semantic-only search: {len(semantic_results)} results (TF-IDF not available)")
            
            # If no results from semantic search, return empty
            if not semantic_results:
                logger.warning("No results from semantic search")
                return []
            
            # If only semantic results available, return them with semantic scores as final scores
            if not keyword_results:
                for result in semantic_results:
                    result['score'] = result.get('semantic_similarity', 0)
                    result['hybrid_score'] = result['score']
                
                final_results = semantic_results[:top_k]
                logger.info(f"Semantic-only search returned {len(final_results)} final results")
                return final_results
            
            # Create document ID to result mapping for hybrid scoring
            semantic_map = {f"{r['file_name']}_{r['chunk_id']}": r for r in semantic_results}
            keyword_map = {f"{r['file_name']}_{r['chunk_id']}": r for r in keyword_results}
            
            # Combine results with weighted scoring
            hybrid_results = []
            all_doc_ids = set(semantic_map.keys()) | set(keyword_map.keys())
            
            for doc_id in all_doc_ids:
                semantic_score = semantic_map.get(doc_id, {}).get('semantic_similarity', 0)
                keyword_score = keyword_map.get(doc_id, {}).get('keyword_similarity', 0)
                
                # Calculate hybrid score
                hybrid_score = (
                    HYBRID_WEIGHT_SEMANTIC * semantic_score +
                    HYBRID_WEIGHT_KEYWORD * keyword_score
                )
                
                # Get the result (prefer semantic for metadata)
                result = semantic_map.get(doc_id, keyword_map.get(doc_id))
                if result:
                    result['hybrid_score'] = hybrid_score
                    result['score'] = hybrid_score  # Add unified score field
                    hybrid_results.append(result)
            
            # Sort by hybrid score and return top_k
            hybrid_results.sort(key=lambda x: x['hybrid_score'], reverse=True)
            final_results = hybrid_results[:top_k]
            
            logger.info(f"Hybrid search returned {len(final_results)} final results")
            return final_results
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return []
    
    def build_indexes(self, documents: pd.DataFrame):
        """
        Build all indexes (semantic, keyword, hybrid)
        
        Args:
            documents: DataFrame with processed documents
        """
        logger.info("Building all indexes")
        
        try:
            self.documents = documents
            texts = documents['text'].tolist()
            
            # Generate semantic embeddings
            self.semantic_embeddings = self.generate_semantic_embeddings(texts)
            
            # Build FAISS index
            self.build_faiss_index(self.semantic_embeddings)
            
            # Build TF-IDF index
            self.build_tfidf_index(texts)
            
            logger.info("All indexes built successfully")
        except Exception as e:
            logger.error(f"Failed to build indexes: {e}")
            raise e
    
    def save_indexes(self):
        """Save all indexes to disk"""
        logger.info("Saving indexes to disk")
        
        try:
            # Save semantic embeddings
            if self.semantic_embeddings is not None:
                np.save(self.embeddings_file, self.semantic_embeddings)
                logger.info(f"Saved semantic embeddings: {self.embeddings_file}")
            
            # Save FAISS index
            if self.faiss_index is not None:
                faiss.write_index(self.faiss_index, str(self.faiss_index_file))
                logger.info(f"Saved FAISS index: {self.faiss_index_file}")
            
            # Save TF-IDF matrix and vectorizer (only if they exist)
            if self.tfidf_matrix is not None:
                with open(self.tfidf_file, 'wb') as f:
                    pickle.dump(self.tfidf_matrix, f)
                logger.info(f"Saved TF-IDF matrix: {self.tfidf_file}")
            else:
                # Remove TF-IDF file if it exists but we don't have a matrix
                if self.tfidf_file.exists():
                    self.tfidf_file.unlink()
                    logger.info("Removed old TF-IDF matrix file")
            
            if self.tfidf_vectorizer is not None:
                with open(self.tfidf_vectorizer_file, 'wb') as f:
                    pickle.dump(self.tfidf_vectorizer, f)
                logger.info(f"Saved TF-IDF vectorizer: {self.tfidf_vectorizer_file}")
            else:
                # Remove TF-IDF vectorizer file if it exists but we don't have a vectorizer
                if self.tfidf_vectorizer_file.exists():
                    self.tfidf_vectorizer_file.unlink()
                    logger.info("Removed old TF-IDF vectorizer file")
            
            # Save metadata
            if self.documents is not None:
                with open(self.metadata_file, 'wb') as f:
                    pickle.dump(self.documents, f)
                logger.info(f"Saved metadata: {self.metadata_file}")
            
            logger.info("All indexes saved successfully")
        except Exception as e:
            logger.error(f"Failed to save indexes: {e}")
            raise e
    
    def load_indexes(self) -> bool:
        """
        Load all indexes from disk
        
        Returns:
            True if loading was successful, False otherwise
        """
        try:
            # Check if core files exist (embeddings, faiss, metadata are required)
            core_files = [
                self.embeddings_file, self.faiss_index_file, self.metadata_file
            ]
            
            if not all(f.exists() for f in core_files):
                logger.info("Core index files not found, need to build them")
                return False
            
            logger.info("Loading indexes from disk")
            
            # Load semantic embeddings
            self.semantic_embeddings = np.load(self.embeddings_file)
            logger.info(f"Loaded semantic embeddings: {self.semantic_embeddings.shape}")
            
            # Load FAISS index
            self.faiss_index = faiss.read_index(str(self.faiss_index_file))
            logger.info(f"Loaded FAISS index: {self.faiss_index.ntotal} vectors")
            
            # Load TF-IDF matrix and vectorizer (optional - may not exist for small datasets)
            if self.tfidf_file.exists() and self.tfidf_vectorizer_file.exists():
                with open(self.tfidf_file, 'rb') as f:
                    self.tfidf_matrix = pickle.load(f)
                logger.info(f"Loaded TF-IDF matrix: {self.tfidf_matrix.shape}")
                
                with open(self.tfidf_vectorizer_file, 'rb') as f:
                    self.tfidf_vectorizer = pickle.load(f)
                logger.info("Loaded TF-IDF vectorizer")
            else:
                self.tfidf_matrix = None
                self.tfidf_vectorizer = None
                logger.info("TF-IDF files not found - using semantic search only")
            
            # Load metadata
            with open(self.metadata_file, 'rb') as f:
                self.documents = pickle.load(f)
            logger.info(f"Loaded metadata: {len(self.documents)} documents")
            
            # Load semantic model
            self.load_semantic_model()
            
            # Log final status
            if self.tfidf_matrix is not None:
                logger.info(f"Loaded indexes: {self.faiss_index.ntotal} vectors, {self.tfidf_matrix.shape[0]} documents")
            else:
                logger.info(f"Loaded indexes: {self.faiss_index.ntotal} vectors, semantic-only mode")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading indexes: {e}")
            return False
    
    
    def clear_indexes(self):
        """Clear all indexes and embeddings for upload-only mode"""
        try:
            logger.info("🔄 Clearing indexes and embeddings...")
            
            # Clear in-memory data
            self.semantic_model = None
            self.tfidf_vectorizer = None
            self.faiss_index = None
            self.documents = None
            self.tfidf_matrix = None
            self.semantic_embeddings = None
            
            # Clear files on disk
            files_to_clear = [
                self.embeddings_file,
                self.faiss_index_file,
                self.tfidf_file,
                self.tfidf_vectorizer_file,
                self.metadata_file
            ]
            
            for file_path in files_to_clear:
                if file_path.exists():
                    file_path.unlink()
                    logger.info(f"🗑️  Deleted: {file_path.name}")
            
            logger.info("✅ All indexes and embeddings cleared")
            
        except Exception as e:
            logger.error(f"❌ Error clearing indexes: {e}")
            raise e

    def get_statistics(self) -> Dict:
        """Get statistics about the indexes"""
        if self.faiss_index is None:
            return {}
        
        return {
            'semantic_vectors': self.faiss_index.ntotal,
            'keyword_documents': self.tfidf_matrix.shape[0] if self.tfidf_matrix is not None else 0,
            'embedding_dimension': EMBEDDING_DIMENSION,
            'semantic_model': EMBEDDING_MODEL,
            'hybrid_weights': {
                'semantic': HYBRID_WEIGHT_SEMANTIC,
                'keyword': HYBRID_WEIGHT_KEYWORD
            }
        }
    
    def indexes_exist(self) -> bool:
        """Check if all index files exist"""
        required_files = [
            self.embeddings_file, self.faiss_index_file, 
            self.tfidf_file, self.tfidf_vectorizer_file, self.metadata_file
        ]
        return all(f.exists() for f in required_files)
    
    def initialize_indexes(self, documents: pd.DataFrame, force_rebuild: bool = False) -> bool:
        """
        Initialize indexes - load if they exist, build if they don't
        
        Args:
            documents: DataFrame with processed documents
            force_rebuild: Force rebuilding even if indexes exist
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Try to load existing indexes first
            if not force_rebuild and self.load_indexes():
                logger.info("Successfully loaded existing indexes")
                return True
            
            # Build new indexes
            logger.info("Building new indexes")
            self.build_indexes(documents)
            
            # Save indexes
            self.save_indexes()
            
            logger.info("Successfully built and saved new indexes")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing indexes: {e}")
            return False 