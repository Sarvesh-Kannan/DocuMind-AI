"""
Phase 1: Data Preparation & Processing
- PDF Processing: Extract text from PDFs using PyMuPDF
- Text Cleaning: Remove noise, normalize formatting
- Chunking: Split documents into overlapping chunks (512 tokens, 100 overlap)
- Metadata Extraction: Store document info, page numbers, chunk indices
"""
import re
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import fitz  # PyMuPDF
import pandas as pd
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

from config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP, SUPPORTED_FORMATS

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

logger = logging.getLogger(__name__)

class Phase1DataProcessor:
    """Phase 1: Data Preparation & Processing"""
    
    def __init__(self):
        """Initialize the data processor"""
        pass
    
    def extract_text_from_pdf(self, pdf_path: Path) -> List[Dict]:
        """
        Extract text from PDF using PyMuPDF with page-level metadata
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of dictionaries with text and metadata for each page
        """
        pages_data = []
        
        try:
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                
                if text.strip():
                    pages_data.append({
                        'page_number': page_num + 1,
                        'text': text,
                        'file_path': str(pdf_path),
                        'file_name': pdf_path.stem
                    })
            
            doc.close()
            logger.info(f"Extracted {len(pages_data)} pages from {pdf_path.name}")
            
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
        
        return pages_data
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text formatting
        
        Args:
            text: Raw text input
            
        Returns:
            Cleaned text
        """
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
        
        # Normalize quotes and dashes
        text = re.sub(r'["""]', '"', text)
        text = re.sub(r"[''']", "'", text)
        text = re.sub(r'–|—', '-', text)
        
        # Remove page numbers and headers/footers
        text = re.sub(r'\b\d+\s*$', '', text, flags=re.MULTILINE)
        
        return text
    
    def tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text into tokens for chunking
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        return word_tokenize(text)
    
    def chunk_text_with_overlap(self, text: str, chunk_size: int = CHUNK_SIZE, 
                               overlap: int = CHUNK_OVERLAP) -> List[Dict]:
        """
        Split text into overlapping chunks with token-level precision
        
        Args:
            text: Input text
            chunk_size: Maximum tokens per chunk
            overlap: Number of tokens to overlap
            
        Returns:
            List of dictionaries containing chunks and metadata
        """
        # Tokenize the text
        tokens = self.tokenize_text(text)
        
        if len(tokens) <= chunk_size:
            return [{
                'text': text,
                'tokens': tokens,
                'token_count': len(tokens),
                'chunk_id': 0
            }]
        
        chunks = []
        start = 0
        
        while start < len(tokens):
            end = start + chunk_size
            
            # Get tokens for this chunk
            chunk_tokens = tokens[start:end]
            
            # Convert tokens back to text
            chunk_text = ' '.join(chunk_tokens)
            
            chunks.append({
                'text': chunk_text,
                'tokens': chunk_tokens,
                'token_count': len(chunk_tokens),
                'chunk_id': len(chunks),
                'start_token': start,
                'end_token': end
            })
            
            # Move start position with overlap
            start = end - overlap
            if start >= len(tokens):
                break
        
        return chunks
    
    def process_document(self, file_path: Path) -> List[Dict]:
        """
        Process a single document: extract, clean, chunk, and add metadata
        
        Args:
            file_path: Path to document file
            
        Returns:
            List of dictionaries containing processed chunks with metadata
        """
        if file_path.suffix.lower() not in SUPPORTED_FORMATS:
            logger.warning(f"Unsupported file format: {file_path}")
            return []
        
        try:
            # Extract text from PDF with page metadata
            pages_data = self.extract_text_from_pdf(file_path)
            
            if not pages_data:
                logger.warning(f"No text extracted from {file_path}")
                return []
            
            processed_chunks = []
            
            for page_data in pages_data:
                # Clean text
                cleaned_text = self.clean_text(page_data['text'])
                
                if not cleaned_text.strip():
                    continue
                
                # Chunk text with overlap
                chunks = self.chunk_text_with_overlap(cleaned_text)
                
                # Add page-level metadata to each chunk
                for chunk in chunks:
                    chunk.update({
                        'file_path': page_data['file_path'],
                        'file_name': page_data['file_name'],
                        'page_number': page_data['page_number'],
                        'total_pages': len(pages_data)
                    })
                    processed_chunks.append(chunk)
            
            logger.info(f"Processed {file_path.name}: {len(processed_chunks)} chunks")
            return processed_chunks
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return []
    
    def process_corpus(self, data_dir: Path = DATA_DIR) -> pd.DataFrame:
        """
        Process entire corpus and return DataFrame with all chunks and metadata
        
        Args:
            data_dir: Directory containing documents
            
        Returns:
            DataFrame with processed chunks and comprehensive metadata
        """
        all_chunks = []
        
        # Get all supported files
        files = []
        for ext in SUPPORTED_FORMATS:
            files.extend(data_dir.glob(f"*{ext}"))
        
        logger.info(f"Found {len(files)} PDF files to process")
        
        # Process each file
        for file_path in tqdm(files, desc="Processing PDF documents"):
            chunks = self.process_document(file_path)
            all_chunks.extend(chunks)
        
        # Create DataFrame
        df = pd.DataFrame(all_chunks)
        
        if not df.empty:
            logger.info(f"Phase 1 completed:")
            logger.info(f"  - Total documents processed: {df['file_name'].nunique()}")
            logger.info(f"  - Total chunks created: {len(df)}")
            logger.info(f"  - Average tokens per chunk: {df['token_count'].mean():.1f}")
            logger.info(f"  - Total pages processed: {df['page_number'].sum()}")
        
        return df
    
    def create_test_set(self, df: pd.DataFrame, test_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create train/test split for evaluation
        
        Args:
            df: DataFrame with processed chunks
            test_ratio: Ratio of data to use for testing
            
        Returns:
            Tuple of (train_df, test_df)
        """
        # Group by file to ensure documents stay together
        file_groups = df.groupby('file_name')
        
        train_chunks = []
        test_chunks = []
        
        for file_name, group in file_groups:
            if len(group) == 1:
                # Single chunk document - add to train
                train_chunks.append(group)
            else:
                # Multiple chunks - split proportionally
                n_test = max(1, int(len(group) * test_ratio))
                test_indices = group.sample(n=n_test, random_state=42).index
                
                train_group = group[~group.index.isin(test_indices)]
                test_group = group[group.index.isin(test_indices)]
                
                train_chunks.append(train_group)
                test_chunks.append(test_group)
        
        train_df = pd.concat(train_chunks, ignore_index=True) if train_chunks else pd.DataFrame()
        test_df = pd.concat(test_chunks, ignore_index=True) if test_chunks else pd.DataFrame()
        
        logger.info(f"Test set created: Train={len(train_df)} chunks, Test={len(test_df)} chunks")
        
        return train_df, test_df 