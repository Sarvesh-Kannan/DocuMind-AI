"""
Configuration for Document Search and Summarization System
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "arxiv_papers"
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create directories
for dir_path in [EMBEDDINGS_DIR, MODELS_DIR, RESULTS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Phase 1: Data Processing
CHUNK_SIZE = 512  # tokens
CHUNK_OVERLAP = 100  # tokens
SUPPORTED_FORMATS = [".pdf"]

# Phase 2: Embedding & Indexing
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
TOP_K_RESULTS = 5
SIMILARITY_THRESHOLD = 0.1  # Lowered for better retrieval
HYBRID_WEIGHT_SEMANTIC = 0.7
HYBRID_WEIGHT_KEYWORD = 0.3

# Phase 3: Local LLM
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "deepseek-r1:8b"
MAX_SUMMARY_LENGTH = 200
SUMMARY_TYPES = ["short", "medium", "long"]
SUMMARY_LENGTHS = {
    "short": 100,
    "medium": 200,
    "long": 300
}

# Phase 4: Evaluation
TEST_SET_RATIO = 0.2
EVALUATION_METRICS = ["rouge-1", "rouge-2", "rouge-l"]

# Processing
BATCH_SIZE = 8
NUM_WORKERS = 4
DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s" 