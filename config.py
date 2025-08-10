"""
Configuration settings for DocuMind AI System
"""
import os

# Basic Configuration
CHUNK_SIZE = 512
CHUNK_OVERLAP = 100
EMBEDDING_DIMENSION = 384
MAX_CONTEXT_LENGTH = 2048

# File Processing
SUPPORTED_FILE_TYPES = ['pdf']
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

# Search Configuration
SIMILARITY_THRESHOLD = 0.1
DEFAULT_TOP_K = 5
SEARCH_TYPES = ["semantic", "keyword", "hybrid"]

# Ollama Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-r1:8b")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))  # seconds

# Summary Configuration
SUMMARY_TYPES = ["short", "medium", "long"]
SUMMARY_LENGTHS = {
    "short": 25,      # Extremely concise, 1 sentence only
    "medium": 100,    # Moderate response, 3-4 sentences
    "long": 300       # Comprehensive explanation, 6-8 sentences
}
MAX_SUMMARY_LENGTH = 400

# Translation Configuration (Sarvam API)
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "sk_inm4n58r_4NIPBvcjjYMhCZ1ryEXAOgqP")
SARVAM_BASE_URL = os.getenv("SARVAM_BASE_URL", "https://api.sarvam.ai/translate")
SARVAM_TIMEOUT = int(os.getenv("SARVAM_TIMEOUT", "30"))  # seconds

# Supported Indian languages for translation
SUPPORTED_LANGUAGES = {
    "english": "en-IN",
    "hindi": "hi-IN",
    "bengali": "bn-IN",
    "gujarati": "gu-IN",
    "kannada": "kn-IN",
    "malayalam": "ml-IN",
    "marathi": "mr-IN",
    "oriya": "od-IN",
    "punjabi": "pa-IN",
    "tamil": "ta-IN",
    "telugu": "te-IN"
}

# Model Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FALLBACK_EMBEDDING_MODEL = "sentence-transformers/paraphrase-MiniLM-L3-v2"

# Performance Configuration
BATCH_SIZE = 32
MAX_CONCURRENT_REQUESTS = 10

# Evaluation Configuration
EVALUATION_METRICS = ["precision", "recall", "mrr", "rouge", "semantic_similarity"]
PERFORMANCE_METRICS = ["response_time", "memory_usage", "accuracy"]

# Storage Configuration
DATA_DIR = "data"
EMBEDDINGS_DIR = os.path.join(DATA_DIR, "embeddings")
MODELS_DIR = os.path.join(DATA_DIR, "models")

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Security Configuration
CORS_ORIGINS = ["*"]  # In production, specify your frontend domain
MAX_UPLOAD_FILES = 10

# Feature Flags
ENABLE_TRANSLATION = True
ENABLE_EVALUATION = True
ENABLE_CACHING = True 