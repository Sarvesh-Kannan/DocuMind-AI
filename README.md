# Document Search and Summarization System

A comprehensive AI-powered system for searching through research papers and generating intelligent summaries using local LLM integration and advanced RAG techniques.

## 🎯 System Overview

This system implements a complete 5-phase pipeline for document search and summarization:

### Phase 1: Data Preparation & Processing
- **PDF Processing**: Extract text from PDFs using PyMuPDF
- **Text Cleaning**: Remove noise, normalize formatting
- **Chunking**: Split documents into overlapping chunks (512 tokens, 100 overlap)
- **Metadata Extraction**: Store document info, page numbers, chunk indices

### Phase 2: Local Embedding & Indexing
- **Local Embeddings**: Use sentence-transformers (all-MiniLM-L6-v2) - runs locally
- **Vector Storage**: Create FAISS index and save locally
- **Keyword Index**: Build TF-IDF vectorizer for traditional IR
- **Hybrid Search**: Combine semantic + keyword search with weighted scoring

### Phase 3: Local LLM Integration
- **Ollama Setup**: Connect to your local Deepseek-R1:8b
- **RAG Pipeline**: Retrieve → Context Preparation → Local LLM Summarization
- **Prompt Engineering**: Optimize prompts for summarization tasks

### Phase 4: Evaluation Framework
- **Test Set Creation**: Generate synthetic queries for each document
- **Retrieval Metrics**: Precision@K, Recall@K, MRR
- **Summary Quality**: ROUGE scores, semantic similarity
- **Performance Metrics**: Response time, memory usage

### Phase 5: Interface & Deployment
- **Streamlit Interface**: Local web interface
- **Query Processing**: Auto-suggestion, dynamic summary length
- **Result Display**: Pagination, source attribution

## 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PDF Papers    │───▶│  Phase 1: Data  │───▶│  Text Chunks    │
│   (30 files)    │    │  Processing     │    │  (512 tokens)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Streamlit UI   │◀───│  Phase 5:       │◀───│  Phase 2:       │
│                 │    │  Interface      │    │  Embeddings     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Evaluation     │◀───│  Phase 4:       │◀───│  Phase 3:       │
│  Reports        │    │  Evaluation     │    │  Local LLM      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📋 Requirements

### System Requirements
- Python 3.8+
- 8GB+ RAM (recommended)
- GPU support (optional, for faster processing)
- Ollama installed and running locally

### Python Dependencies
```
torch>=2.0.0
transformers>=4.30.0
sentence-transformers>=2.2.0
PyMuPDF>=1.23.0
faiss-cpu>=1.7.0
scikit-learn>=1.3.0
numpy>=1.24.0
pandas>=2.0.0
streamlit>=1.28.0
tqdm>=4.65.0
rouge-score>=0.1.2
nltk>=3.8.0
ollama>=0.1.0
requests>=2.31.0
```

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd document-search-summarizer
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install Ollama
```bash
# Download and install Ollama from https://ollama.ai
# Then pull the required model:
ollama pull deepseek-coder:6.7b
```

### 5. Download NLTK Data
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

## 📁 Project Structure

```
document-search-summarizer/
├── arxiv_papers/              # Research papers (PDF files)
├── embeddings/                # Generated embeddings (auto-created)
├── models/                    # Model cache (auto-created)
├── results/                   # Evaluation results (auto-created)
├── config.py                  # Configuration settings
├── phase1_data_processor.py   # Phase 1: Data processing
├── phase2_embedding_indexer.py # Phase 2: Embedding & indexing
├── phase3_local_llm.py        # Phase 3: Local LLM integration
├── phase4_evaluator.py        # Phase 4: Evaluation framework
├── phase5_streamlit_app.py    # Phase 5: Streamlit interface
├── main_system.py             # Main system orchestrator
├── requirements.txt           # Python dependencies
└── README.md                 # This file
```

## 🎮 Usage

### Quick Start
```bash
# Run the complete system
python main_system.py
```

### Streamlit Interface
```bash
# Run the Streamlit interface directly
streamlit run phase5_streamlit_app.py
```

### Step-by-Step Usage

1. **Prepare Documents**: Place your PDF research papers in the `arxiv_papers/` directory

2. **Start Ollama**: Ensure Ollama is running with the required model
   ```bash
   ollama serve
   ```

3. **Launch the System**: Run the main system
   ```bash
   python main_system.py
   ```

4. **Initialize System**: The system will automatically:
   - Process all PDF documents (Phase 1)
   - Generate embeddings and build indexes (Phase 2)
   - Test local LLM connection (Phase 3)
   - Launch Streamlit interface (Phase 5)

5. **Search and Summarize**: Use the web interface to:
   - Enter queries
   - Choose search type (semantic/keyword/hybrid)
   - Select summary length
   - View results with source attribution

## 🔧 Configuration

### Model Settings (`config.py`)
```python
# Phase 1: Data Processing
CHUNK_SIZE = 512  # tokens
CHUNK_OVERLAP = 100  # tokens

# Phase 2: Embedding & Indexing
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HYBRID_WEIGHT_SEMANTIC = 0.7
HYBRID_WEIGHT_KEYWORD = 0.3

# Phase 3: Local LLM
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "deepseek-coder:6.7b"

# Phase 4: Evaluation
TEST_SET_RATIO = 0.2
```

### Summary Types
- **Short**: ~100 words
- **Medium**: ~200 words  
- **Long**: ~300 words

## 📊 Evaluation Metrics

### Retrieval Performance
- **Precision@K**: Accuracy of top-K retrieved documents
- **Recall@K**: Completeness of relevant documents in top-K
- **MRR**: Mean Reciprocal Rank for ranking quality

### Summary Quality
- **ROUGE-1**: Unigram overlap between generated and reference summaries
- **ROUGE-2**: Bigram overlap between generated and reference summaries
- **ROUGE-L**: Longest common subsequence overlap
- **Semantic Similarity**: Word overlap and semantic coherence

### System Performance
- **Search Time**: Average time for document retrieval
- **Summary Time**: Average time for summary generation
- **Memory Usage**: System memory consumption
- **Total Response Time**: End-to-end processing time

## 🎯 Key Features

### Advanced Search Capabilities
- ✅ **Semantic Search**: Uses sentence transformers for understanding query intent
- ✅ **Keyword Search**: Traditional TF-IDF based search
- ✅ **Hybrid Search**: Combines semantic and keyword search with weighted scoring
- ✅ **FAISS Indexing**: Fast and scalable similarity search
- ✅ **Local Storage**: Embeddings stored locally for privacy and speed

### Intelligent Summarization
- ✅ **Local LLM Integration**: Uses Ollama with Deepseek model
- ✅ **RAG Pipeline**: Retrieve → Context Preparation → Summarization
- ✅ **Prompt Engineering**: Optimized prompts for different summary types
- ✅ **Context-Aware Summaries**: Summaries that directly address user queries
- ✅ **Multiple Length Options**: Short, medium, and long summary types

### Comprehensive Evaluation
- ✅ **Synthetic Query Generation**: Automatic test query creation
- ✅ **Retrieval Metrics**: Precision@K, Recall@K, MRR evaluation
- ✅ **Summary Quality**: ROUGE scores and semantic similarity
- ✅ **Performance Metrics**: Response time and memory usage tracking
- ✅ **Automated Testing**: Test set creation and evaluation

### User-Friendly Interface
- ✅ **Streamlit Web Interface**: Easy-to-use web application
- ✅ **Real-time Search**: Instant search results and summaries
- ✅ **Query Suggestions**: AI-generated query suggestions
- ✅ **Pagination**: Navigate through multiple search results
- ✅ **Source Attribution**: Clear document and page references
- ✅ **Performance Monitoring**: Real-time system statistics

## 🔍 Example Queries

Try these example queries to test the system:

- **"What is machine learning?"**
- **"How do neural networks work?"**
- **"What are the challenges in deep learning?"**
- **"Explain quantum computing algorithms"**
- **"What is the future of AI?"**
- **"What methods are used in computer vision?"**
- **"What are the applications of natural language processing?"**

## 🛠️ Troubleshooting

### Common Issues

1. **Ollama Connection Issues**
   ```bash
   # Ensure Ollama is running
   ollama serve
   
   # Check available models
   ollama list
   
   # Pull the required model
   ollama pull deepseek-coder:6.7b
   ```

2. **Memory Issues**
   - Reduce `BATCH_SIZE` in config.py
   - Use smaller chunk sizes
   - Process documents in batches

3. **PDF Extraction Issues**
   - Ensure PDFs are text-based (not scanned images)
   - Check file permissions
   - Verify PyMuPDF installation

4. **Model Download Issues**
   - Check internet connection
   - Clear model cache: `rm -rf ~/.cache/huggingface/`
   - Use smaller models if needed

### Error Messages

- **"Ollama not available"**: Ensure Ollama is running and model is downloaded
- **"No documents found"**: Check document directory and file formats
- **"System not initialized"**: Run system initialization first
- **"CUDA out of memory"**: Reduce batch size or use CPU

## 📈 Performance Optimization

### For Large Datasets
- Use smaller chunk sizes in `config.py`
- Reduce batch size for memory efficiency
- Process documents in batches
- Use CPU instead of GPU if memory is limited

### For Better Search Results
- Use specific, detailed queries
- Try different search types (semantic/keyword/hybrid)
- Adjust similarity thresholds
- Increase number of results for better coverage

### For Faster Processing
- Enable GPU acceleration if available
- Use smaller embedding models
- Reduce embedding dimensions
- Optimize chunk overlap settings

## 🧪 Testing

### Automated Testing
```bash
# Run system evaluation
python -c "from main_system import MainSystem; s = MainSystem(); s.initialize_complete_system(); print(s.run_phase4_evaluation())"
```

### Manual Testing
1. Initialize the system
2. Test various search queries
3. Verify summary quality
4. Check evaluation metrics
5. Monitor performance

## 📊 Evaluation Results

The system provides comprehensive evaluation reports including:

- **Retrieval Performance**: Precision@K, Recall@K, MRR metrics
- **Summary Quality**: ROUGE scores and semantic similarity
- **System Performance**: Response times and memory usage
- **Overall Assessment**: Combined system score and recommendations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Sentence Transformers**: For semantic embeddings
- **FAISS**: For efficient similarity search
- **Ollama**: For local LLM integration
- **Streamlit**: For the web interface
- **PyMuPDF**: For PDF processing
- **Hugging Face**: For transformer models

## 📞 Support

For questions or issues:
1. Check the troubleshooting section
2. Review the configuration options
3. Open an issue on GitHub
4. Contact the development team

---

**Built with ❤️ for efficient document search and summarization using local AI**

**Complete 5-phase system with advanced RAG capabilities and local LLM integration** 