"""
DocuMind AI - Hugging Face Spaces Deployment
A simplified version for easy deployment on Hugging Face Spaces
"""

import gradio as gr
import os
import tempfile
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simplified imports for HF Spaces
try:
    from backend_api import app as fastapi_app
    BACKEND_AVAILABLE = True
except ImportError:
    BACKEND_AVAILABLE = False
    logger.warning("Backend not available, using demo mode")

def process_pdf_demo(files):
    """Demo function for PDF processing"""
    if not files:
        return "Please upload a PDF file", [], ""
    
    return (
        f"✅ Processed {len(files)} PDF(s) successfully!\n"
        f"📄 Files: {', '.join([f.name for f in files])}\n"
        f"🔍 Ready for search and analysis",
        ["What is the main topic?", "Summarize the key points", "What are the conclusions?"],
        "Upload completed. You can now search the documents."
    )

def search_documents_demo(query, search_type, summary_type):
    """Demo function for document search"""
    if not query:
        return "Please enter a search query", ""
    
    # Demo response
    demo_response = f"""
    **Search Results for:** "{query}"
    
    Based on the uploaded documents, here are the findings:
    
    **Summary ({summary_type}):**
    This is a demonstration of the DocuMind AI system. The actual deployment would process your PDF documents using advanced AI models to provide accurate search results and intelligent summaries.
    
    **Key Points:**
    • Advanced PDF processing with PyMuPDF
    • Semantic search using sentence transformers
    • AI-powered summarization with local LLM
    • Translation support for 11 Indian languages
    • Hybrid search combining semantic and keyword matching
    
    **Search Type:** {search_type}
    **Documents Found:** 3 relevant sections
    **Confidence Score:** 85%
    
    *Note: This is a demo response. Deploy the full system for actual document processing.*
    """
    
    return demo_response, f"Found 3 relevant sections for '{query}'"

def translate_text_demo(text, target_language):
    """Demo function for translation"""
    if not text:
        return "Please provide text to translate"
    
    if target_language == "english":
        return text
    
    return f"[{target_language.upper()} TRANSLATION]: {text}\n\n*Note: This is a demo. The full system uses Sarvam AI for accurate translation.*"

# Create Gradio interface
def create_gradio_app():
    with gr.Blocks(
        title="DocuMind AI - Document Search & Analysis",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            font-family: 'Inter', sans-serif;
        }
        .main-header {
            text-align: center;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        """
    ) as demo:
        
        # Header
        gr.HTML("""
        <div class="main-header">
            <h1>🧠 DocuMind AI</h1>
            <p>Intelligent Document Search & Analysis with AI Translation</p>
            <p><strong>Features:</strong> PDF Processing • Semantic Search • AI Summarization • Multi-language Translation</p>
        </div>
        """)
        
        # Status indicator
        status_text = "🟢 Demo Mode - Upload your own deployment for full functionality" if not BACKEND_AVAILABLE else "🟢 Full System Active"
        gr.HTML(f"<div style='text-align: center; padding: 1rem; background: #f0f9ff; border-radius: 8px; margin-bottom: 2rem;'>{status_text}</div>")
        
        with gr.Tab("📤 Document Upload"):
            gr.HTML("<h3>Upload PDF Documents</h3>")
            
            files_input = gr.File(
                label="Select PDF Files",
                file_count="multiple",
                file_types=[".pdf"],
                height=150
            )
            
            upload_btn = gr.Button("🚀 Process Documents", variant="primary", size="lg")
            
            upload_status = gr.Textbox(
                label="Processing Status",
                placeholder="Upload PDFs to see processing status...",
                lines=4,
                interactive=False
            )
            
            suggestions_output = gr.Textbox(
                label="Auto-generated Query Suggestions",
                placeholder="Suggested queries will appear here...",
                lines=3,
                interactive=False
            )
            
            upload_btn.click(
                process_pdf_demo,
                inputs=[files_input],
                outputs=[upload_status, suggestions_output, gr.Textbox(visible=False)]
            )
        
        with gr.Tab("🔍 Search & Analysis"):
            gr.HTML("<h3>Search Your Documents</h3>")
            
            with gr.Row():
                query_input = gr.Textbox(
                    label="Search Query",
                    placeholder="Ask questions about your documents...",
                    lines=2
                )
                
                with gr.Column(scale=1):
                    search_type = gr.Radio(
                        ["semantic", "keyword", "hybrid"],
                        label="Search Type",
                        value="hybrid"
                    )
                    
                    summary_type = gr.Radio(
                        ["short", "medium", "long"],
                        label="Summary Length",
                        value="medium"
                    )
            
            search_btn = gr.Button("🔎 Search Documents", variant="primary", size="lg")
            
            search_results = gr.Textbox(
                label="Search Results & AI Summary",
                placeholder="Search results will appear here...",
                lines=12,
                interactive=False
            )
            
            search_status = gr.Textbox(
                label="Search Status",
                placeholder="Search status...",
                lines=1,
                interactive=False
            )
            
            search_btn.click(
                search_documents_demo,
                inputs=[query_input, search_type, summary_type],
                outputs=[search_results, search_status]
            )
        
        with gr.Tab("🌐 Translation"):
            gr.HTML("<h3>Translate AI Summaries</h3>")
            
            translate_input = gr.Textbox(
                label="Text to Translate",
                placeholder="Enter or paste text from AI summaries...",
                lines=4
            )
            
            language_select = gr.Dropdown(
                choices=[
                    "english", "hindi", "bengali", "gujarati", "kannada",
                    "malayalam", "marathi", "oriya", "punjabi", "tamil", "telugu"
                ],
                label="Target Language",
                value="hindi"
            )
            
            translate_btn = gr.Button("🔄 Translate", variant="primary")
            
            translation_output = gr.Textbox(
                label="Translation Result",
                placeholder="Translation will appear here...",
                lines=6,
                interactive=False
            )
            
            translate_btn.click(
                translate_text_demo,
                inputs=[translate_input, language_select],
                outputs=[translation_output]
            )
        
        with gr.Tab("ℹ️ About & Deployment"):
            gr.HTML("""
            <h3>About DocuMind AI</h3>
            <p>DocuMind AI is an advanced document analysis system that combines:</p>
            <ul>
                <li><strong>PDF Processing:</strong> Extract and process text from PDF documents</li>
                <li><strong>Semantic Search:</strong> Find relevant information using AI embeddings</li>
                <li><strong>AI Summarization:</strong> Generate intelligent summaries with local LLM</li>
                <li><strong>Multi-language Translation:</strong> Translate to 11 Indian languages</li>
            </ul>
            
            <h3>🚀 Deploy Your Own Instance</h3>
            <p>This is a demo version. For full functionality:</p>
            <ol>
                <li><strong>GitHub:</strong> <a href="https://github.com/Sarvesh-Kannan/DocuMind-AI" target="_blank">Clone the repository</a></li>
                <li><strong>Railway:</strong> Deploy backend for FREE with $5 monthly credits</li>
                <li><strong>Vercel:</strong> Deploy frontend for FREE</li>
            </ol>
            
            <h3>🔧 Tech Stack</h3>
            <ul>
                <li>FastAPI + Python backend</li>
                <li>Sentence Transformers for embeddings</li>
                <li>FAISS for vector search</li>
                <li>Local LLM integration (Ollama)</li>
                <li>Sarvam AI for translation</li>
                <li>Modern web interface</li>
            </ul>
            
            <p><strong>Live Demo:</strong> <a href="https://documind-ai-rust.vercel.app" target="_blank">https://documind-ai-rust.vercel.app</a></p>
            """)
    
    return demo

# Create and launch the app
if __name__ == "__main__":
    demo = create_gradio_app()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    ) 