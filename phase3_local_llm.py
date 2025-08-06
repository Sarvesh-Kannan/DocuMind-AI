"""
Phase 3: Local LLM Integration
- Ollama Setup: Connect to your local Deepseek-R1:8b
- RAG Pipeline: Retrieve → Context Preparation → Local LLM Summarization
- Prompt Engineering: Optimize prompts for summarization tasks
"""
import logging
import requests
import json
from typing import List, Dict, Optional
import time

from config import OLLAMA_BASE_URL, OLLAMA_MODEL, MAX_SUMMARY_LENGTH, SUMMARY_LENGTHS

logger = logging.getLogger(__name__)

class Phase3LocalLLM:
    """Phase 3: Local LLM Integration with Ollama"""
    
    def __init__(self):
        """Initialize the local LLM system"""
        self.base_url = OLLAMA_BASE_URL
        self.model = OLLAMA_MODEL
        self.is_available = False
        self._check_ollama_availability()
    
    def _check_ollama_availability(self):
        """Check if Ollama is running and model is available"""
        try:
            # Check if Ollama is running
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [model['name'] for model in models]
                
                if self.model in model_names:
                    self.is_available = True
                    logger.info(f"Ollama model '{self.model}' is available")
                else:
                    logger.warning(f"Model '{self.model}' not found. Available models: {model_names}")
                    # Try to pull the model
                    self._pull_model()
            else:
                logger.error(f"Ollama not responding: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Cannot connect to Ollama: {e}")
            logger.info("Please ensure Ollama is running: ollama serve")
    
    def _pull_model(self):
        """Pull the specified model from Ollama"""
        try:
            logger.info(f"Pulling model '{self.model}' from Ollama...")
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": self.model},
                timeout=300  # 5 minutes timeout for model download
            )
            
            if response.status_code == 200:
                self.is_available = True
                logger.info(f"Model '{self.model}' pulled successfully")
            else:
                logger.error(f"Failed to pull model: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error pulling model: {e}")
    
    def _generate_prompt(self, query: str, context: List[Dict], summary_type: str = "medium") -> str:
        """
        Generate optimized prompt for summarization
        
        Args:
            query: User query
            context: Retrieved documents
            summary_type: Type of summary (short/medium/long)
            
        Returns:
            Formatted prompt for LLM
        """
        # Get summary length
        max_length = SUMMARY_LENGTHS.get(summary_type, MAX_SUMMARY_LENGTH)
        
        # Format context
        context_text = ""
        for i, doc in enumerate(context, 1):
            context_text += f"Document {i}:\n{doc['text']}\n\n"
        
        # Create optimized prompt
        prompt = f"""Based on the provided documents, answer this question: {query}

Documents:
{context_text}

Answer:"""
        
        return prompt
    
    def _call_ollama(self, prompt: str, max_tokens: int = None) -> str:
        """
        Call Ollama API for text generation
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text response
        """
        if not self.is_available:
            return "Error: Ollama not available"
        
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.5,
                    "top_p": 0.9,
                    "num_predict": max_tokens or MAX_SUMMARY_LENGTH
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                raw_response = result.get('response', '').strip()
                logger.info(f"Raw LLM response: {raw_response[:200]}...")
                cleaned_response = self._clean_response(raw_response)
                logger.info(f"Cleaned response: {cleaned_response[:200]}...")
                return cleaned_response
            else:
                logger.error(f"Ollama API error: {response.status_code}")
                return f"Error: API returned {response.status_code}"
                
        except Exception as e:
            logger.error(f"Error calling Ollama: {e}")
            return f"Error: {str(e)}"
    
    def _clean_response(self, response: str) -> str:
        """
        Clean the response by removing thinking tags and formatting
        
        Args:
            response: Raw response from LLM
            
        Returns:
            Cleaned response
        """
        import re
        
        # Remove thinking tags and internal monologue
        if "<think>" in response:
            # Extract content after </think> tag
            if "</think>" in response:
                response = response.split("</think>")[-1].strip()
            else:
                # If no closing tag, remove everything before the first meaningful content
                lines = response.split('\n')
                cleaned_lines = []
                for line in lines:
                    if line.strip() and not line.strip().startswith('<'):
                        cleaned_lines.append(line)
                response = '\n'.join(cleaned_lines)
        
        # Remove thinking patterns like "Alright, I need to..." or "First, I'll..."
        thinking_patterns = [
            r'<think>.*?</think>',
            r'Okay, so I need to.*?\.',
            r'Alright, I need to.*?\.',
            r'First, I\'ll.*?\.',
            r'Let me start by.*?\.',
            r'Document \d+ talks about.*?\.',
            r'Document \d+ seems.*?\.',
        ]
        
        for pattern in thinking_patterns:
            response = re.sub(pattern, '', response, flags=re.DOTALL)
        
        # Remove any remaining XML-like tags
        response = re.sub(r'<[^>]+>', '', response)
        
        # Clean up extra whitespace and normalize
        response = re.sub(r'\n\s*\n', '\n\n', response)
        response = response.strip()
        
        # If response is too short after cleaning, return a fallback
        if len(response.strip()) < 20:
            return "Unable to generate a proper summary. Please try a different query or search type."
        
        # Ensure the response ends with a complete sentence
        response = response.strip()
        if response and not response.endswith(('.', '!', '?')):
            # Find the last complete sentence
            sentences = response.split('.')
            if len(sentences) > 1:
                # Keep all complete sentences
                complete_sentences = [s.strip() for s in sentences[:-1] if s.strip()]
                if complete_sentences:
                    response = '. '.join(complete_sentences) + '.'
        
        # If the response still contains thinking content, try to extract the actual answer
        if any(phrase in response.lower() for phrase in ['document', 'talks about', 'seems', 'mentions']):
            # Try to find the actual answer after the thinking part
            lines = response.split('\n')
            answer_lines = []
            for line in lines:
                line = line.strip()
                if line and not any(phrase in line.lower() for phrase in ['document', 'talks about', 'seems', 'mentions', 'okay', 'first', 'let me']):
                    answer_lines.append(line)
            
            if answer_lines:
                response = ' '.join(answer_lines)
                
            # Ensure the response ends with a complete sentence
            if response and not response.strip().endswith(('.', '!', '?')):
                # Find the last complete sentence
                sentences = response.split('.')
                if len(sentences) > 1:
                    # Keep all complete sentences
                    complete_sentences = [s.strip() for s in sentences[:-1] if s.strip()]
                    if complete_sentences:
                        response = '. '.join(complete_sentences) + '.'
        
        return response
    
    def summarize_with_rag(self, query: str, retrieved_docs: List[Dict], 
                          summary_type: str = "medium") -> Dict:
        """
        Perform RAG-based summarization using local LLM
        
        Args:
            query: User query
            retrieved_docs: Retrieved documents from search
            summary_type: Type of summary (short/medium/long)
            
        Returns:
            Dictionary with summary and metadata
        """
        if not retrieved_docs:
            return {
                'summary': f"No relevant documents found for query: '{query}'",
                'query': query,
                'summary_type': summary_type,
                'num_documents': 0,
                'llm_model': self.model,
                'status': 'no_documents'
            }
        
        try:
            # Generate optimized prompt
            prompt = self._generate_prompt(query, retrieved_docs, summary_type)
            
            # Call local LLM
            start_time = time.time()
            summary = self._call_ollama(prompt)
            generation_time = time.time() - start_time
            
            # Get summary statistics
            summary_stats = self._get_summary_statistics(summary)
            
            return {
                'summary': summary,
                'query': query,
                'summary_type': summary_type,
                'num_documents': len(retrieved_docs),
                'llm_model': self.model,
                'generation_time': generation_time,
                'summary_stats': summary_stats,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Error in RAG summarization: {e}")
            return {
                'summary': f"Error generating summary: {str(e)}",
                'query': query,
                'summary_type': summary_type,
                'num_documents': len(retrieved_docs),
                'llm_model': self.model,
                'status': 'error',
                'error': str(e)
            }
    
    def _get_summary_statistics(self, summary: str) -> Dict:
        """
        Get statistics about the generated summary
        
        Args:
            summary: Generated summary text
            
        Returns:
            Dictionary with summary statistics
        """
        if not summary:
            return {}
        
        words = summary.split()
        sentences = summary.split('.')
        
        return {
            'word_count': len(words),
            'sentence_count': len([s for s in sentences if s.strip()]),
            'character_count': len(summary),
            'average_sentence_length': len(words) / max(1, len([s for s in sentences if s.strip()]))
        }
    
    def test_connection(self) -> Dict:
        """
        Test Ollama connection and model availability
        
        Returns:
            Dictionary with test results
        """
        test_prompt = "Please respond with 'Ollama is working correctly' if you can see this message."
        
        try:
            response = self._call_ollama(test_prompt, max_tokens=50)
            
            return {
                'available': self.is_available,
                'model': self.model,
                'test_response': response,
                'status': 'success' if 'Ollama is working' in response else 'error'
            }
            
        except Exception as e:
            return {
                'available': False,
                'model': self.model,
                'error': str(e),
                'status': 'error'
            }
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        try:
            response = requests.get(f"{self.base_url}/api/show", 
                                 json={"name": self.model}, timeout=10)
            
            if response.status_code == 200:
                model_info = response.json()
                return {
                    'name': model_info.get('name'),
                    'size': model_info.get('size'),
                    'modified_at': model_info.get('modified_at'),
                    'parameters': model_info.get('parameter_size')
                }
            else:
                return {'error': f"API returned {response.status_code}"}
                
        except Exception as e:
            return {'error': str(e)} 