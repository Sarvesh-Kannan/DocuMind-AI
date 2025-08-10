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

from config import OLLAMA_BASE_URL, OLLAMA_MODEL, MAX_SUMMARY_LENGTH, SUMMARY_LENGTHS, SUPPORTED_LANGUAGES
from translation_service import TranslationService

logger = logging.getLogger(__name__)

class Phase3LocalLLM:
    """Phase 3: Local LLM Integration with Ollama"""
    
    def __init__(self):
        """Initialize the local LLM system"""
        self.base_url = OLLAMA_BASE_URL
        self.model = OLLAMA_MODEL
        self.is_available = False
        self.translation_service = TranslationService()
        self._check_ollama_availability()
        logger.info(f"Translation service initialized with {len(SUPPORTED_LANGUAGES)} supported languages")
    
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
        
        # Format context with source information
        context_text = ""
        for i, doc in enumerate(context, 1):
            source_info = f"[Source: {doc.get('file_name', 'Unknown')}, Page {doc.get('page_number', 'N/A')}]"
            context_text += f"Document {i} {source_info}:\n{doc['text']}\n\n"
        
        # Create role-based length guidance with significant differences
        length_guidance = {
            "short": "You are a concise expert who answers in EXACTLY ONE sentence. Give ONLY the most essential answer. No elaboration, no examples, no context - just the core answer in one complete sentence.",
            "medium": "You are an informative teacher who explains clearly in EXACTLY 3-4 sentences. Provide the main answer plus key supporting details. Include brief context but stay focused.", 
            "long": "You are a comprehensive researcher who provides thorough analysis in 6-8 sentences. Include the full answer, detailed context, examples, implications, and comprehensive explanations."
        }
        
        # Create query-focused prompt with strict length control
        length_instruction = length_guidance.get(summary_type, length_guidance["medium"])
        
        prompt = f"""You are a helpful AI assistant that answers specific questions based on provided documents.

QUESTION: {query}

CRITICAL INSTRUCTIONS:
- Answer ONLY the specific question being asked - "{query}"
- {length_instruction}
- STRICTLY FOLLOW THE LENGTH REQUIREMENT - this is mandatory
- Use ONLY information from the provided documents below
- Focus your entire response on addressing this exact question
- If the documents don't contain information to answer this specific question, state this clearly
- Be precise and directly relevant to the question asked

RELEVANT DOCUMENTS:
{context_text}

RESPONSE (Answer the question "{query}" specifically with the required length):"""
        
        return prompt
    
    
    def _call_ollama_with_retry(self, prompt: str, max_tokens: int = None, max_retries: int = 2) -> str:
        """Call Ollama with retry logic for better reliability"""
        for attempt in range(max_retries + 1):
            try:
                result = self._call_ollama(prompt, max_tokens)
                if not result.startswith("Error:") and len(result.strip()) > 20:
                    return result
                elif attempt < max_retries:
                    logger.warning(f"Retry {attempt + 1}/{max_retries}: Previous attempt returned: {result[:100]}...")
                    import time
                    time.sleep(2)  # Wait before retry
                else:
                    return result
            except Exception as e:
                if attempt < max_retries:
                    logger.warning(f"Retry {attempt + 1}/{max_retries}: Error: {e}")
                    import time
                    time.sleep(2)
                else:
                    return f"Error after {max_retries + 1} attempts: {str(e)}"
        return "Error: Maximum retries exceeded"

    
    def _call_ollama_simple(self, prompt: str, max_tokens: int = None) -> str:
        """Simple Ollama call with basic retry"""
        if not self.is_available:
            return "Error: Ollama not available"
        
        for attempt in range(3):  # 3 attempts
            try:
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,  # Lower temperature for more consistent responses
                        "top_p": 0.9,
                        "num_predict": max_tokens or 200  # Allow longer responses for better quality
                    }
                }
                
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=60  # Shorter timeout per attempt
                )
                
                if response.status_code == 200:
                    result = response.json()
                    raw_response = result.get('response', '').strip()
                    if len(raw_response) > 10:  # Basic validation
                        cleaned_response = self._clean_response(raw_response)
                        return cleaned_response
                
                if attempt < 2:  # Don't sleep on last attempt
                    import time
                    time.sleep(1)
                    
            except Exception as e:
                if attempt < 2:
                    import time
                    time.sleep(1)
                else:
                    logger.error(f"All attempts failed: {e}")
                    return "The AI model is currently unavailable. Please try a simpler query or try again later."
        
        return "Unable to generate response. Please try again with a shorter query."

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
                timeout=120  # Increased timeout for complex queries
            )
            
            if response.status_code == 200:
                result = response.json()
                raw_response = result.get('response', '').strip()
                logger.info(f"Raw LLM response: {raw_response[:200]}...")
                cleaned_response = self._clean_response(raw_response)
                logger.info(f"Cleaned response: {cleaned_response[:200]}...")
                return cleaned_response
            elif response.status_code == 500:
                # Ollama 500 error - try to get error details
                try:
                    error_data = response.json()
                    error_msg = error_data.get('error', 'Internal server error')
                    logger.error(f"Ollama internal error: {error_msg}")
                    # Return a user-friendly message
                    return "The AI model encountered an issue processing your request. Please try with a shorter or simpler query."
                except:
                    logger.error(f"Ollama API error: {response.status_code} (no error details)")
                    return "The AI model is temporarily unavailable. Please try again in a moment."
            else:
                logger.error(f"Ollama API error: {response.status_code}")
                try:
                    error_details = response.text
                    logger.error(f"Error details: {error_details}")
                except:
                    pass
                return f"AI service error: {response.status_code}. Please try again."
                
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
        
        # Store original response for debugging
        original_response = response
        
        # Remove thinking tags and internal monologue
        if "<think>" in response:
            # Extract content after </think> tag
            if "</think>" in response:
                response = response.split("</think>")[-1].strip()
            else:
                # If no closing tag, remove everything before the first meaningful content
                lines = response.split('\n')
                cleaned_lines = []
                skip_until_meaningful = True
                for line in lines:
                    line_stripped = line.strip()
                    if line_stripped and not line_stripped.startswith('<think'):
                        skip_until_meaningful = False
                    if not skip_until_meaningful and line_stripped:
                        cleaned_lines.append(line)
                response = '\n'.join(cleaned_lines)
        
        # Remove only explicit thinking patterns, not all document mentions
        thinking_patterns = [
            r'<think>.*?</think>',
            r'^Okay, so I need to.*?\.',
            r'^Alright, I need to.*?\.',
            r'^First, I\'ll.*?\.',
            r'^Let me start by.*?\.',
            r'^Let me analyze.*?\.',
            r'^Looking at.*?\.'
        ]
        
        for pattern in thinking_patterns:
            response = re.sub(pattern, '', response, flags=re.DOTALL | re.MULTILINE)
        
        # Remove any remaining XML-like tags
        response = re.sub(r'<[^>]+>', '', response)
        
        # Clean up extra whitespace and normalize
        response = re.sub(r'\n\s*\n', '\n\n', response)
        response = response.strip()
        
        # Preserve content that directly answers the query, only remove thinking patterns
        lines = response.split('\n')
        filtered_lines = []
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            # Only filter obvious thinking content at the beginning
            if (i < 2 and any(phrase in line_stripped.lower()[:40] for phrase in 
                ['okay, so i need', 'alright, i need', 'let me analyze', 'first, i\'ll'])):
                continue
            # Keep lines that seem to contain actual content
            if line_stripped:
                filtered_lines.append(line)
        
        if filtered_lines:
            response = '\n'.join(filtered_lines)
        
        response = response.strip()
        
        # If response is too short after cleaning, check original
        if len(response.strip()) < 30:
            # Try to extract meaningful content from original
            original_lines = original_response.split('\n')
            meaningful_lines = []
            for line in original_lines:
                line_stripped = line.strip()
                if (len(line_stripped) > 20 and 
                    not line_stripped.startswith('<') and 
                    not any(phrase in line_stripped.lower()[:30] for phrase in 
                           ['<think>', 'okay, so', 'alright', 'first, i', 'let me'])):
                    meaningful_lines.append(line_stripped)
            
            if meaningful_lines:
                response = ' '.join(meaningful_lines[:3])  # Take first 3 meaningful lines
            
            # Final fallback
            if len(response.strip()) < 30:
                return "Based on the provided documents, I was unable to generate a comprehensive summary. Please try rephrasing your query or using different search parameters."
        
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
        
        return response
    
    def summarize_with_rag(self, query: str, retrieved_docs: List[Dict], 
                          summary_type: str = "medium", target_language: str = "english") -> Dict:
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
            
            # Get appropriate token limit based on summary type
            max_tokens = SUMMARY_LENGTHS.get(summary_type, SUMMARY_LENGTHS["medium"])
            
            # Call local LLM with length-specific token limit
            start_time = time.time()
            raw_summary = self._call_ollama_simple(prompt, max_tokens)
            
            # Enforce length constraints post-processing
            summary = self._enforce_length_constraints(raw_summary, summary_type)
            generation_time = time.time() - start_time
            
            # Get summary statistics
            summary_stats = self._get_summary_statistics(summary)
            
            # Handle translation if requested
            translation_result = None
            if target_language.lower() != 'english':
                try:
                    translation_start = time.time()
                    translation_result = self.translate_summary(summary, target_language)
                    translation_time = time.time() - translation_start
                    
                    if translation_result['status'] == 'success':
                        logger.info(f"Translation to {target_language} completed in {translation_time:.2f}s")
                    else:
                        logger.warning(f"Translation to {target_language} failed: {translation_result.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    logger.error(f"Translation error: {e}")
                    translation_result = {
                        'status': 'error',
                        'error': f'Translation failed: {str(e)}',
                        'original_text': summary,
                        'translated_text': summary,
                        'target_language': target_language
                    }
            
            result = {
                'summary': summary,
                'query': query,
                'summary_type': summary_type,
                'target_language': target_language,
                'num_documents': len(retrieved_docs),
                'llm_model': self.model,
                'generation_time': generation_time,
                'summary_stats': summary_stats,
                'status': 'success'
            }
            
            # Add translation result if requested
            if translation_result:
                result['translation'] = translation_result
                if translation_result['status'] == 'success':
                    result['translated_summary'] = translation_result['translated_text']
                else:
                    result['translated_summary'] = summary  # Fallback to original
            
            return result
            
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
    
    def _enforce_length_constraints(self, response: str, summary_type: str) -> str:
        """Enforce strict length constraints based on summary type"""
        if not response or not response.strip():
            return response
            
        # Split into sentences
        sentences = [s.strip() for s in response.split('.') if s.strip()]
        
        if summary_type == "short":
            # Take only the first sentence, ensure it's complete
            if sentences:
                return sentences[0] + '.'
            else:
                return response
                
        elif summary_type == "medium":
            # Take first 3-4 sentences maximum
            selected_sentences = sentences[:4]
            if selected_sentences:
                return '. '.join(selected_sentences) + '.'
            else:
                return response
                
        elif summary_type == "long":
            # Take up to 8 sentences, ensure completeness
            selected_sentences = sentences[:8]
            if selected_sentences:
                return '. '.join(selected_sentences) + '.'
            else:
                return response
        
        return response
    
    def get_supported_languages(self) -> Dict[str, str]:
        """
        Get list of supported languages for translation
        
        Returns:
            Dict mapping language names to language codes
        """
        return self.translation_service.get_supported_languages()
    
    def translate_summary(self, summary: str, target_language: str) -> Dict:
        """
        Translate generated summary to target Indian language
        
        Args:
            summary: Generated summary text to translate
            target_language: Target language name (e.g., 'hindi', 'tamil')
            
        Returns:
            Dict containing translation result and metadata
        """
        return self.translation_service.translate_text(summary, target_language)
    
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