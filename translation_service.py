"""
Translation Service using Sarvam AI API
Handles translation of text to Indian languages
"""
import logging
import requests
from typing import Dict
from config import SARVAM_API_KEY, SARVAM_BASE_URL, SARVAM_TIMEOUT, SUPPORTED_LANGUAGES

logger = logging.getLogger(__name__)

class TranslationService:
    def __init__(self):
        self.api_key = SARVAM_API_KEY
        self.base_url = SARVAM_BASE_URL
        self.timeout = SARVAM_TIMEOUT
        self.supported_languages = SUPPORTED_LANGUAGES
        
        logger.info(f"Translation service initialized with {len(self.supported_languages)} supported languages")
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get dictionary of supported languages and their codes"""
        return self.supported_languages.copy()
    
    def translate_text(self, text: str, target_language: str) -> Dict:
        """
        Translate text to target Indian language using Sarvam AI API
        
        Args:
            text: Text to translate
            target_language: Target language name (e.g., 'hindi', 'tamil')
            
        Returns:
            Dict with translation result and metadata
        """
        try:
            if not text or not text.strip():
                return {
                    'status': 'error',
                    'error': 'Empty text provided',
                    'original_text': text,
                    'translated_text': text,
                    'target_language': target_language
                }
            
            # Handle English case
            if target_language.lower() == "english":
                return {
                    'status': 'success',
                    'original_text': text,
                    'translated_text': text,
                    'target_language': 'english',
                    'source_language': 'en-IN',
                    'target_language_code': 'en-IN'
                }
            
            # Validate target language
            if target_language.lower() not in self.supported_languages:
                return {
                    'status': 'error',
                    'error': f'Unsupported language: {target_language}',
                    'original_text': text,
                    'translated_text': text,
                    'target_language': target_language
                }
            
            target_lang_code = self.supported_languages[target_language.lower()]
            
            # Prepare API request
            headers = {
                'api-subscription-key': self.api_key,
                'Content-Type': 'application/json'
            }
            
            payload = {
                'input': text,
                'source_language_code': 'en-IN',
                'target_language_code': target_lang_code
            }
            
            logger.info(f"Translating text to {target_language} ({target_lang_code})")
            
            # Make API call with retry logic
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    response = requests.post(
                        self.base_url,
                        headers=headers,
                        json=payload,
                        timeout=self.timeout
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        if 'translated_text' in result:
                            logger.info(f"Translation successful: {len(text)} chars -> {len(result['translated_text'])} chars")
                            return {
                                'status': 'success',
                                'original_text': text,
                                'translated_text': result['translated_text'],
                                'target_language': target_language,
                                'source_language': 'en-IN',
                                'target_language_code': target_lang_code,
                                'confidence_score': result.get('confidence_score', 'N/A')
                            }
                        else:
                            logger.error(f"Invalid response format: {result}")
                            return {
                                'status': 'error',
                                'error': 'Invalid response format from translation API',
                                'original_text': text,
                                'translated_text': text,
                                'target_language': target_language
                            }
                    else:
                        error_msg = f"Translation API error: {response.status_code}"
                        try:
                            error_detail = response.json().get('error', 'Unknown error')
                            error_msg += f" - {error_detail}"
                        except:
                            error_msg += f" - {response.text}"
                        
                        logger.error(error_msg)
                        
                        # Don't retry on client errors (4xx)
                        if response.status_code >= 400 and response.status_code < 500:
                            break
                            
                except requests.exceptions.Timeout:
                    logger.warning(f"Translation request timed out (attempt {attempt + 1}/{max_retries})")
                    if attempt == max_retries - 1:
                        return {
                            'status': 'error',
                            'error': f'Translation request timed out after {max_retries} attempts',
                            'original_text': text,
                            'translated_text': text,
                            'target_language': target_language
                        }
                    continue
                    
                except requests.exceptions.ConnectionError as e:
                    logger.warning(f"Connection error (attempt {attempt + 1}/{max_retries}): {e}")
                    if attempt == max_retries - 1:
                        return {
                            'status': 'error',
                            'error': 'Translation service temporarily unavailable',
                            'original_text': text,
                            'translated_text': text,
                            'target_language': target_language
                        }
                    continue
                    
                except Exception as e:
                    logger.error(f"Unexpected error during translation (attempt {attempt + 1}/{max_retries}): {e}")
                    if attempt == max_retries - 1:
                        return {
                            'status': 'error',
                            'error': f'Unexpected error: {str(e)}',
                            'original_text': text,
                            'translated_text': text,
                            'target_language': target_language
                        }
                    continue
            
            # If all retries failed
            return {
                'status': 'error',
                'error': 'Translation failed after multiple attempts',
                'original_text': text,
                'translated_text': text,
                'target_language': target_language
            }
            
        except Exception as e:
            logger.error(f"Critical error in translation service: {e}")
            return {
                'status': 'error',
                'error': f'Translation service error: {str(e)}',
                'original_text': text,
                'translated_text': text,
                'target_language': target_language
            }
    
    def is_api_available(self) -> bool:
        """Check if the Sarvam API is available"""
        try:
            test_result = self.translate_text("Hello", "hindi")
            return test_result['status'] == 'success'
        except Exception as e:
            logger.warning(f"API availability check failed: {e}")
            return False  # Return False to indicate unavailability 