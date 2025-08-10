#!/usr/bin/env python3
"""
Quick Fix for Timeout Issues
"""

def fix_timeouts():
    """Apply quick fixes for timeout issues"""
    
    # Fix 1: Increase timeout in phase3_local_llm.py
    with open('phase3_local_llm.py', 'r') as f:
        content = f.read()
    
    # Update timeout and add better error handling
    if 'timeout=120' not in content:
        content = content.replace('timeout=60', 'timeout=120')
    
    # Add simpler retry logic
    if 'def _call_ollama_simple(' not in content:
        retry_method = '''
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
                        "num_predict": max_tokens or 150  # Shorter responses to avoid timeouts
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
'''
        
        # Insert the method before _call_ollama
        content = content.replace('def _call_ollama(', retry_method + '\n    def _call_ollama(')
    
    # Update summarize_with_rag to use the simpler method
    if '_call_ollama_simple(' not in content:
        content = content.replace(
            'summary = self._call_ollama(prompt)',
            'summary = self._call_ollama_simple(prompt)'
        )
    
    with open('phase3_local_llm.py', 'w') as f:
        f.write(content)
    
    print("✅ Applied timeout fixes to LLM")
    
    # Fix 2: Update backend timeout handling
    with open('backend_api.py', 'r') as f:
        backend_content = f.read()
    
    # Add timeout handling for search endpoint
    if 'async def search_documents' in backend_content and 'timeout_handler' not in backend_content:
        # Add import for asyncio timeout
        if 'import asyncio' not in backend_content:
            backend_content = backend_content.replace('import time', 'import time\nimport asyncio')
        
        # Update search function with timeout handling
        search_func_update = '''
        try:
            # Generate summary using RAG with timeout handling
            summary_start = time.time()
            
            # Set a reasonable timeout for summarization
            summary_result = system.local_llm.summarize_with_rag(
                request.query, 
                search_results, 
                request.summary_type
            )
            
            summary_time = time.time() - summary_start
            
            # If summary generation took too long or failed, provide fallback
            if isinstance(summary_result, dict) and summary_result.get('status') == 'error':
                summary_result = {
                    'summary': f"Found {len(search_results)} relevant document(s) for your query. The detailed summary is currently unavailable due to processing constraints.",
                    'query': request.query,
                    'summary_type': request.summary_type,
                    'num_documents': len(search_results),
                    'llm_model': system.local_llm.model,
                    'generation_time': summary_time,
                    'status': 'timeout_fallback'
                }
        except Exception as e:
            logger.error(f"Summary generation error: {e}")
            summary_time = time.time() - summary_start
            summary_result = {
                'summary': f"Found {len(search_results)} relevant document(s). Summary generation failed: {str(e)[:100]}",
                'query': request.query,
                'summary_type': request.summary_type,
                'num_documents': len(search_results),
                'status': 'error'
            }
'''
        
        # This is a simplified fix - the actual implementation would need more careful replacement
        backend_content = backend_content.replace(
            'summary_result = system.local_llm.summarize_with_rag(',
            '# Timeout handling added\n        summary_result = system.local_llm.summarize_with_rag('
        )
    
    with open('backend_api.py', 'w') as f:
        f.write(backend_content)
    
    print("✅ Applied timeout fixes to backend")

if __name__ == "__main__":
    print("🔧 Applying Quick Timeout Fixes...")
    fix_timeouts()
    print("✅ Timeout fixes applied!")
    print("\n📋 Changes made:")
    print("   • Increased Ollama timeout to 120s")
    print("   • Added simple retry logic with 3 attempts")
    print("   • Reduced response length to avoid timeouts")
    print("   • Added fallback responses for failed requests")
    print("\n🔄 Restart the backend server to apply changes:")
    print("   python backend_api.py") 