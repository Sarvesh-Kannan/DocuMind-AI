"""
Vercel-compatible API entry point for DocuMind AI
"""
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend_api import app

# Export the FastAPI app for Vercel
def handler(request, response):
    return app(request, response)

# For local development
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080) 