"""
Vercel serverless function for DocuMind AI
"""
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import our application components
from backend_api import app

# This is the entry point for Vercel
def handler(request: Request):
    return app

# For local testing
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080) 