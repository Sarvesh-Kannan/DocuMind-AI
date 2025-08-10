#!/usr/bin/env python3
"""
DocuMind AI Web Interface Startup Script
Starts the backend API server and provides instructions for accessing the web interface
"""

import subprocess
import sys
import webbrowser
import time
import requests
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'fastapi',
        'uvicorn',
        'sentence_transformers',
        'torch',
        'transformers',
        'faiss-cpu',
        'scikit-learn',
        'PyMuPDF',
        'nltk',
        'pandas',
        'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   • {package}")
        print("\n📦 Install missing packages:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    return True

def check_ollama():
    """Check if Ollama is running"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [model['name'] for model in models]
            
            if 'deepseek-r1:8b' in model_names:
                print("✅ Ollama is running with deepseek-r1:8b model")
                return True
            else:
                print("⚠️  Ollama is running but deepseek-r1:8b model not found")
                print("   Run: ollama pull deepseek-r1:8b")
                return False
        else:
            print("❌ Ollama is not responding")
            return False
    except requests.exceptions.RequestException:
        print("❌ Ollama is not running")
        print("   Start Ollama: ollama serve")
        return False

def start_backend():
    """Start the FastAPI backend server"""
    try:
        print("🚀 Starting DocuMind AI Backend Server...")
        
        # Start the backend server
        process = subprocess.Popen([
            sys.executable, 
            "backend_api.py"
        ], cwd=Path.cwd())
        
        # Wait for server to start
        print("⏳ Waiting for server to start...")
        for i in range(10):
            try:
                response = requests.get("http://localhost:8080/api/health", timeout=2)
                if response.status_code == 200:
                    print("✅ Backend server is running!")
                    break
            except requests.exceptions.RequestException:
                time.sleep(1)
                continue
        else:
            print("❌ Failed to start backend server")
            return None
        
        return process
        
    except Exception as e:
        print(f"❌ Error starting backend: {e}")
        return None

def main():
    """Main startup function"""
    print("🎯 DocuMind AI Web Interface Startup")
    print("=" * 50)
    
    # Check dependencies
    print("\n1. Checking dependencies...")
    if not check_dependencies():
        return 1
    print("✅ All dependencies are installed")
    
    # Check Ollama
    print("\n2. Checking Ollama LLM...")
    ollama_ok = check_ollama()
    if not ollama_ok:
        print("⚠️  Continue anyway? LLM features will be limited.")
        response = input("Continue? (y/N): ")
        if response.lower() != 'y':
            return 1
    
    # Start backend
    print("\n3. Starting backend server...")
    backend_process = start_backend()
    if not backend_process:
        return 1
    
    # Success message
    print("\n" + "=" * 50)
    print("🎉 DocuMind AI Web Interface is Ready!")
    print("=" * 50)
    print("\n🌐 Web Interface:")
    print("   http://localhost:8080/static/index.html")
    print("\n📖 API Documentation:")
    print("   http://localhost:8080/docs")
    print("\n🔧 System Features:")
    print("   • Upload PDF documents")
    print("   • AI-powered search & summarization")
    print("   • Real-time performance analytics")
    print("   • Local processing (100% privacy)")
    print("\n⚡ Quick Start:")
    print("   1. Open the web interface")
    print("   2. Upload your PDF documents")
    print("   3. Start asking questions!")
    print("\n💡 Pro Tips:")
    print("   • Use the suggested queries for better results")
    print("   • Try different search types (hybrid/semantic/keyword)")
    print("   • Adjust summary length for your needs")
    print("\n🛑 To stop the server:")
    print("   Press Ctrl+C in this terminal")
    
    # Open browser
    try:
        print("\n🌐 Opening web interface in browser...")
        webbrowser.open("http://localhost:8080/static/index.html")
    except Exception as e:
        print(f"⚠️  Could not open browser automatically: {e}")
        print("   Please open http://localhost:8080/static/index.html manually")
    
    # Keep the script running
    try:
        print("\n⏳ Server is running... Press Ctrl+C to stop")
        backend_process.wait()
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down DocuMind AI...")
        backend_process.terminate()
        backend_process.wait()
        print("✅ Server stopped successfully")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 