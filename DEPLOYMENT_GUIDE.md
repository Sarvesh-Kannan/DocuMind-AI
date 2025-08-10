# DocuMind AI Deployment Guide

This guide covers multiple deployment options for the DocuMind AI system.

## Architecture Overview

DocuMind AI consists of two main components:
1. **Frontend**: HTML/CSS/JS web interface 
2. **Backend**: FastAPI Python application with ML models

## Deployment Options

### Option 1: Full Local Deployment (Recommended for Development)

1. **Prerequisites**:
   ```bash
   python 3.8+
   pip install -r requirements.txt
   ollama (for local LLM)
   ```

2. **Setup**:
   ```bash
   git clone https://github.com/Sarvesh-Kannan/DocuMind-AI.git
   cd DocuMind-AI
   pip install -r requirements.txt
   ```

3. **Run**:
   ```bash
   python start_web_interface.py
   ```

### Option 2: Frontend on Vercel + Backend on Heroku/Railway

#### Deploy Frontend to Vercel:

1. **Connect GitHub to Vercel**:
   - Go to [vercel.com](https://vercel.com)
   - Connect your GitHub account
   - Import the DocuMind-AI repository

2. **Configure Build Settings**:
   - Build Command: `echo "Static site"`
   - Output Directory: `web_interface`
   - Install Command: `echo "No install needed"`

3. **Environment Variables**:
   - Set `BACKEND_URL` to your backend deployment URL

#### Deploy Backend to Heroku:

1. **Create Heroku App**:
   ```bash
   heroku create documind-ai-backend
   ```

2. **Set Environment Variables**:
   ```bash
   heroku config:set SARVAM_API_KEY=your_api_key
   heroku config:set OLLAMA_BASE_URL=your_ollama_url
   ```

3. **Create Procfile**:
   ```
   web: uvicorn backend_api:app --host=0.0.0.0 --port=$PORT
   ```

4. **Deploy**:
   ```bash
   git push heroku main
   ```

### Option 3: Railway Deployment (Backend)

1. **Connect to Railway**:
   - Go to [railway.app](https://railway.app)
   - Connect GitHub repository
   - Select the DocuMind-AI repo

2. **Configure Environment**:
   - Add environment variables for API keys
   - Set Python runtime

3. **Deploy**:
   - Railway will auto-deploy on push

### Option 4: Docker Deployment

1. **Create Dockerfile**:
   ```dockerfile
   FROM python:3.9-slim
   
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   
   COPY . .
   
   EXPOSE 8080
   CMD ["uvicorn", "backend_api:app", "--host", "0.0.0.0", "--port", "8080"]
   ```

2. **Build and Run**:
   ```bash
   docker build -t documind-ai .
   docker run -p 8080:8080 documind-ai
   ```

## Environment Variables

### Required:
- `SARVAM_API_KEY`: Your Sarvam AI translation API key
- `OLLAMA_BASE_URL`: URL to your Ollama instance

### Optional:
- `LOG_LEVEL`: Logging level (default: INFO)
- `OLLAMA_MODEL`: LLM model name (default: deepseek-r1:8b)
- `OLLAMA_TIMEOUT`: Request timeout in seconds (default: 120)

## Production Considerations

### Performance:
- Use a proper ASGI server (uvicorn/gunicorn)
- Enable caching for embeddings
- Use a CDN for static files

### Security:
- Set proper CORS origins
- Use HTTPS for all endpoints
- Validate file uploads
- Rate limiting

### Monitoring:
- Set up logging and monitoring
- Health check endpoints
- Error tracking

## Troubleshooting

### Common Issues:

1. **Torch Installation Failed**:
   - Use CPU-only version: `torch==2.2.0+cpu`
   - Or use conda: `conda install pytorch`

2. **FAISS Installation Issues**:
   - Use: `pip install faiss-cpu`
   - For GPU: `pip install faiss-gpu`

3. **Memory Issues**:
   - Reduce batch size in config
   - Use smaller embedding models
   - Implement caching

4. **Translation API Errors**:
   - Check API key validity
   - Verify network connectivity
   - Check rate limits

## Support

For issues and questions:
- Check the [GitHub Issues](https://github.com/Sarvesh-Kannan/DocuMind-AI/issues)
- Review the logs for error details
- Verify all dependencies are installed correctly

## Next Steps

1. Set up monitoring and logging
2. Implement user authentication
3. Add document management features
4. Scale for multiple users 