# Toweel - Emotion analysis tool - backend 🎭

## Project Overview

Toweel is an intelligent emotion analysis search engine demo project developed for the **AI in Action** Hackathon. This project adopts a frontend-backend separation architecture, and this repository contains the backend API service.

By combining Google Cloud Vertex AI's text embedding models with MongoDB vector database, Toweel can understand users' emotional expressions and return relevant emotional content, helping users better understand and express their emotional states.

## Features 🌟

- **Intelligent Emotion Analysis**: Based on Google Cloud Vertex AI's advanced text embedding models
- **Vector Semantic Search**: Precise semantic matching using MongoDB vector search
- **Conversational Guidance**: Progressive collection of user input for more accurate search results
- **RAG Enhanced Analysis**: Combined with Retrieval-Augmented Generation technology for in-depth emotion analysis reports
- **Session Management**: Multi-turn conversation support with context continuity
- **Health Monitoring**: Complete service health status monitoring

## Tech Stack 🛠️

### Core Framework
- **FastAPI** - High-performance async web framework
- **Uvicorn** - ASGI server
- **Pydantic** - Data validation and settings management

### AI & ML
- **Google Cloud Vertex AI** - Text embedding model service
    - **Text-Embedding-005** - 256-dimensional text vectorization model
    - **Gemini-2.0-flash** - Intelligent analysis, provide guidance, and RAG

### Database
- **MongoDB** - Document database and vector search
    - **Motor** - Async MongoDB driver
    - **PyMongo** - Sync MongoDB driver

### Data Processing
- **NumPy** - Numerical computation and vector operations
- **Pandas** - Data processing and analysis

### Deployment
- **Docker** - Containerized deployment
- **Docker Compose** - Multi-container orchestration

## Project Structure 📁

```
backend/
├── app/
│   ├── api/
│   │   └── endpoints/          # API route definitions
│   │       ├── search.py       # Search and analysis endpoints
│   │       ├── history.py      # History record endpoints
│   │       └── debug.py        # Debug endpoints
│   ├── services/               # Business logic layer
│   │   ├── search_service.py   # Search service
│   │   ├── rag_service.py      # RAG analysis service
│   │   ├── session_service.py  # Session management service
│   │   ├── conversation_guide_service.py  # Conversation guidance service
│   │   └── history_service.py  # History record service
│   ├── main.py                 # Application entry point
│   ├── database.py             # Database connection configuration
│   └── vector_index.py         # Vector index management
├── DataProcess/                # Data preprocessing (development phase only)
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker image build
└── docker-compose.yml          # Container orchestration configuration
```

> **Note**: The `DataProcess/` folder contains code and data for dataset cleaning and vectorization performed before the project started. The software does not use any code from this folder during actual runtime.

## API Endpoints 🔌

### Search Related
- `POST /search/` - Process user text input, perform emotion analysis and conversational guidance
- `POST /search/execute` - Execute search and return RAG analysis results
- `GET /search/session` - Create new search session
- `GET /search/session-status` - Get current session status

### System Endpoints
- `GET /health` - Service health check
- `GET /debug/*` - Debug and monitoring endpoints
- `GET /history/*` - History record management endpoints

## Requirements 📋

- Python 3.12+
- MongoDB database and connection string
- Google Cloud Platform account and service keys
- Docker (optional, for containerized deployment)

## Quick Start 🚀

If you want to try the backend of your own, please replace the environment variables to the variables of your own project.

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/Towe-el/backend.git
cd backend

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Variables

Create a `.env` file or set the following environment variables:

```bash
# Google Cloud configuration
GOOGLE_CLOUD_PROJECT=project_id
GOOGLE_APPLICATION_CREDENTIALS=path_to_`toweel-cred.json`

# MongoDB configuration
MONGODB_URI=MongoDB_connection_string
MONGODB_DATABASE=GoEmotion
MONGODB_COLLECTION=vectorizedText

# Service configuration
PORT=8080
```

### 3. Start Service

#### Development Mode
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

#### Docker Deployment
```bash
# Build and start
docker-compose up --build

# Run in background
docker-compose up -d
```

### 4. Verify Deployment

Visit `http://localhost:8080/health` to check service status:

```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T00:00:00.000Z",
  "services": {
    "database": "connected",
    "vertex_ai": "connected"
  }
}
```

## Configuration ⚙️

### Google Cloud Configuration
1. Create project in Google Cloud Console
2. Enable Vertex AI API
3. Create service account and download key file
4. Place key file as `toweel-cred.json`

### MongoDB Configuration
- Database: `GoEmotion`
- Collection: `vectorizedText`
- Requires vector search capability

### CORS Configuration
Default allowed domains:
- `https://toweel-frontend.web.app` (production environment)
- `https://toweel-frontend.firebaseapp.com` (Firebase hosting)

## Development Guide 👩‍💻

### Code Structure Principles
- **Layered Architecture**: API -> Services -> Database
- **Async First**: Use async/await for improved performance
- **Error Handling**: Comprehensive exception handling and retry mechanisms
- **Type Annotations**: Use Pydantic for data validation

### Testing
```bash
# Run tests
pytest app/tests/

# Code coverage
pytest --cov=app app/tests/
```

### Debugging
- Use `/debug` endpoints for system debugging
- Check log output for service status
- Use `/search/session-status` to monitor session state

## Deployment Guide 🚀

### Production Environment Recommendations
1. Use environment variables to manage sensitive information
2. Configure MongoDB replica sets for high availability
3. Set appropriate resource limits and health checks
4. Enable HTTPS and security headers
5. Configure log collection and monitoring

### Monitoring and Logging
- Health check endpoint: `/health`
- Service status monitoring integration
- Structured log output
- Error tracking and reporting

## License 📄

This project is licensed under an open source license. See [LICENSE](LICENSE) file for details.

---

**Note**: This is a Hackathon demo project, intended only for demonstrating technical concepts and prototype validation.