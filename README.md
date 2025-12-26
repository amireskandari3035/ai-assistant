# Master's AI Assistant

A Persian-language AI assistant for master's students using RAG (Retrieval Augmented Generation) with Groq AI.

## Features

- 📚 Process Persian PDF documents (up to 70+ pages)
- 🤖 Powered by Groq AI (LLaMA 3.3 / Mixtral)
- 🔍 Intelligent document search with RAG
- 💬 Modern Persian RTL chat interface

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your Groq API key
```

### 3. Add Your Documents

Place your PDF documents in the `data/documents/` folder.

### 4. Run the Application

```bash
python -m uvicorn backend.main:app --reload
```

Open your browser to `http://localhost:8000`

## Project Structure

```
agent-assistant/
├── backend/           # FastAPI backend
│   ├── services/      # Core services (chat, documents, embeddings)
│   └── routers/       # API endpoints
├── frontend/          # Chat UI
├── data/documents/    # Your PDF documents
└── vectorstore/       # ChromaDB storage
```

## API Endpoints

- `POST /api/chat` - Send a message and get AI response
- `POST /api/upload` - Upload new documents
- `GET /api/status` - Check system status

## License

MIT
