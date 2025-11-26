# Anime Chatbot with NLP - Project Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [NLP Features](#nlp-features)
4. [Installation & Setup](#installation--setup)
5. [Running the Application](#running-the-application)
6. [Configuration](#configuration)
7. [API Endpoints](#api-endpoints)
8. [Troubleshooting](#troubleshooting)

---

## Project Overview

The Anime Chatbot is an intelligent conversational system designed to help users discover, track, and discuss anime. It integrates cutting-edge NLP techniques with the MyAnimeList (MAL) API to provide context-aware recommendations and real-time anime information.

**Key Capabilities:**
- Intent-based conversation classification
- Anime recommendation with genre/name-based search
- Conversation history management
- Multi-stage NLP pipeline for accurate understanding
- Real-time anime data retrieval from MAL API

---

## System Architecture

### Core Components

```
┌─────────────────────────────────────────────────────┐
│              Frontend (React/JSX)                    │
│         Chat Interface + User Authentication         │
└────────────────┬────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────┐
│          FastAPI Backend (Python)                    │
│  Authentication | Route Handling | Data Management   │
└────────────────┬────────────────────────────────────┘
                 │
    ┌────────────┴──────────────┬──────────────┐
    │                           │              │
┌───▼──────┐         ┌──────────▼────┐    ┌───▼───┐
│ NLP      │         │ MAL API       │    │ DB    │
│ Pipeline │         │ Integration   │    │       │
└──────────┘         └───────────────┘    └───────┘
```

### NLP Pipeline

```
User Input
    ↓
Stage 1: DistilBERT Classification (30% confidence threshold)
    ├─ High Confidence → Use Intent Directly
    └─ Low Confidence → Stage 2
         ↓
      Stage 2: Phi-3 Classification
         ↓
      Intent Verification
    ↓
Entity Extraction (Anime Names, Genres)
    ↓
Task Execution (MAL Search, Status Update)
    ↓
Response Generation (Phi-3)
    ↓
User Output
```

---

## NLP Features

### 1. Dual-Stage Intent Classification

The system uses a two-stage classification approach for robust intent detection:

**Stage 1: DistilBERT (Fast)**
- Lightweight transformer model for quick classification
- Operates at 30% confidence threshold
- Processes 10 intents: recommendations, comparisons, ratings, status updates, etc.
- Response time: <100ms

**Stage 2: Phi-3 (Accurate)**
- Activates only when DistilBERT confidence is below 30%
- Leverages larger language model for context understanding
- Provides secondary verification
- Response time: 2-3 seconds

### 2. Intent Categories

| Intent | Purpose | Example |
|--------|---------|---------|
| `get_recommendation` | Find anime recommendations | "Recommend me action anime" |
| `check_anime_rating` | Retrieve anime ratings | "What is the rating of Demon Slayer?" |
| `compare_anime` | Compare multiple anime | "Compare Naruto and Bleach" |
| `search_anime` | Search for specific anime | "Find school girls anime" |
| `set_watching` | Add to watching list | "Add Naruto to watching" |
| `set_completed` | Mark as completed | "I finished Death Note" |
| `set_dropped` | Mark as dropped | "Drop Pokemon" |
| `view_list` | View user's anime list | "Show my watch list" |

### 3. Entity Extraction

Uses Phi-3 to extract relevant entities from user messages:
- **Anime Names**: Identifies specific anime titles mentioned
- **Genres**: Detects genre keywords (action, romance, isekai, etc.)
- **Status Keywords**: Recognizes watching/completed/dropped indicators

### 4. Context Management

Maintains conversation history for:
- Previous 5 user-bot interactions in memory
- Context-aware response generation
- Better entity resolution
- Natural conversation flow

### 5. Adaptive Responses

**Short Responses** (Status Updates):
- Instant confirmation: "✓ Added to watching: Naruto"
- No unnecessary explanation
- Direct API action execution

**Detailed Responses** (Analysis/Recommendations):
- Full Phi-3 generated responses
- 2-3 sentences of context and reasoning
- Incorporates MAL data (ratings, descriptions)

### 6. MAL Integration

When recommendation intent detected with anime/genre:
1. Extract search term from user input
2. Query MAL API for matching anime
3. Return top 5 results with ratings
4. Pass to Phi-3 for natural response generation

---

## Installation & Setup

### Prerequisites
- Python 3.10+
- CUDA 11.8+ (for GPU acceleration)
- 8GB+ VRAM (for model inference)
- Node.js 16+ (for frontend)

### Backend Setup

```bash
# 1. Clone repository
cd anime-chatbot/backend

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download models (automatic on first run)
python -c "from transformers import AutoModel; AutoModel.from_pretrained('microsoft/Phi-3-mini-4k-instruct')"
```

### Frontend Setup

```bash
# 1. Navigate to frontend
cd anime-chatbot/frontend

# 2. Install dependencies
npm install

# 3. Build
npm run build
```

### Database Setup

```bash
# Run migrations
python -m alembic upgrade head

# Or initialize SQLite (if using SQLite)
sqlite3 anime_chatbot.db < schema.sql
```

---

## Running the Application

### Start Backend

```bash
cd backend

# Activate virtual environment
source venv/bin/activate  # Linux/Mac

# Run FastAPI server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Backend runs on: `http://localhost:8000`

### Start Frontend

```bash
cd frontend

# Start development server
npm run dev
```

Frontend runs on: `http://localhost:5173`

### Access Application

Open browser and navigate to:
```
http://localhost:5173
```

---

## Configuration

### Environment Variables

Create `.env` file in backend directory:

```
DATABASE_URL=sqlite:///anime_chatbot.db
MAL_CLIENT_ID=your_mal_client_id
SECRET_KEY=your_secret_key
DISTILBERT_CONFIDENCE_THRESHOLD=0.3
```

### Model Configuration

Edit `backend/chat.py` for model settings:

```python
DISTILBERT_CONFIDENCE_THRESHOLD = 0.3  # 30% threshold for DistilBERT
SHORT_RESPONSE_INTENTS = {             # Intents returning brief responses
    "set_watching", "set_completed", 
    "set_plan_to_watch", "set_dropped", 
    "set_on_hold"
}
```

### MAL API Configuration

Update `backend/mal_service.py`:

```python
MAL_API_BASE = "https://api.myanimelist.net/v2"
SEARCH_LIMIT = 5  # Max results per search
```

---

## API Endpoints

### Chat Endpoint

**POST** `/chat`

Request:
```json
{
  "message": "recommend me action anime",
  "conversation_id": 1
}
```

Response:
```json
{
  "response": "Great action anime include Attack on Titan, Demon Slayer...",
  "action_type": "get_recommendation",
  "action_data": [...],
  "recommendations": [...]
}
```

### Conversation Management

**POST** `/conversations` - Create new conversation
**GET** `/conversations` - List all conversations
**GET** `/conversations/{id}/messages` - Get conversation history
**DELETE** `/conversations/{id}` - Delete conversation

---

## Troubleshooting

### Models Not Loading
```
Error: "Model not found in cache"
Solution: Models download automatically. Ensure internet connection 
and sufficient disk space (~5GB)
```

### Low Intent Confidence
```
Problem: Many intents fall back to Phi-3 (slow)
Solution: 
1. Check DISTILBERT_CONFIDENCE_THRESHOLD (default 0.3 is lenient)
2. Add more training examples to improve DistilBERT
3. Retrain intent classifier: python train_intent_classifier.py
```

### CUDA Out of Memory
```
Error: "CUDA out of memory"
Solution:
1. Reduce batch size in chat.py
2. Reduce max token length (512 → 256)
3. Use CPU fallback: Set DEVICE = "cpu"
```

### MAL API Errors
```
Error: "401 Unauthorized"
Solution: Regenerate MAL OAuth token, update in .env
```

---

## Fine-Tuning (Optional)

To fine-tune Phi-3 on your anime conversation data:

```bash
# Run fine-tuning script
python finetune_phi3_fast.py

# This trains for 3 epochs using LoRA (only 1-2% parameters)
# Saves to: ~/phi3_finetuned_lora/lora_adapter/

# Takes 30-60 minutes on GPU
```

---

## Performance Metrics

| Metric | Value |
|--------|-------|
| DistilBERT Classification Time | <100ms |
| Phi-3 Response Generation | 2-3s |
| Status Update Response | <100ms |
| Avg Conversation Latency | 3-5s |
| Intent Accuracy (Test Set) | 88-92% |
| Model Size (Quantized) | 2.7GB |

---

## Project Structure

```
anime-chatbot/
├── backend/
│   ├── routers/
│   │   └── chat.py           # NLP chat endpoint
│   ├── mal_service.py        # MAL API integration
│   ├── models.py             # Database models
│   ├── auth.py               # Authentication
│   ├── main.py               # FastAPI app
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/       # React components
│   │   ├── pages/            # Pages
│   │   └── App.jsx
│   └── package.json
├── anime_chatbot_5000.jsonl  # Training data (5000 examples)
└── finetune_phi3_fast.py     # Fine-tuning script
```

---

## Technologies Used

**NLP & ML:**
- DistilBERT (Intent Classification)
- Phi-3-mini (Response Generation & Secondary Classification)
- Transformers (Model Loading/Inference)
- Peft (LoRA Fine-tuning)

**Backend:**
- FastAPI (REST API)
- SQLAlchemy (ORM)
- PyJWT (Authentication)

**Frontend:**
- React 18
- Vite (Build Tool)
- Axios (HTTP Client)

**External APIs:**
- MyAnimeList API v2 (Anime Data)

---

## License & Credits

Created for anime enthusiasts who want intelligent recommendations and conversation.

**Key Contributors:**
- NLP Pipeline Implementation
- MAL API Integration
- Frontend Development

---

**Last Updated:** November 2025
