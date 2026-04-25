# 🏥 MediBot — AI Medical Chatbot

> **GenAI + NLP + RAG-based Medical Assistant**  
> A production-level AI health assistant using Sentence Transformers, FAISS vector search, and intelligent agentic follow-up questioning.

---

## 🧠 System Architecture

```
User Input (Natural Language Symptoms)
         │
         ▼
  NLP Preprocessing (NLTK)
  ┌─────────────────────────┐
  │  Lowercase → Tokenize   │
  │  Remove stopwords       │
  │  Lemmatize keywords     │
  └─────────────────────────┘
         │
         ▼
  Embedding Layer (SentenceTransformer)
  ┌─────────────────────────────────────┐
  │  all-MiniLM-L6-v2 model            │
  │  384-dimensional dense vectors      │
  └─────────────────────────────────────┘
         │
         ▼
  FAISS Vector Search
  ┌─────────────────────────────────────┐
  │  IndexFlatIP (cosine similarity)    │
  │  Top-K nearest disease match        │
  └─────────────────────────────────────┘
         │
         ▼
  RAG Response Generation
  ┌─────────────────────────────────────┐
  │  Disease Name + Confidence Score    │
  │  Description from knowledge base   │
  │  Precautions list                  │
  │  Medication guidance               │
  └─────────────────────────────────────┘
         │
         ▼
  Agentic Follow-Up System
  ┌─────────────────────────────────────┐
  │  Disease-specific follow-up Qs     │
  │  Universal clinical questions       │
  │  Session state management          │
  └─────────────────────────────────────┘
         │
         ▼
  Flask Web UI (Structured Output)
```

---

## 🚀 Quick Start

### 1. Clone / Download the project
```bash
cd medical_chatbot
```

### 2. Create virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
python app.py
```

### 5. Open browser
```
http://localhost:5000
```

> **First run:** The FAISS index builds automatically (10–20 seconds). Subsequent runs load from cache.

---

## 📁 Project Structure

```
medical_chatbot/
├── app.py              # Flask backend (routes, session management)
├── model.py            # AI core: NLP + Embeddings + FAISS + Agent
├── data.csv            # Medical knowledge base (20 diseases)
├── requirements.txt    # Python dependencies
├── README.md           # This file
├── templates/
│   └── index.html      # Chatbot UI (Jinja2)
└── static/
    └── style.css       # Dark clinical UI styling
```

---

## 🔬 Tech Stack Explained

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **NLP Preprocessing** | NLTK | Tokenization, stopword removal, lemmatization |
| **Embeddings** | `all-MiniLM-L6-v2` (Sentence Transformers) | 384-dim semantic vectors |
| **Vector DB** | FAISS (IndexFlatIP) | Cosine similarity search |
| **Backend** | Flask 3.x | REST API + Jinja2 templating |
| **Frontend** | HTML5 + CSS3 + Vanilla JS | Real-time chat interface |
| **Agentic Logic** | Custom rule-based agent | Follow-up Q&A session management |

---

## 💡 How It Works (Step by Step)

### Step 1: Dataset Loading
```python
df = pd.read_csv('data.csv')
# Columns: disease, symptoms, description, precautions, medications, followup_questions
```

### Step 2: NLP Preprocessing
```python
preprocessor = NLPPreprocessor()
clean = preprocessor.preprocess("I have high fever and body aches")
# → "high fever body ache"
```

### Step 3: Embedding Generation
```python
encoder = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = encoder.encode(symptom_texts, normalize_embeddings=True)
# Shape: [20, 384]
```

### Step 4: FAISS Indexing
```python
index = faiss.IndexFlatIP(384)   # Inner Product = cosine (normalized)
index.add(embeddings)             # Add all disease embeddings
faiss.write_index(index, 'faiss_index.bin')  # Cache to disk
```

### Step 5: Similarity Search
```python
query_vec = encoder.encode(["fever headache body pain"], normalize_embeddings=True)
scores, indices = index.search(query_vec, k=3)
# Returns top-3 closest diseases with similarity scores
```

### Step 6: Agentic Follow-Up
```python
agent.start_session(session_id, top_prediction)
question = agent.get_next_question(session_id)
# → "How many days have you had these symptoms?"
agent.record_answer(session_id, "3 days")
# → continues to next question
```

---

## 📊 Dataset Format (data.csv)

```
disease, symptoms, description, precautions, medications, followup_questions

"Influenza (Flu)",
"high fever|chills|body aches|cough",
"Influenza is a contagious respiratory illness...",
"Rest|Stay hydrated|Avoid contact with sick",
"Paracetamol|Decongestants",
"When did symptoms start?|Do you have severe body pain?"
```

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Chatbot UI |
| `POST` | `/predict` | Symptom → Disease prediction |
| `POST` | `/followup` | Submit follow-up answer |
| `GET` | `/reset` | Reset conversation |
| `GET` | `/health` | System status |

### Example: /predict
```json
POST /predict
{
  "symptoms": "I have fever, severe headache, and body aches for 3 days"
}

Response:
{
  "status": "success",
  "primary": {
    "disease": "Influenza (Flu)",
    "confidence": "High",
    "score": 82.3,
    "description": "...",
    "precautions": ["Rest", "Stay hydrated", ...],
    "medications": ["Paracetamol", ...]
  },
  "alternatives": [...],
  "followup": {
    "question": "When did symptoms start suddenly?",
    "has_more": true
  },
  "disclaimer": "..."
}
```

---

## ⚠️ Disclaimer

This system is for **educational and informational purposes only**. It does **NOT** provide medical diagnoses or prescriptions. Always consult a qualified healthcare professional for medical advice.

---


---

*Built with Python, Flask, SentenceTransformers, FAISS, and NLTK*
