"""
model.py - AI Medical Chatbot: NLP + Embeddings + FAISS RAG System
====================================================================
Architecture:
  1. Load dataset (CSV)
  2. Preprocess text using NLP (NLTK)
  3. Generate embeddings using SentenceTransformer
  4. Store embeddings in FAISS index
  5. Similarity search for disease prediction
  6. Agentic follow-up question logic
"""

import os
import re
import json
import pickle
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
import faiss

# ─────────────────────────────────────────────
# 1. NLTK SETUP
# ─────────────────────────────────────────────
def download_nltk_resources():
    """Download required NLTK data if not already present."""
    resources = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 'punkt_tab']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass

download_nltk_resources()

# ─────────────────────────────────────────────
# 2. NLP PREPROCESSOR
# ─────────────────────────────────────────────
class NLPPreprocessor:
    """
    Text preprocessing pipeline:
      - Lowercase
      - Remove punctuation & numbers
      - Tokenize
      - Remove stopwords
      - Lemmatize
    """
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        # Medical stopwords to KEEP (important for symptom matching)
        self.medical_keep = {
            'pain', 'ache', 'fever', 'high', 'low', 'severe', 'mild', 'chronic',
            'acute', 'no', 'not', 'without', 'loss', 'difficulty', 'inability'
        }
        self.stop_words -= self.medical_keep
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text: str) -> str:
        """Remove special characters, normalize whitespace."""
        text = text.lower().strip()
        text = re.sub(r'[^a-z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text

    def preprocess(self, text: str) -> str:
        """Full preprocessing pipeline — returns cleaned string."""
        text = self.clean_text(text)
        tokens = word_tokenize(text)
        tokens = [
            self.lemmatizer.lemmatize(t)
            for t in tokens
            if t not in self.stop_words and len(t) > 2
        ]
        return ' '.join(tokens)

    def extract_keywords(self, text: str) -> list:
        """Return list of meaningful symptom keywords."""
        processed = self.preprocess(text)
        return processed.split()


# ─────────────────────────────────────────────
# 3. MEDICAL RAG SYSTEM (Core AI Engine)
# ─────────────────────────────────────────────
class MedicalRAGSystem:
    """
    RAG-based Medical AI:
      - SentenceTransformer for dense embeddings
      - FAISS for vector similarity search
      - Agentic follow-up logic
    """

    MODEL_NAME = 'all-MiniLM-L6-v2'   # Fast, accurate, 384-dim
    INDEX_PATH = 'faiss_index.bin'
    META_PATH  = 'metadata.pkl'

    def __init__(self, data_path: str = 'data.csv'):
        self.data_path   = data_path
        self.preprocessor = NLPPreprocessor()
        self.encoder     = None   # SentenceTransformer
        self.index       = None   # FAISS index
        self.metadata    = []     # List of disease dicts
        self.is_ready    = False

    # ── 3a. Load & parse dataset ──────────────
    def load_dataset(self) -> pd.DataFrame:
        df = pd.read_csv(self.data_path)
        df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
        df = df.fillna('')
        return df

    # ── 3b. Build / load index ────────────────
    def build_index(self, force_rebuild: bool = False):
        """
        Build FAISS index from dataset embeddings.
        Caches to disk; reloads on subsequent runs.
        """
        if not force_rebuild and os.path.exists(self.INDEX_PATH) and os.path.exists(self.META_PATH):
            print("[INFO] Loading cached FAISS index…")
            self._load_index()
            return

        print("[INFO] Building FAISS index from dataset…")
        df = self.load_dataset()

        # Load SentenceTransformer
        self.encoder = SentenceTransformer(self.MODEL_NAME)

        # Combine symptom text for each disease into one embedding string
        symptom_texts = []
        for _, row in df.iterrows():
            # Preprocess symptoms
            raw_symptoms = str(row.get('symptoms', ''))
            processed    = self.preprocessor.preprocess(raw_symptoms)
            symptom_texts.append(processed)

            # Store full metadata for retrieval
            precautions = [p.strip() for p in str(row.get('precautions', '')).split('|') if p.strip()]
            medications = [m.strip() for m in str(row.get('medications', '')).split('|') if m.strip()]
            followups   = [f.strip() for f in str(row.get('followup_questions', '')).split('|') if f.strip()]

            self.metadata.append({
                'disease'     : row.get('disease', 'Unknown'),
                'symptoms_raw': raw_symptoms,
                'description' : row.get('description', ''),
                'precautions' : precautions,
                'medications' : medications,
                'followup_q'  : followups,
            })

        # Generate embeddings  [N, 384]
        print("[INFO] Generating embeddings…")
        embeddings = self.encoder.encode(
            symptom_texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True   # cosine similarity via inner product
        ).astype(np.float32)

        # Build FAISS index (Inner Product ≡ cosine when vectors are normalized)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        # Save to disk
        faiss.write_index(self.index, self.INDEX_PATH)
        with open(self.META_PATH, 'wb') as f:
            pickle.dump(self.metadata, f)

        self.is_ready = True
        print(f"[INFO] Index built: {self.index.ntotal} diseases indexed.")

    def _load_index(self):
        """Load pre-built FAISS index and metadata from disk."""
        self.index    = faiss.read_index(self.INDEX_PATH)
        with open(self.META_PATH, 'rb') as f:
            self.metadata = pickle.load(f)
        self.encoder  = SentenceTransformer(self.MODEL_NAME)
        self.is_ready = True
        print(f"[INFO] Loaded index with {self.index.ntotal} diseases.")

    # ── 3c. Symptom → Disease Retrieval ───────
    def predict(self, user_input: str, top_k: int = 3) -> list:
        """
        Given free-text symptoms, return top-k most similar diseases.

        Returns:
            list of dicts: [{disease, description, precautions,
                             medications, followup_q, score}, ...]
        """
        if not self.is_ready:
            raise RuntimeError("Index not built. Call build_index() first.")

        # Preprocess input
        processed = self.preprocessor.preprocess(user_input)
        if not processed:
            processed = user_input.lower()

        # Embed query
        query_vec = self.encoder.encode(
            [processed],
            normalize_embeddings=True
        ).astype(np.float32)

        # FAISS search
        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            entry = dict(self.metadata[idx])
            entry['score']      = float(score)
            entry['confidence'] = self._score_to_confidence(float(score))
            results.append(entry)

        return results

    @staticmethod
    def _score_to_confidence(score: float) -> str:
        """Convert cosine similarity score to human-readable confidence."""
        if score >= 0.75:
            return 'High'
        elif score >= 0.50:
            return 'Moderate'
        elif score >= 0.30:
            return 'Low'
        else:
            return 'Very Low'


# ─────────────────────────────────────────────
# 4. AGENTIC FOLLOW-UP SYSTEM
# ─────────────────────────────────────────────
class AgentFollowUp:
    """
    Manages multi-turn conversation state.
    Tracks follow-up Q&A and refines predictions.
    """

    # Universal follow-up questions asked for any prediction
    UNIVERSAL_FOLLOWUPS = [
        "How many days have you been experiencing these symptoms?",
        "Do you have any known allergies to medications?",
        "Have you taken any medications for this already?",
        "Are you experiencing any pain? If yes, on a scale of 1–10, how severe?",
        "Do you have any pre-existing medical conditions?",
    ]

    def __init__(self):
        self.session: dict = {}

    def start_session(self, session_id: str, prediction: dict):
        """Initialize a follow-up session for a prediction result."""
        disease_followups = prediction.get('followup_q', [])
        all_questions = disease_followups + [
            q for q in self.UNIVERSAL_FOLLOWUPS
            if q not in disease_followups
        ]
        self.session[session_id] = {
            'disease'        : prediction['disease'],
            'questions'      : all_questions[:5],  # max 5 follow-ups
            'answers'        : {},
            'current_q_idx'  : 0,
            'prediction'     : prediction,
        }

    def get_next_question(self, session_id: str) -> str | None:
        """Return the next unanswered follow-up question, or None if done."""
        s = self.session.get(session_id)
        if not s:
            return None
        idx = s['current_q_idx']
        if idx >= len(s['questions']):
            return None
        return s['questions'][idx]

    def record_answer(self, session_id: str, answer: str):
        """Record user's answer and advance to next question."""
        s = self.session.get(session_id)
        if not s:
            return
        idx = s['current_q_idx']
        if idx < len(s['questions']):
            question = s['questions'][idx]
            s['answers'][question] = answer
            s['current_q_idx'] += 1

    def get_session_summary(self, session_id: str) -> dict:
        """Return full follow-up summary for response enrichment."""
        return self.session.get(session_id, {})

    def is_complete(self, session_id: str) -> bool:
        s = self.session.get(session_id)
        if not s:
            return True
        return s['current_q_idx'] >= len(s['questions'])

    def clear_session(self, session_id: str):
        self.session.pop(session_id, None)


# ─────────────────────────────────────────────
# 5. RESPONSE FORMATTER
# ─────────────────────────────────────────────
class ResponseFormatter:
    """Formats raw prediction data into structured chatbot responses."""

    @staticmethod
    def format_prediction(result: dict, rank: int = 1) -> dict:
        """Format a single disease prediction into display-ready dict."""
        return {
            'rank'        : rank,
            'disease'     : result.get('disease', 'Unknown'),
            'confidence'  : result.get('confidence', 'Low'),
            'score'       : round(result.get('score', 0) * 100, 1),
            'description' : result.get('description', 'No description available.'),
            'precautions' : result.get('precautions', []),
            'medications' : result.get('medications', []),
            'followup_q'  : result.get('followup_q', []),
        }

    @staticmethod
    def format_full_response(predictions: list, user_input: str) -> dict:
        """Build complete response payload for the API."""
        if not predictions:
            return {
                'status' : 'no_match',
                'message': 'Could not identify a matching condition. Please consult a doctor.',
                'user_input': user_input,
            }

        primary = predictions[0]
        alternatives = predictions[1:] if len(predictions) > 1 else []

        return {
            'status'       : 'success',
            'user_input'   : user_input,
            'primary'      : ResponseFormatter.format_prediction(primary, rank=1),
            'alternatives' : [
                ResponseFormatter.format_prediction(p, rank=i+2)
                for i, p in enumerate(alternatives)
            ],
            'disclaimer'   : (
                "⚠️ This is an AI-generated assessment for informational purposes only. "
                "It is NOT a medical diagnosis. Always consult a qualified healthcare professional."
            )
        }


# ─────────────────────────────────────────────
# 6. SINGLETON INSTANCES (imported by app.py)
# ─────────────────────────────────────────────
rag_system    = MedicalRAGSystem(data_path='data.csv')
agent         = AgentFollowUp()
formatter     = ResponseFormatter()
