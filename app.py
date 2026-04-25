"""
app.py - AI Medical Chatbot: Flask Backend
==========================================
Routes:
  GET  /           → Home page (chat UI)
  POST /predict    → Symptom → Disease prediction (JSON)
  POST /followup   → Submit follow-up answer (JSON)
  GET  /reset      → Reset conversation session
  GET  /health     → System health check
"""

import os
import uuid
import threading
from flask import Flask, request, jsonify, render_template, session

from model import rag_system, agent, formatter

# ─────────────────────────────────────────────
# FLASK APP SETUP
# ─────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'med-chatbot-secret-2024-xyz')
app.config['SESSION_TYPE'] = 'filesystem'

# ─────────────────────────────────────────────
# STARTUP: Build FAISS Index in Background
# ─────────────────────────────────────────────
_index_ready = threading.Event()

def _init_index():
    print("[STARTUP] Initializing RAG system…")
    rag_system.build_index(force_rebuild=False)
    _index_ready.set()
    print("[STARTUP] RAG system ready ✓")

threading.Thread(target=_init_index, daemon=True).start()


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route('/')
def home():
    """Serve the main chatbot interface."""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return render_template('index.html')


@app.route('/health')
def health():
    """System health check — useful for deployment."""
    return jsonify({
        'status' : 'ok',
        'index_ready': _index_ready.is_set(),
        'diseases_loaded': rag_system.index.ntotal if rag_system.index else 0,
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint: Symptom → Disease Prediction

    Request body (JSON):
      { "symptoms": "I have fever, headache and body aches" }

    Response (JSON):
      {
        "status": "success",
        "primary": { disease, confidence, description, precautions, medications },
        "alternatives": [...],
        "followup": { "question": "...", "session_id": "..." },
        "disclaimer": "..."
      }
    """
    if not _index_ready.is_set():
        return jsonify({
            'status' : 'loading',
            'message': 'AI system is initializing, please wait 10–15 seconds and try again.',
        }), 503

    data = request.get_json(silent=True) or {}
    symptoms = data.get('symptoms', '').strip()

    if not symptoms or len(symptoms) < 3:
        return jsonify({
            'status' : 'error',
            'message': 'Please describe your symptoms in more detail.',
        }), 400

    # Run RAG retrieval
    try:
        predictions = rag_system.predict(symptoms, top_k=3)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

    # Format response
    response = formatter.format_full_response(predictions, symptoms)

    # Initialize follow-up agent session
    session_id = session.get('session_id', str(uuid.uuid4()))
    session['session_id'] = session_id

    if predictions:
        agent.start_session(session_id, predictions[0])
        first_q = agent.get_next_question(session_id)
        response['followup'] = {
            'session_id': session_id,
            'question'  : first_q,
            'has_more'  : not agent.is_complete(session_id),
        }

    return jsonify(response)


@app.route('/followup', methods=['POST'])
def followup():
    """
    Endpoint: Submit Follow-Up Answer

    Request body (JSON):
      { "session_id": "...", "answer": "Yes, I have had fever for 3 days" }

    Response (JSON):
      { "recorded": true, "next_question": "...", "complete": false }
    """
    data = request.get_json(silent=True) or {}
    session_id = data.get('session_id') or session.get('session_id')
    answer     = data.get('answer', '').strip()

    if not session_id or not answer:
        return jsonify({'status': 'error', 'message': 'Missing session_id or answer'}), 400

    agent.record_answer(session_id, answer)
    next_q    = agent.get_next_question(session_id)
    complete  = agent.is_complete(session_id)

    return jsonify({
        'recorded'     : True,
        'next_question': next_q,
        'complete'     : complete,
        'message'      : (
            'Follow-up complete. Based on your answers, we recommend consulting a doctor with this information.'
            if complete else ''
        ),
    })


@app.route('/reset', methods=['GET', 'POST'])
def reset():
    """Reset conversation and start fresh."""
    session_id = session.get('session_id')
    if session_id:
        agent.clear_session(session_id)
    session['session_id'] = str(uuid.uuid4())
    return jsonify({'status': 'reset', 'session_id': session['session_id']})


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
