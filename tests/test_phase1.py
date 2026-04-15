import pytest
import os
import sys
from unittest.mock import MagicMock
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.document_loader import load_document
from src.utils.chat_history import init_chat_history, add_chat_turn, get_chat_history, clear_chat_history

# --- Mocking Streamlit Session State ---
@pytest.fixture
def mock_session_state():
    mock_state = {}
    return mock_state

# --- Test Case 1: DOCX Processing ---
def test_docx_processing():
    sample_path = r"e:\VSCode\PhatTrienPhanMemMaNguonMo\SmartDoc-AI\data\samples\sample_test.docx"
    if not os.path.exists(sample_path):
        pytest.skip("Sample DOCX file not found. Ensure it was created.")
    
    docs = load_document(sample_path)
    
    assert len(docs) > 0
    assert any("SmartDoc AI" in doc.page_content for doc in docs)
    # Check metadata
    assert docs[0].metadata["file_name"] == "sample_test.docx"
    assert "section" in docs[0].metadata

# --- Test Case 2: Chat History Lifecycle ---
def test_chat_history_lifecycle(mock_session_state):
    # Initialize
    init_chat_history(session_state=mock_session_state)
    assert "messages" in mock_session_state
    assert len(mock_session_state["messages"]) == 0
    
    # Add turn
    add_chat_turn(
        question="What is SmartDoc AI?",
        answer="SmartDoc AI is an assistant.",
        sources=[{"snippet": "info about AI", "file_name": "test.pdf"}],
        session_state=mock_session_state
    )
    
    history = get_chat_history(session_state=mock_session_state)
    assert len(history) == 1
    assert history[0]["question"] == "What is SmartDoc AI?"
    assert history[0]["answer"] == "SmartDoc AI is an assistant."
    assert len(history[0]["sources"]) == 1
    
    # Clear
    clear_chat_history(session_state=mock_session_state)
    assert len(get_chat_history(session_state=mock_session_state)) == 0

# --- Test Case 3: Clear Functionality (Vector Store & Files) ---
# Note: Full vector store clearing involves OS level changes, 
# here we test the session state part.
def test_clear_functionality(mock_session_state):
    init_chat_history(session_state=mock_session_state)
    add_chat_turn("q", "a", session_state=mock_session_state)
    
    # Simulate Clear Chat
    clear_chat_history(session_state=mock_session_state)
    assert len(mock_session_state["messages"]) == 0
    
    # Simulate Vector Store Clear impact on history
    add_chat_turn("q2", "a2", session_state=mock_session_state)
    clear_chat_history(session_state=mock_session_state, clear_documents=True)
    assert len(mock_session_state["messages"]) == 0
    assert mock_session_state["registered_documents"] == {}
