import os
import sys
import types

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app import build_chroma_index


def test_build_chroma_index_fallback(monkeypatch):
    # Simulate chromadb not installed
    monkeypatch.setitem(sys.modules, 'chromadb', None)

    class DummyDoc:
        def __init__(self, text):
            self.text = text
            self.metadata = {'file_name': 'doc.pdf'}

    docs = [DummyDoc('hello world')]
    # Should fall back to an in-memory retriever; idx may be None depending on implementation
    idx, retr = build_chroma_index(docs)
    assert retr is not None
    assert hasattr(retr, 'retrieve') or hasattr(retr, 'as_retriever')
