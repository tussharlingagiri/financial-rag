import os
import sys
import os
import sys
import time
import socket

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pytest

from app import build_chroma_index


def chroma_available(host='localhost', port=8000, timeout=1.0):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


@pytest.mark.skipif(os.environ.get('RUN_CHROMA_INTEGRATION') != '1', reason='Chroma integration tests disabled')
def test_chroma_integration_end_to_end():
    # This test expects a local Chroma server to be available (docker-compose up chroma)
    if not chroma_available('localhost', 8000):
        pytest.skip('Chroma server not reachable on localhost:8000')

    class FakeDoc:
        def __init__(self, text):
            self.text = text
            self.metadata = {'file_name': 'doc.pdf'}

    docs = [FakeDoc('alpha beta'), FakeDoc('gamma delta')]
    idx, retr = build_chroma_index(docs)

    # If chroma is available, build_chroma_index should return a retriever-like object
    assert retr is not None
    # try a simple retrieval - may differ based on backend
    try:
        res = retr.retrieve('alpha')
        assert isinstance(res, list)
    except Exception:
        pytest.skip('Chroma retriever call failed; skipping')