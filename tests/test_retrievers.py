import os
import sys
import types

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app import build_auto_index


def test_build_auto_index_fallback_retriever(monkeypatch):
    # Ensure llama_index is not importable to trigger fallback
    monkeypatch.setitem(sys.modules, 'llama_index', None)

    class DummyDoc:
        def __init__(self, text):
            self.text = text
            self.metadata = {'file_name': 'f.pdf'}

    docs = [DummyDoc('hello world'), DummyDoc('foo bar')]
    idx, retr = build_auto_index(docs)
    assert retr is not None

    # The fallback retriever implements retrieve(query)
    results = retr.retrieve('hello')
    assert isinstance(results, list)
    assert any('hello' in n.node.text for n in results)


def test_simple_retriever_scoring(monkeypatch):
    monkeypatch.setitem(sys.modules, 'llama_index', None)

    class D:
        def __init__(self, text):
            self.text = text

    docs = [D('apple banana'), D('orange apple'), D('banana')]
    _, retr = build_auto_index(docs)

    res1 = retr.retrieve('orange')
    res2 = retr.retrieve('banana')
    # First result for 'orange' should contain 'orange'
    assert any('orange' in n.node.text for n in res1)
    assert any('banana' in n.node.text for n in res2)
