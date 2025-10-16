import os
import sys
import types
import pytest

# Ensure repo root is on sys.path for importing app.py when tests run
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# A light-weight integration-style smoke test that mocks heavy external
# dependencies (like llama_parse) so CI can run quickly.

from app import DocumentIngestionPipeline


class DummyParser:
    def __init__(self, **kwargs):
        pass

    def load_data(self, filename):
        # return a list of simple objects with text attribute
        return [types.SimpleNamespace(text=f"dummy from {filename}", metadata={})]


def test_ingestion_pipeline_tmp(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    # create a fake pdf file (content irrelevant for mocked parser)
    f = d / "doc1.pdf"
    f.write_text("dummy")

    # instantiate pipeline with dummy parser
    pipeline = DocumentIngestionPipeline(data_dir=str(d), parser_cls=DummyParser, parser_kwargs={})
    docs = pipeline.parse_documents([str(f)])
    assert len(docs) == 1
    assert hasattr(docs[0], "text")