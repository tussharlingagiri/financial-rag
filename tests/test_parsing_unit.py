import os
import sys
import types

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app import DocumentIngestionPipeline


class FakeDoc:
    def __init__(self, text, meta=None):
        self.text = text
        self.metadata = meta or {}


class ParserThatReturnsDuplicates:
    def __init__(self, **kwargs):
        pass

    def load_data(self, filename):
        # returns duplicate documents
        return [FakeDoc('dup1'), FakeDoc('dup1'), FakeDoc('unique')]


def test_parsing_dedup_and_metadata(tmp_path):
    d = tmp_path / 'data'
    d.mkdir()
    f = d / 'a.pdf'
    f.write_text('x')

    pipeline = DocumentIngestionPipeline(data_dir=str(d), parser_cls=ParserThatReturnsDuplicates, parser_kwargs={})
    docs = pipeline.parse_documents([str(f)])
    # duplicates should be deduplicated by hash -> expect 2 unique
    assert len(docs) == 2


def test_parsing_max_docs_cap(tmp_path):
    d = tmp_path / 'data'
    d.mkdir()
    f = d / 'b.pdf'
    f.write_text('x')

    class ManyDocsParser:
        def load_data(self, filename):
            return [FakeDoc(str(i)) for i in range(100)]

    pipeline = DocumentIngestionPipeline(data_dir=str(d), parser_cls=ManyDocsParser, parser_kwargs={}, max_docs=10)
    docs = pipeline.parse_documents([str(f)])
    assert len(docs) <= 10
