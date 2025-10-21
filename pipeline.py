# -*- coding: utf-8 -*-
"""
pipeline.py

Minimal, parser-agnostic RAG pipeline using OpenAI API key.

Usage:
    from pipeline import DocumentIngestionPipeline, build_auto_index
    # Supply your own parser class
    ingestion = DocumentIngestionPipeline(data_dir="./data", parser_cls=YourParser, parser_kwargs={})
    pdf_files = ingestion.get_pdf_files()
    documents = ingestion.parse_documents(pdf_files)
    index, retriever = build_auto_index(documents)
    response = retriever.retrieve("Your query here")
"""
import asyncio

import os
import logging
from pathlib import Path
from typing import List
import hashlib
import types
import time

def load_api_keys(save_to_env: bool = False):
    import getpass
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        openai_key = getpass.getpass(prompt="Enter your OPENAI_API_KEY (input hidden): ")
    if not openai_key:
        raise EnvironmentError("OPENAI_API_KEY is required for this workflow.")
    if save_to_env:
        env_path = Path(".env")
        try:
            with env_path.open("a") as f:
                f.write(f"OPENAI_API_KEY={openai_key}\n")
            logging.info("Saved OPENAI_API_KEY to %s (be careful not to commit this file)", env_path)
        except Exception:
            logging.exception("Failed to save key to .env")
    return openai_key, None

class DocumentIngestionPipeline:
    """Load and parse documents from a directory using a parser.
    User must supply parser_cls and parser_kwargs.
    """
    def __init__(self, data_dir: str = "./data", parser_cls=None, parser_kwargs: dict = None, max_docs: int = None):
        self.data_dir = Path(data_dir)
        self.parser_cls = parser_cls
        self.parser_kwargs = parser_kwargs or {}
        self.max_docs = max_docs
    def get_pdf_files(self) -> List[str]:
        pdf_files = [str(p) for p in self.data_dir.glob("*.pdf")]
        logging.info("Found %d PDF files in %s", len(pdf_files), str(self.data_dir))
        for p in pdf_files:
            logging.info("  - %s", p)
        return pdf_files
    def parse_documents(self, pdf_files: List[str]):
        if not self.parser_cls:
            raise ValueError("parser_cls must be provided to parse documents")
        parser = self.parser_cls(**self.parser_kwargs)
        all_documents = []
        successful = []
        failed = []
        seen_hashes = set()
        for i, filename in enumerate(pdf_files, start=1):
            logging.info("[%d/%d] Parsing %s", i, len(pdf_files), filename)
            attempts = 0
            max_attempts = 3
            backoff = 1.0
            docs = None
            while attempts < max_attempts:
                try:
                    docs = parser.load_data(filename)
                    break
                except Exception as e:
                    attempts += 1
                    logging.warning("Attempt %d to parse %s failed: %s", attempts, filename, e)
                    time.sleep(backoff)
                    backoff *= 2
            if docs is None:
                failed.append(filename)
                logging.exception("  ❌ failed to parse %s after %d attempts", filename, max_attempts)
                continue
            added = 0
            for doc in docs:
                try:
                    text = getattr(doc, "text", None) or (doc.get_text() if hasattr(doc, "get_text") else str(doc))
                except Exception:
                    text = str(doc)
                h = hashlib.sha256(text.encode("utf-8")).hexdigest()
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)
                try:
                    if hasattr(doc, "metadata") and isinstance(doc.metadata, dict):
                        doc.metadata["file_name"] = os.path.basename(filename)
                    elif hasattr(doc, "extra_info") and isinstance(doc.extra_info, dict):
                        doc.extra_info["file_name"] = os.path.basename(filename)
                except Exception:
                    logging.debug("Failed to enrich metadata for doc from %s", filename)
                all_documents.append(doc)
                added += 1
                if self.max_docs is not None and len(all_documents) >= self.max_docs:
                    logging.info("Reached max_docs cap (%d), stopping ingestion", self.max_docs)
                    break
            successful.append(filename)
            logging.info("  ✓ extracted %d unique chunks from %s", added, filename)
        logging.info("Parsing complete. successful=%d failed=%d total_chunks=%d", len(successful), len(failed), len(all_documents))
        return all_documents

def build_auto_index(documents):
    try:
        from llama_index.core import VectorStoreIndex
        index = VectorStoreIndex.from_documents(documents)
        retriever = index.as_retriever(similarity_top_k=15)
        return index, retriever
    except Exception:
        logging.warning("llama_index not available; using simple in-memory retriever fallback")
        class SimpleRetriever:
            def __init__(self, docs):
                self._docs = docs
            def retrieve(self, query, n_results=15):
                nodes = []
                for d in self._docs:
                    text = getattr(d, 'text', str(d))
                    score = 1.0 if query.lower() in text.lower() else 0.0
                    node = types.SimpleNamespace(node=types.SimpleNamespace(text=text, metadata=getattr(d, 'metadata', {})), score=score)
                    nodes.append(node)
                nodes.sort(key=lambda n: n.score, reverse=True)
                return nodes[:n_results]
        retriever = SimpleRetriever(documents)
        return None, retriever


# ---------- Additional Index and Retriever Helpers ----------
def build_sentence_window_index(documents):
    try:
        from llama_index.core.node_parser import SentenceWindowNodeParser
        from llama_index.core import VectorStoreIndex
        from llama_index.core.postprocessor import MetadataReplacementPostProcessor
        node_parser = SentenceWindowNodeParser.from_defaults(
            window_size=3,
            window_metadata_key="window",
            original_text_metadata_key="original_text",
        )
        nodes = node_parser.get_nodes_from_documents(documents)
        index = VectorStoreIndex(nodes)
        retriever = index.as_retriever(similarity_top_k=15)
        postprocessor = MetadataReplacementPostProcessor(target_metadata_key="window")
        return index, retriever, postprocessor
    except Exception:
        logging.warning("llama_index not available; sentence window index not created")
        return None, None, None

def build_auto_merging_index(documents):
    try:
        from llama_index.core.node_parser import SimpleNodeParser
        from llama_index.core import VectorStoreIndex
        node_parser = SimpleNodeParser.from_defaults(chunk_size=256, chunk_overlap=50)
        nodes = node_parser.get_nodes_from_documents(documents)
        index = VectorStoreIndex(nodes)
        retriever = index.as_retriever(similarity_top_k=12)
        return index, retriever
    except Exception:
        logging.warning("llama_index not available; auto merging index not created")
        return None, None

def build_chroma_index(documents, persist_directory: str = "./chroma_db"):
    try:
        import chromadb
        CHROMADB_AVAILABLE = True
    except Exception:
        CHROMADB_AVAILABLE = False
    if not CHROMADB_AVAILABLE:
        logging.warning("chromadb not available, falling back to in-memory index")
        return build_auto_index(documents)
    try:
        client = chromadb.Client()
        coll = client.get_or_create_collection(name="rag_collection")
        to_upsert = []
        for i, d in enumerate(documents):
            try:
                text = getattr(d, "text", None) or (d.get_text() if hasattr(d, "get_text") else str(d))
            except Exception:
                text = str(d)
            meta = getattr(d, "metadata", None) or {}
            to_upsert.append({"id": str(i), "document": text, "meta": meta})
        coll.add(to_upsert)
        class ChromaRetrieverWrapper:
            def __init__(self, collection):
                self.collection = collection
            def retrieve(self, query, n_results=15):
                results = self.collection.query(query_texts=[query], n_results=n_results)
                nodes = []
                for ids, docs, metas in zip(results["ids"], results["documents"], results["metadatas"]):
                    for d, m in zip(docs, metas):
                        node = types.SimpleNamespace(node=types.SimpleNamespace(text=d, metadata=m), score=1.0)
                        nodes.append(node)
                return nodes
        retriever = ChromaRetrieverWrapper(coll)
        return None, retriever
    except Exception:
        logging.exception("Failed to create chroma index; falling back to in-memory VectorStoreIndex")
        return build_auto_index(documents)


def run_async_query_safe(query_engine, query: str, timeout: float = 10.0):
    try:
        result = query_engine.query(query)
        if asyncio.iscoroutine(result):
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop and loop.is_running():
                fut = asyncio.run_coroutine_threadsafe(result, loop)
                return fut.result(timeout)
            else:
                return asyncio.run(result)
        else:
            return result
    except Exception as e:
        logging.exception("Retriever query failed: %s", e)
        raise RuntimeError("Retriever query failed") from e

def safe_query_with_retry(query_engine, query: str, retries: int = 1, timeout: float = 10.0):
    last_exc = None
    for attempt in range(1, retries + 2):
        try:
            return run_async_query_safe(query_engine, query, timeout=timeout)
        except Exception as e:
            logging.warning("Query attempt %d failed: %s", attempt, e)
            last_exc = e
            continue
    raise last_exc
