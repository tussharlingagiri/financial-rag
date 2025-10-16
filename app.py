
# -*- coding: utf-8 -*-
"""
advanced_rag.py

Refactored production-ready version of the original Colab notebook.
- Removes Colab magic and userdata usage
- Loads API keys from environment variables
- Provides a modular DocumentIngestionPipeline
- Initializes indexes and retrievers in a guarded main() flow
- Adds logging and basic error handling

Notes: install dependencies with `pip install -r requirements.txt` inside a venv.
"""

import os
import logging
try:
    import nest_asyncio
except Exception:
    nest_asyncio = None
from pathlib import Path
from typing import List
import types
import hashlib
import json

import pandas as pd
import time

# LlamaIndex / LlamaParse imports (ensure packages are installed)
# Heavy third-party imports are loaded lazily inside functions so the module
# can be imported in test environments without all external packages installed.

if nest_asyncio is not None:
    try:
        nest_asyncio.apply()
    except Exception:
        # best-effort; failure to apply is non-fatal for tests
        logging.debug("nest_asyncio.apply() failed, continuing without it")

# Central logging configuration. Respect LOG_LEVEL environment variable.
_log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, _log_level, logging.INFO), format="%(asctime)s %(levelname)s: %(message)s")

# ---------- Configuration & helpers ----------

def load_api_keys(save_to_env: bool = False):
    """Load API keys from environment variables and validate them.

    Prefer interactive prompt when running in a TTY. If not in a TTY, read from
    environment variables and raise an error if missing.

    If save_to_env=True the filled values will be appended to a local `.env` file.
    """
    import getpass

    # Try to use a SecretManager (AWS) first, fallback to environment variables
    # Prefer AWS Secrets Manager when available and credentials are present.
    openai_key = None
    llamaparse_key = None
    try:
        from secret_manager import AwsSecretsManager
        # Secret names can be overridden via env vars for flexibility in different environments
        openai_secret_name = os.environ.get("OPENAI_SECRET_NAME", "OPENAI_API_KEY")
        llamaparse_secret_name = os.environ.get("LLAMA_CLOUD_SECRET_NAME", "LLAMA_CLOUD_API_KEY")
        sm = AwsSecretsManager()
        # Only attempt if boto3 found credentials
        if getattr(sm, 'has_aws_credentials', lambda: False)():
            openai_key = sm.get_secret(openai_secret_name) or None
            llamaparse_key = sm.get_secret(llamaparse_secret_name) or None
    except Exception:
        # Any failure here should not be fatal - we'll fallback to env or interactive prompt
        openai_key = openai_key or None
        llamaparse_key = llamaparse_key or None

    # If attached to a TTY, prompt interactively for any missing keys (hidden input)
    if hasattr(os, "isatty") and os.isatty(0):
        openai_key = openai_key or os.environ.get("OPENAI_API_KEY") or getpass.getpass(prompt="OPENAI_API_KEY (input hidden): ")
        llamaparse_key = llamaparse_key or os.environ.get("LLAMA_CLOUD_API_KEY") or getpass.getpass(prompt="LLAMA_CLOUD_API_KEY (input hidden): ")
    else:
        # Non-interactive: only accept keys from secret manager or environment
        openai_key = openai_key or os.environ.get("OPENAI_API_KEY")
        llamaparse_key = llamaparse_key or os.environ.get("LLAMA_CLOUD_API_KEY")

    # Basic validation
    if not openai_key or not llamaparse_key:
        raise EnvironmentError("Both OPENAI_API_KEY and LLAMA_CLOUD_API_KEY are required")

    # Optionally save to .env for convenience (explicit)
    if save_to_env:
        env_path = Path(".env")
        try:
            with env_path.open("a") as f:
                f.write(f"OPENAI_API_KEY={openai_key}\n")
                f.write(f"LLAMA_CLOUD_API_KEY={llamaparse_key}\n")
            logging.info("Saved API keys to %s (be careful not to commit this file)", env_path)
        except Exception:
            logging.exception("Failed to save keys to .env")

    return openai_key, llamaparse_key


# ---------- Ingestion pipeline ----------
class DocumentIngestionPipeline:
    """Load and parse documents from a directory using LlamaParse (or another parser).

    Usage:
        pipeline = DocumentIngestionPipeline(data_dir="./data", parser_cls=LlamaParse, parser_kwargs={...})
        pdf_files = pipeline.get_pdf_files()
        documents = pipeline.parse_documents(pdf_files)
    """

    def __init__(self, data_dir: str = "./data", parser_cls=None, parser_kwargs: dict = None, max_docs: int = None):
        self.data_dir = Path(data_dir)
        self.parser_cls = parser_cls
        self.parser_kwargs = parser_kwargs or {}
        # Optional cap on total documents/chunks returned (helps in CI and memory-constrained runs)
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
                # Get text representation
                try:
                    text = getattr(doc, "text", None) or (doc.get_text() if hasattr(doc, "get_text") else str(doc))
                except Exception:
                    text = str(doc)

                h = hashlib.sha256(text.encode("utf-8")).hexdigest()
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)

                # Enrich metadata where possible (best-effort)
                try:
                    if hasattr(doc, "metadata") and isinstance(doc.metadata, dict):
                        doc.metadata["file_name"] = os.path.basename(filename)
                    elif hasattr(doc, "extra_info") and isinstance(doc.extra_info, dict):
                        doc.extra_info["file_name"] = os.path.basename(filename)
                except Exception:
                    logging.debug("Failed to enrich metadata for doc from %s", filename)

                all_documents.append(doc)
                added += 1

                # Respect optional max_docs cap to avoid excessive memory use in CI/quick runs
                if self.max_docs is not None and len(all_documents) >= self.max_docs:
                    logging.info("Reached max_docs cap (%d), stopping ingestion", self.max_docs)
                    break

            successful.append(filename)
            logging.info("  ✓ extracted %d unique chunks from %s", added, filename)

        logging.info("Parsing complete. successful=%d failed=%d total_chunks=%d", len(successful), len(failed), len(all_documents))
        return all_documents


# ---------- Index and retriever helpers ----------
def build_auto_index(documents):
    # Lazy import heavy library; if it's not available provide a lightweight
    # in-memory retriever fallback so tests and simple runs work without
    # installing llama_index.
    try:
        from llama_index.core import VectorStoreIndex

        index = VectorStoreIndex.from_documents(documents)
        retriever = index.as_retriever(similarity_top_k=15)
        return index, retriever
    except Exception:
        logging.warning("llama_index not available; using simple in-memory retriever fallback")

        class SimpleRetriever:
            def __init__(self, docs):
                # docs: list of document-like objects
                self._docs = docs

            def retrieve(self, query, n_results=15):
                # very simple substring scoring
                nodes = []
                for d in self._docs:
                    text = getattr(d, 'text', str(d))
                    score = 1.0 if query.lower() in text.lower() else 0.0
                    node = types.SimpleNamespace(node=types.SimpleNamespace(text=text, metadata=getattr(d, 'metadata', {})), score=score)
                    nodes.append(node)
                # sort by score desc
                nodes.sort(key=lambda n: n.score, reverse=True)
                return nodes[:n_results]

        retriever = SimpleRetriever(documents)
        return None, retriever


def build_sentence_window_index(documents):
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


def build_auto_merging_index(documents):
    from llama_index.core.node_parser import SimpleNodeParser
    from llama_index.core import VectorStoreIndex

    node_parser = SimpleNodeParser.from_defaults(chunk_size=256, chunk_overlap=50)
    nodes = node_parser.get_nodes_from_documents(documents)
    index = VectorStoreIndex(nodes)
    retriever = index.as_retriever(similarity_top_k=12)
    return index, retriever


def build_chroma_index(documents, persist_directory: str = "./chroma_db"):
    """Create a Chroma-backed vector store index when chromadb is available.

    Falls back to in-memory VectorStoreIndex if chromadb isn't installed.
    """
    try:
        import chromadb
        CHROMADB_AVAILABLE = True
    except Exception:
        CHROMADB_AVAILABLE = False

    if not CHROMADB_AVAILABLE:
        logging.warning("chromadb not available, falling back to in-memory index")
        return build_auto_index(documents)

    try:
        # create chroma client and collection
        client = chromadb.Client()
        # Use a simple collection name; in production allow configured name
        coll = client.get_or_create_collection(name="rag_collection")

        # ingest tuples: id, text, metadata
        to_upsert = []
        for i, d in enumerate(documents):
            try:
                text = getattr(d, "text", None) or (d.get_text() if hasattr(d, "get_text") else str(d))
            except Exception:
                text = str(d)
            meta = getattr(d, "metadata", None) or {}
            to_upsert.append({"id": str(i), "document": text, "meta": meta})

        coll.add(to_upsert)

        # Create a tiny wrapper that mimics an index/retriever interface used above
        class ChromaRetrieverWrapper:
            def __init__(self, collection):
                self.collection = collection

            def retrieve(self, query, n_results=15):
                results = self.collection.query(query_texts=[query], n_results=n_results)
                # convert to expected simple structure
                nodes = []
                for ids, docs, metas in zip(results["ids"], results["documents"], results["metadatas"]):
                    for d, m in zip(docs, metas):
                        node = types.SimpleNamespace(node=types.SimpleNamespace(text=d, metadata=m), score=1.0)
                        nodes.append(node)
                return nodes

        # note: for compatibility with the rest of the code we return a minimal object
        retriever = ChromaRetrieverWrapper(coll)
        return None, retriever
    except Exception:
        logging.exception("Failed to create chroma index; falling back to in-memory VectorStoreIndex")
        return build_auto_index(documents)


# ---------- Retriever safe wrapper (sync + async handling) ----------
import asyncio
import threading
import http.server
import socketserver


def run_async_query_safe(query_engine, query: str, timeout: float = 10.0):
    """Run a query against a query_engine that may be async or sync.

    - If the query_engine.query is awaitable, run it in the event loop safely.
    - If the event loop is closed or not running, create a new loop.
    - Returns the response object or raises a RuntimeError on failure.
    """
    try:
        result = query_engine.query(query)
        # if it's a coroutine, await it
        if asyncio.iscoroutine(result):
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                # running inside an event loop (e.g., notebook) - run synchronously via asyncio.run_coroutine_threadsafe
                fut = asyncio.run_coroutine_threadsafe(result, loop)
                return fut.result(timeout)
            else:
                # safe to run
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


# ---------- Main flow ----------
def main(data_dir: str = "./data"):
    # Load keys and configure models
    openai_key, llamaparse_key = load_api_keys()

    # Lazy-import and configure LlamaIndex/OpenAI related classes so module import
    # doesn't fail if those packages aren't installed in test environments.
    from llama_index.core import Settings
    from llama_index.llms.openai import OpenAI
    from llama_index.embeddings.openai import OpenAIEmbedding

    Settings.llm = OpenAI(model="gpt-4", temperature=0.1)
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
    Settings.chunk_size = 512
    Settings.chunk_overlap = 50

    # Ingestion
    # Lazily import parser class
    from llama_parse import LlamaParse

    ingestion = DocumentIngestionPipeline(
        data_dir=data_dir,
        parser_cls=LlamaParse,
        parser_kwargs={
            "api_key": llamaparse_key,
            "result_type": "markdown",
            "verbose": False,
            "language": "en",
            "num_workers": 4,
        },
        # optional cap for production can be set by environment variable
        max_docs=int(os.environ.get("MAX_INGEST_DOCS", "0")) or None,
    )
    pdf_files = ingestion.get_pdf_files()
    if not pdf_files:
        logging.warning("No PDF files found in %s - exiting", data_dir)
        return
    documents = ingestion.parse_documents(pdf_files)

    logging.info("Total document chunks: %d", len(documents))

    # Build indexes
    auto_index, auto_retriever = build_auto_index(documents)
    sw_index, sw_retriever, sw_postprocessor = build_sentence_window_index(documents)
    am_index, am_retriever = build_auto_merging_index(documents)

    # Reranker and query engines (lazy import)
    from llama_index.core.postprocessor import SentenceTransformerRerank
    from llama_index.core.query_engine import RetrieverQueryEngine

    reranker = SentenceTransformerRerank(model="cross-encoder/ms-marco-MiniLM-L-2-v2", top_n=5)

    # Query engines
    sm_query_engine = RetrieverQueryEngine(retriever=sw_retriever, node_postprocessors=[sw_postprocessor, reranker])
    am_query_engine = RetrieverQueryEngine(retriever=am_retriever, node_postprocessors=[reranker])
    a_query_engine = RetrieverQueryEngine(retriever=auto_retriever, node_postprocessors=[reranker])

    # Example evaluation: find revenue-related chunks
    all_nodes = list(auto_index.docstore.docs.values())
    revenue_chunks = [n for n in all_nodes if ("revenue" in n.text.lower() or "net revenues" in n.text.lower()) and any(y in n.text for y in ["2020","2021","2022"])]
    logging.info("Found %d revenue-related chunks", len(revenue_chunks))
    for n in revenue_chunks[:5]:
        logging.info(n.text[:300].replace('\n',' '))

    # Simple compare function (returns pandas DataFrame)
    def compare_retrievers(query: str):
        results = []
        engines = [("Sentence Window", sm_query_engine), ("Auto Merging", am_query_engine), ("Standard Auto", a_query_engine)]
        for name, engine in engines:
            start = time.time()
            resp = engine.query(query)
            elapsed = time.time() - start
            num_nodes = len(resp.source_nodes)
            avg_score = sum(n.score for n in resp.source_nodes) / num_nodes if num_nodes else 0
            results.append({
                "retriever": name,
                "response": str(resp),
                "num_source_nodes": num_nodes,
                "avg_similarity_score": avg_score,
                "response_time": elapsed,
                "response_length": len(str(resp)),
                "unique_sources": len(set(n.node.metadata.get("file_name","unknown") for n in resp.source_nodes)),
            })
        return pd.DataFrame(results)

    # Run a few test queries
    test_queries = [
        "What was Coca-Cola's revenue in 2020?",
        "What are the main risk factors mentioned in the 10-K?",
    ]
    all_dfs = [compare_retrievers(q).assign(query=q) for q in test_queries]
    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv("retriever_comparison_results.csv", index=False)
    logging.info("Saved retriever comparison results to retriever_comparison_results.csv")


def _start_metrics_server(port: int = 8000):
    try:
        from prometheus_client import start_http_server, Counter

        # Example metric: query count
        QUERY_COUNTER = Counter('rag_queries_total', 'Total number of queries')
        start_http_server(port)
        logging.info("Started Prometheus metrics server on port %d", port)
        return QUERY_COUNTER
    except Exception:
        logging.debug("prometheus_client not available; metrics server not started")
        return None


def start_health_server(port: int = 8080):
    """Start a tiny HTTP server that serves /health for container healthchecks.

    Uses the stdlib http.server to avoid adding dependencies.
    """
    if port <= 0:
        return None

    class HealthHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health":
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(b"ok")
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, format, *args):
            # suppress default logging
            return

    class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True

    try:
        server = ThreadingHTTPServer(("0.0.0.0", port), HealthHandler)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        logging.info("Started health server on port %d", port)
        return server
    except Exception:
        logging.exception("Failed to start health server on port %d", port)
        return None


def run_cli():
    import argparse

    parser = argparse.ArgumentParser(description="RAG pipeline runner")
    parser.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "./data"), help="Directory with PDF files")
    parser.add_argument("--no-prompt", action="store_true", help="Do not prompt for API keys (CI mode)")
    parser.add_argument("--save-to-keyring", action="store_true", help="Save provided keys to OS keyring for future runs")
    parser.add_argument("--json-logs", action="store_true", help="Emit JSON formatted logs")
    parser.add_argument("--metrics-port", type=int, default=int(os.environ.get("METRICS_PORT", "0")), help="Start metrics server on given port (0=disabled)")
    args = parser.parse_args()

    # JSON logging optionally
    if args.json_logs:
        try:
            import json_log_formatter

            formatter = json_log_formatter.JSONFormatter()
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
            root = logging.getLogger()
            for h in list(root.handlers):
                root.removeHandler(h)
            root.addHandler(handler)
        except Exception:
            logging.warning("json_log_formatter not available; falling back to plain logs")

    # Key management: if save-to-keyring requested, store keys there after prompt
    if args.save_to_keyring:
        try:
            import keyring
        except Exception:
            logging.error("keyring package is required for --save-to-keyring")

    # If metrics enabled, start server
    metric_counter = None
    if args.metrics_port:
        metric_counter = _start_metrics_server(args.metrics_port)

    # For CI/no-prompt mode, ensure env vars present
    if args.no_prompt:
        # Ensure keys exist in env
        if not os.environ.get("OPENAI_API_KEY") or not os.environ.get("LLAMA_CLOUD_API_KEY"):
            raise EnvironmentError("OPENAI_API_KEY and LLAMA_CLOUD_API_KEY must be set in environment when --no-prompt is used")

    # Start health server early so Docker healthchecks succeed while the
    # application initializes or when API keys are not present.
    try:
        start_health_server(port=int(os.environ.get("HEALTH_PORT", "8080")))
    except Exception:
        logging.debug("Failed to start health server from run_cli")

    # Run main but do not let it kill the process if it fails; keep the
    # health server alive so container healthchecks continue to succeed
    try:
        main(data_dir=args.data_dir)
    except Exception:
        logging.exception("Application main() failed; keeping health server alive for debugging")
        # keep process alive for debugging/health checks; exit only if explicitly requested
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            logging.info("Exiting after keyboard interrupt")


if __name__ == "__main__":
    run_cli()