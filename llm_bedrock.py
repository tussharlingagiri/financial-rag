import os
import json
import logging
import typing
import io

# boto3 is optional for tests that mock it. Provide a lightweight stub when
# the real package isn't available so tests can monkeypatch `llm_bedrock.boto3.client`.
try:
    import boto3
except Exception:
    import types

    boto3 = types.SimpleNamespace()

    def _boto3_client_missing(*args, **kwargs):
        raise ImportError("boto3 is not installed. Install boto3 or monkeypatch llm_bedrock.boto3.client in tests.")

    boto3.client = _boto3_client_missing


BEDROCK_RUNTIME_NAME = os.getenv("BEDROCK_RUNTIME_NAME", "bedrock-runtime")
REGION = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")


def _get_client():
    kwargs = {}
    if REGION:
        kwargs["region_name"] = REGION
    return boto3.client(BEDROCK_RUNTIME_NAME, **kwargs)


def invoke_bedrock_model(model_id: str, prompt: str, content_type: str = "application/json") -> str:
    """Invoke Bedrock model and return text output (best-effort parsing).

    This is a minimal wrapper that handles the common JSON body response used by many
    Bedrock-compatible models. For models with custom payloads you may need to adapt
    the request/response parsing.
    """
    client = _get_client()
    payload = {"input": prompt}
    try:
        response = client.invoke_model(
            modelId=model_id,
            contentType=content_type,
            accept="application/json",
            body=json.dumps(payload).encode("utf-8"),
        )
    except Exception:
        logging.exception("Bedrock invoke_model call failed")
        raise

    # response may contain a streaming body or a direct bytes body under different keys
    body = response.get("body") or response.get("Body") or response.get("responseBody")
    if hasattr(body, "read"):
        raw = body.read()
    else:
        raw = body

    if raw is None:
        return ""

    if isinstance(raw, bytes):
        raw_bytes = raw
    elif isinstance(raw, str):
        raw_bytes = raw.encode("utf-8")
    else:
        # attempt to coerce
        raw_bytes = str(raw).encode("utf-8")

    try:
        data = json.loads(raw_bytes.decode("utf-8"))
        # common shapes: {"output": "..."} or {"results": [{"output": "..."}]}
        if isinstance(data, dict):
            if "output" in data:
                return data["output"]
            if "results" in data and isinstance(data["results"], list) and data["results"]:
                first = data["results"][0]
                if isinstance(first, dict):
                    return first.get("output") or first.get("generated_text") or json.dumps(first)
            # fallback to a stringified representation
            return json.dumps(data)
        return str(data)
    except Exception:
        try:
            return raw_bytes.decode("utf-8", errors="replace")
        except Exception:
            return str(raw)


class BedrockLLMAdapter:
    """A minimal adapter exposing a synchronous and asynchronous generate API.

    Methods:
      - generate(prompt: str) -> str
      - __call__(prompt: str) -> str  (alias to generate)
      - agenerate(prompt: str) -> str (async wrapper)
    """

    def __init__(self, model_id: str, temperature: float = 0.0):
        if not model_id:
            raise ValueError("model_id is required for BedrockLLMAdapter")
        self.model_id = model_id
        self.temperature = float(temperature)

    def generate(self, prompt: str) -> str:
        return invoke_bedrock_model(self.model_id, prompt)

    def __call__(self, prompt: str) -> str:
        return self.generate(prompt)

    async def agenerate(self, prompt: str) -> str:
        # run blocking call in a thread
        import asyncio

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.generate, prompt)
