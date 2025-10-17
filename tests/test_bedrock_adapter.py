import json
import types
import pytest


def make_fake_boto3(client_response_bytes: bytes):
    class FakeBody:
        def __init__(self, b):
            self._b = b

        def read(self):
            return self._b

    class FakeClient:
        def invoke_model(self, **kwargs):
            return {"body": FakeBody(client_response_bytes)}

    def fake_client(name, **kwargs):
        return FakeClient()

    return fake_client


def test_bedrock_adapter_parses_json(monkeypatch):
    from llm_bedrock import BedrockLLMAdapter

    payload = {"output": "Hello from Bedrock"}
    fake = make_fake_boto3(json.dumps(payload).encode("utf-8"))
    monkeypatch.setattr("llm_bedrock.boto3.client", fake)

    adapter = BedrockLLMAdapter(model_id="test-model")
    out = adapter.generate("hi")
    assert "Hello from Bedrock" in out


def test_bedrock_adapter_handles_text(monkeypatch):
    from llm_bedrock import BedrockLLMAdapter

    raw = b"plain text response"
    fake = make_fake_boto3(raw)
    monkeypatch.setattr("llm_bedrock.boto3.client", fake)

    adapter = BedrockLLMAdapter(model_id="test-model")
    out = adapter.generate("hi")
    assert "plain text response" in out
