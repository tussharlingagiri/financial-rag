import os
import pytest


BEDROCK_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID")


@pytest.mark.integration
def test_bedrock_integration_smoke():
    """Run a minimal Bedrock smoke test. This test is skipped unless BEDROCK_MODEL_ID is provided.

    This test is intended for manual runs (the workflow `bedrock-integration.yml`).
    It will attempt to instantiate the adapter and perform a single short generation.
    """
    if not BEDROCK_MODEL_ID:
        pytest.skip("BEDROCK_MODEL_ID not set; skipping live Bedrock integration test")

    # Import lazily to avoid errors in dev where the adapter or boto3 aren't installed
    try:
        from llm_bedrock import BedrockLLMAdapter
    except Exception as e:
        pytest.fail(f"Failed to import Bedrock adapter: {e}")

    try:
        adapter = BedrockLLMAdapter(model_id=BEDROCK_MODEL_ID, temperature=0.0)
    except Exception as e:
        pytest.fail(f"Failed to initialize Bedrock adapter: {e}")

    # Run a tiny generation; this may incur costs in your AWS account.
    try:
        out = adapter.generate("Hello from integration test", max_tokens=8)
    except Exception as e:
        pytest.fail(f"Bedrock generate failed: {e}")

    assert out is not None
