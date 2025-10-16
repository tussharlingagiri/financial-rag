import os
import pytest

from secret_manager import EnvSecretManager, AwsSecretsManager


def test_env_secret_manager_get_set(monkeypatch):
    sm = EnvSecretManager()
    sm.set_secret('TEST_SECRET', 'value123')
    assert sm.get_secret('TEST_SECRET') == 'value123'


def test_aws_has_credentials_monkeypatched(monkeypatch):
    # If boto3 is not available, has_aws_credentials should return False
    try:
        from secret_manager import AwsSecretsManager
    except Exception:
        pytest.skip('AwsSecretsManager not importable')

    # Monkeypatch the boto3.Session to simulate no credentials
    class DummySession:
        def get_credentials(self):
            return None

    monkeypatch.setattr('secret_manager.boto3.Session', lambda: DummySession())

    sm = AwsSecretsManager()
    assert sm.has_aws_credentials() is False
