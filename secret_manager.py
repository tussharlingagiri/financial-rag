"""Secret manager abstraction with fallback to environment variables.

Implements a simple interface:
- get_secret(name) -> str | None
- set_secret(name, value) -> None (optional)

Current implementations:
- AwsSecretsManager (requires boto3 and AWS creds)
- EnvSecretManager (reads from environment variables)
"""
import os
from typing import Optional

# Import boto3 at module level (if available). Tests can monkeypatch
# `secret_manager.boto3.Session` when boto3 is present; otherwise boto3
# will be None and AwsSecretsManager will gracefully disable itself.
try:
    import boto3  # type: ignore
except Exception:
    # Create a tiny dummy module-like object so tests can monkeypatch
    # `secret_manager.boto3.Session` even when boto3 isn't installed.
    try:
        from types import SimpleNamespace
        boto3 = SimpleNamespace(Session=lambda: None)  # type: ignore
    except Exception:
        boto3 = None  # fallback; has_aws_credentials will return False


class SecretManager:
    def get_secret(self, name: str) -> Optional[str]:
        raise NotImplementedError()

    def set_secret(self, name: str, value: str) -> None:
        raise NotImplementedError()


class EnvSecretManager(SecretManager):
    def get_secret(self, name: str) -> Optional[str]:
        return os.environ.get(name)

    def set_secret(self, name: str, value: str) -> None:
        # Not recommended to write to environment in production; this is a no-op
        os.environ[name] = value


class AwsSecretsManager(SecretManager):
    def __init__(self, region_name: str = None):
        # Use module-level boto3 if available
        if boto3 is None:
            self.client = None
        else:
            try:
                # boto3 will pick up credentials from environment, shared config, or IAM role
                self.client = boto3.client('secretsmanager', region_name=region_name)
            except Exception:
                self.client = None

    def get_secret(self, name: str) -> Optional[str]:
        if not self.client:
            return None
        try:
            resp = self.client.get_secret_value(SecretId=name)
            return resp.get('SecretString')
        except Exception:
            # Log exception upstream rather than raising here to allow fallbacks
            import logging
            logging.debug("AwsSecretsManager.get_secret failed for %s", name, exc_info=True)
            return None

    def set_secret(self, name: str, value: str) -> None:
        if not self.client:
            raise RuntimeError("AWS Secrets Manager client not available")
        try:
            self.client.put_secret_value(SecretId=name, SecretString=value)
        except Exception as e:
            import logging
            logging.exception("Failed to put secret %s", name)
            raise

    @staticmethod
    def has_aws_credentials() -> bool:
        """Quick check whether boto3 can find credentials in the environment or IAM role."""
        if boto3 is None:
            return False
        try:
            session = boto3.Session()
            creds = session.get_credentials()
            return creds is not None
        except Exception:
            return False
