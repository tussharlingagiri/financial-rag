import os
import sys
import types

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app import _start_metrics_server


def test_metrics_server_not_installed(monkeypatch):
    # Simulate prometheus_client not available
    monkeypatch.setitem(sys.modules, 'prometheus_client', None)
    counter = _start_metrics_server(0)
    assert counter is None
