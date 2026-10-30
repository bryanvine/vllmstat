import pytest


@pytest.fixture(autouse=True)
def _no_host_process_discovery(monkeypatch):
    """Keep CLI tests hermetic: never pick up real `vllm serve` processes on the test host."""
    from vllmstat import cli

    monkeypatch.setattr(cli, "discover_procs", lambda *a, **k: [])
