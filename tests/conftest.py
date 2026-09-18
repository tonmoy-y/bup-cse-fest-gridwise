import pytest


@pytest.fixture
def mp(monkeypatch):
    """Alias used by the failover tests (which also run standalone with a fake)."""
    return monkeypatch
