import pytest


@pytest.fixture
def tushare_token():
    """Read token from .env or skip if not available."""
    import os
    from dotenv import load_dotenv
    load_dotenv()
    token = os.getenv("TUSHARE_TOKEN", "")
    if not token:
        pytest.skip("TUSHARE_TOKEN not set")
    return token
