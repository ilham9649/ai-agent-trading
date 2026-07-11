from pathlib import Path

import pytest

from trading_agent.config import load_settings


@pytest.fixture
def settings():
    return load_settings(Path(__file__).resolve().parents[1] / "config" / "settings.yaml")
