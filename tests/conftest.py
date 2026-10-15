import json
from pathlib import Path

import pytest


@pytest.fixture
def example_data():
    path = Path(__file__).resolve().parents[1] / "datasets/annotations/example.json"
    return json.loads(path.read_text())
