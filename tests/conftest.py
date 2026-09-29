import os
import warnings

import pytest

warnings.filterwarnings("ignore")

CIRCUITS = os.path.join(os.path.dirname(__file__), "..", "examples", "circuits")


@pytest.fixture
def circuit():
    def _path(name):
        return os.path.join(CIRCUITS, f"{name}.qasm")
    return _path
