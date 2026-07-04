# Creator: Sulabh Bansod
# Description: Pytest configuration and shared test fixtures.
# Use: Prepares test environments, mock clients, and dummy data for automated testing.

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="session")
def client() -> TestClient:
    return TestClient(app)
