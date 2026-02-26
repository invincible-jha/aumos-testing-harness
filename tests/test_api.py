"""API endpoint tests for aumos-testing-harness.

Tests all REST endpoints using FastAPI's TestClient with mocked services.
Auth dependencies are overridden using aumos_common.testing helpers.

Coverage target: 80% for api/ routes.
"""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from aumos_testing_harness.core.models import RunStatus, SuiteType
from aumos_testing_harness.main import app


@pytest.fixture()
def mock_tenant_context() -> MagicMock:
    """Provide a mock authenticated tenant context for request injection.

    Returns:
        Mock TenantContext with a fixed tenant_id.
    """
    tenant = MagicMock()
    tenant.tenant_id = uuid.UUID("00000000-0000-0000-0000-000000000001")
    return tenant


@pytest.fixture()
def client(mock_tenant_context: MagicMock) -> TestClient:
    """Provide a FastAPI TestClient with auth overridden.

    Args:
        mock_tenant_context: Mock tenant injected for all requests.

    Returns:
        FastAPI TestClient with dependency overrides applied.
    """
    from aumos_common.auth import get_current_user  # noqa: PLC0415

    app.dependency_overrides[get_current_user] = lambda: mock_tenant_context
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture()
def sample_suite_response() -> dict:
    """Provide a sample TestSuite API response dict.

    Returns:
        Dict matching TestSuiteResponse schema.
    """
    suite_id = str(uuid.UUID("00000000-0000-0000-0000-000000000002"))
    tenant_id = str(uuid.UUID("00000000-0000-0000-0000-000000000001"))
    return {
        "id": suite_id,
        "tenant_id": tenant_id,
        "name": "Test Suite",
        "description": None,
        "suite_type": "llm",
        "config": {"metrics": ["accuracy"], "threshold": 0.7, "test_cases": []},
        "is_active": True,
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
    }


@pytest.fixture()
def sample_run_response() -> dict:
    """Provide a sample TestRun API response dict.

    Returns:
        Dict matching TestRunResponse schema.
    """
    run_id = str(uuid.UUID("00000000-0000-0000-0000-000000000003"))
    suite_id = str(uuid.UUID("00000000-0000-0000-0000-000000000002"))
    tenant_id = str(uuid.UUID("00000000-0000-0000-0000-000000000001"))
    return {
        "id": run_id,
        "tenant_id": tenant_id,
        "suite_id": suite_id,
        "status": "completed",
        "started_at": None,
        "completed_at": None,
        "summary": {"total_tests": 1, "passed": 1, "failed": 0},
        "ci_build_id": None,
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
    }


class TestMetricsEndpoint:
    """Tests for GET /api/v1/metrics."""

    def test_list_metrics_returns_catalog(self, client: TestClient) -> None:
        """GET /metrics returns the full 23-metric catalog.

        Args:
            client: FastAPI test client.
        """
        response = client.get("/api/v1/metrics")

        assert response.status_code == 200
        data = response.json()
        assert "metrics" in data
        assert "total" in data
        assert data["total"] == 23
        assert len(data["metrics"]) == 23

    def test_metrics_catalog_has_required_fields(self, client: TestClient) -> None:
        """Each metric in the catalog has the required fields.

        Args:
            client: FastAPI test client.
        """
        response = client.get("/api/v1/metrics")
        assert response.status_code == 200

        metrics = response.json()["metrics"]
        for metric in metrics:
            assert "name" in metric
            assert "display_name" in metric
            assert "category" in metric
            assert "requires_ground_truth" in metric
            assert "description" in metric
            assert "framework" in metric

    def test_metrics_catalog_covers_llm_rag_agent_categories(self, client: TestClient) -> None:
        """The catalog covers all three evaluation categories.

        Args:
            client: FastAPI test client.
        """
        response = client.get("/api/v1/metrics")
        metrics = response.json()["metrics"]
        categories = {m["category"] for m in metrics}

        assert "llm" in categories
        assert "rag" in categories
        assert "agent" in categories


class TestTestSuitesEndpoint:
    """Tests for /api/v1/suites endpoints."""

    def test_create_suite_returns_201(
        self, client: TestClient, sample_suite_response: dict
    ) -> None:
        """POST /suites with valid data returns 201 Created.

        Args:
            client: FastAPI test client.
            sample_suite_response: Mock suite response.
        """
        mock_suite = MagicMock()
        for key, value in sample_suite_response.items():
            setattr(mock_suite, key, value)

        with patch(
            "aumos_testing_harness.api.router.LLMEvalService.create_suite",
            new_callable=AsyncMock,
            return_value=mock_suite,
        ):
            response = client.post(
                "/api/v1/suites",
                json={
                    "name": "Test Suite",
                    "suite_type": "llm",
                    "config": {
                        "metrics": ["accuracy"],
                        "threshold": 0.7,
                        "test_cases": [],
                    },
                },
            )

        assert response.status_code == 201

    def test_create_suite_validates_required_fields(self, client: TestClient) -> None:
        """POST /suites without required fields returns 422.

        Args:
            client: FastAPI test client.
        """
        response = client.post(
            "/api/v1/suites",
            json={"name": "Missing suite_type and config"},
        )
        assert response.status_code == 422

    def test_list_suites_returns_200(
        self, client: TestClient, sample_suite_response: dict
    ) -> None:
        """GET /suites returns 200 with paginated results.

        Args:
            client: FastAPI test client.
            sample_suite_response: Sample suite data.
        """
        mock_suite = MagicMock()
        for key, value in sample_suite_response.items():
            setattr(mock_suite, key, value)

        mock_page_result = MagicMock()
        mock_page_result.items = [mock_suite]
        mock_page_result.total = 1
        mock_page_result.page = 1
        mock_page_result.page_size = 20

        with patch(
            "aumos_testing_harness.api.router.LLMEvalService.list_suites",
            new_callable=AsyncMock,
            return_value=mock_page_result,
        ):
            response = client.get("/api/v1/suites")

        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert "total" in data


class TestTestRunsEndpoint:
    """Tests for /api/v1/runs endpoints."""

    def test_list_runs_returns_200(
        self, client: TestClient, sample_run_response: dict
    ) -> None:
        """GET /runs returns 200 with paginated results.

        Args:
            client: FastAPI test client.
            sample_run_response: Sample run data.
        """
        mock_run = MagicMock()
        for key, value in sample_run_response.items():
            setattr(mock_run, key, value)

        mock_page_result = MagicMock()
        mock_page_result.items = [mock_run]
        mock_page_result.total = 1
        mock_page_result.page = 1
        mock_page_result.page_size = 20

        with patch(
            "aumos_testing_harness.api.router.LLMEvalService.list_runs",
            new_callable=AsyncMock,
            return_value=mock_page_result,
        ):
            response = client.get("/api/v1/runs")

        assert response.status_code == 200


class TestRedTeamEndpoint:
    """Tests for /api/v1/red-team endpoints."""

    def test_launch_red_team_returns_202(
        self, client: TestClient, sample_run_response: dict
    ) -> None:
        """POST /red-team returns 202 Accepted.

        Args:
            client: FastAPI test client.
            sample_run_response: Sample run data.
        """
        mock_run = MagicMock()
        for key, value in sample_run_response.items():
            setattr(mock_run, key, value)

        with patch(
            "aumos_testing_harness.api.router.RedTeamService.launch_assessment",
            new_callable=AsyncMock,
            return_value=mock_run,
        ):
            response = client.post(
                "/api/v1/red-team",
                json={
                    "suite_id": "00000000-0000-0000-0000-000000000002",
                    "target_endpoint": "https://api.example.com/v1/chat",
                    "owasp_categories": ["LLM01", "LLM06"],
                    "max_attempts_per_category": 10,
                },
            )

        assert response.status_code == 202

    def test_launch_red_team_validates_max_attempts(self, client: TestClient) -> None:
        """POST /red-team with max_attempts_per_category=0 returns 422.

        Args:
            client: FastAPI test client.
        """
        response = client.post(
            "/api/v1/red-team",
            json={
                "suite_id": "00000000-0000-0000-0000-000000000002",
                "target_endpoint": "https://api.example.com/v1/chat",
                "max_attempts_per_category": 0,
            },
        )
        assert response.status_code == 422
