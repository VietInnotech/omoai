import pytest
from litestar.testing import TestClient


@pytest.mark.asyncio
async def test_pipeline_text_plain_response(monkeypatch):
    from omoai.api.app import create_app
    from omoai.api.models import PipelineResponse

    async def fake_run_full_pipeline(data, params):
        return PipelineResponse(
            summary={"title": "T", "summary": "Abstract", "bullets": ["P1", "P2"]},
            segments=[],
            transcript_punct="Hello world.",
        )

    import omoai.api.main_controller as mc

    monkeypatch.setattr(mc, "run_full_pipeline", fake_run_full_pipeline)

    app = create_app()
    with TestClient(app=app) as client:
        # Provide a small file to satisfy multipart form
        resp = client.post(
            "/v1/pipeline",
            files={"audio_file": ("a.wav", b"123", "audio/wav")},
            headers={"Accept": "text/plain"},
        )
        assert resp.status_code == 200 or resp.status_code == 201
        assert resp.headers.get("content-type", "").startswith("text/plain")
        body = resp.text
        assert "Hello world." in body
        assert "# Points" in body


@pytest.mark.asyncio
async def test_pipeline_text_plain_return_raw(monkeypatch):
    from omoai.api.app import create_app
    from omoai.api.models import PipelineResponse

    async def fake_run_full_pipeline(data, params):
        # service would normally include raw only when requested; directly include here for controller test
        return PipelineResponse(
            summary={"title": "T", "summary": "A", "bullets": [], "raw": "Title: Raw\nSummary: Raw text"},
            segments=[],
            transcript_punct=None,
        )

    import omoai.api.main_controller as mc

    monkeypatch.setattr(mc, "run_full_pipeline", fake_run_full_pipeline)

    app = create_app()
    with TestClient(app=app) as client:
        resp = client.post(
            "/v1/pipeline?return_summary_raw=true",
            files={"audio_file": ("a.wav", b"123", "audio/wav")},
            headers={"Accept": "text/plain"},
        )
        assert resp.status_code == 200 or resp.status_code == 201
        assert resp.headers.get("content-type", "").startswith("text/plain")
        body = resp.text
        assert body.startswith("Title: Raw")
