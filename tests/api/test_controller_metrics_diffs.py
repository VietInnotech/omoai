import pytest
from litestar.testing import TestClient


@pytest.mark.asyncio
async def test_controller_includes_metrics_and_diffs_when_requested(monkeypatch):
    from omoai.api.app import create_app
    from omoai.api.models import (
        HumanReadableDiff,
        PipelineResponse,
        QualityMetrics,
    )

    async def fake_run_full_pipeline(data, params):
        return PipelineResponse(
            summary={
                "title": "t",
                "summary": "a",
                "bullets": ["p1"],
                "raw": "raw summary",
            },
            segments=[],
            transcript_punct="Hello.",
            quality_metrics=QualityMetrics(wer=0.1, cer=0.05, alignment_confidence=0.9),
            diffs=HumanReadableDiff(
                original_text="hello",
                punctuated_text="Hello.",
                diff_output="- hello\n+ Hello.",
                alignment_summary="ok",
            ),
        )

    import omoai.api.main_controller as mc

    monkeypatch.setattr(mc, "run_full_pipeline", fake_run_full_pipeline)

    app = create_app()
    with TestClient(app=app) as client:
        resp = client.post(
            "/v1/pipeline?include_quality_metrics=true&include_diffs=true&return_summary_raw=true",
            files={"audio_file": ("a.wav", b"123", "audio/wav")},
        )
        assert resp.status_code in (200, 201)
        data = resp.json()
        assert data["quality_metrics"] is not None
        assert data["quality_metrics"]["wer"] == 0.1
        assert data["diffs"] is not None
        assert data["diffs"]["punctuated_text"] == "Hello."
        assert data["summary"].get("raw") == "raw summary"


@pytest.mark.asyncio
async def test_controller_excludes_metrics_and_diffs_when_not_requested(monkeypatch):
    from omoai.api.app import create_app
    from omoai.api.models import (
        HumanReadableDiff,
        PipelineResponse,
        QualityMetrics,
    )

    async def fake_run_full_pipeline(data, params):
        # service may compute metrics/diffs but API should omit them when not requested
        return PipelineResponse(
            summary={
                "title": "t",
                "summary": "a",
                "bullets": [],
                "raw": "raw summary",
            },
            segments=[],
            transcript_punct="Hello.",
            quality_metrics=QualityMetrics(wer=0.1),
            diffs=HumanReadableDiff(punctuated_text="Hello."),
        )

    import omoai.api.main_controller as mc

    monkeypatch.setattr(mc, "run_full_pipeline", fake_run_full_pipeline)

    app = create_app()
    with TestClient(app=app) as client:
        resp = client.post(
            "/v1/pipeline", files={"audio_file": ("a.wav", b"123", "audio/wav")}
        )
        assert resp.status_code in (200, 201)
        data = resp.json()
        # Not requested => omitted by default
        assert data.get("quality_metrics") is None
        assert data.get("diffs") is None
        assert data["summary"].get("raw") is None
