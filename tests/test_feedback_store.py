from __future__ import annotations

import json
import urllib.error

import pytest

from secondment.feedback_store import FeedbackSubmissionError, submit_listening_feedback


class _Response:
    status = 201

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None


def test_submit_listening_feedback_uses_insert_only_request(monkeypatch) -> None:
    captured = {}

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return _Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    payload = {
        "session_id": "f39011dc-e443-4f38-9548-528c18efb4d3",
        "response": "clearly",
    }

    submit_listening_feedback(
        "https://example.supabase.co/",
        "sb_publishable_test",
        payload,
    )

    request = captured["request"]
    assert request.full_url == (
        "https://example.supabase.co/rest/v1/listening_feedback"
    )
    assert request.method == "POST"
    assert request.get_header("Apikey") == "sb_publishable_test"
    assert request.get_header("Prefer") == "return=minimal"
    assert request.get_header("Authorization") is None
    assert json.loads(request.data) == payload
    assert captured["timeout"] == 8.0


def test_submit_listening_feedback_hides_transport_details(monkeypatch) -> None:
    def fake_urlopen(request, timeout):
        del request, timeout
        raise urllib.error.URLError("private connection detail")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with pytest.raises(
        FeedbackSubmissionError,
        match="feedback service could not be reached",
    ):
        submit_listening_feedback(
            "https://example.supabase.co",
            "sb_publishable_test",
            {"response": "no"},
        )

