from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Mapping


FEEDBACK_TABLE = "listening_feedback"


class FeedbackSubmissionError(RuntimeError):
    """Raised when a feedback response cannot be stored."""


def submit_listening_feedback(
    supabase_url: str,
    publishable_key: str,
    payload: Mapping[str, object],
    *,
    timeout_seconds: float = 8.0,
) -> None:
    """Insert one listening response through the Supabase Data API.

    The request asks PostgREST not to return the inserted row. This keeps the
    public database role insert-only; no SELECT policy is required.
    """

    base_url = str(supabase_url).strip().rstrip("/")
    api_key = str(publishable_key).strip()
    if not base_url or not api_key:
        raise FeedbackSubmissionError("Supabase feedback settings are incomplete.")

    request = urllib.request.Request(
        f"{base_url}/rest/v1/{FEEDBACK_TABLE}",
        data=json.dumps(dict(payload)).encode("utf-8"),
        method="POST",
        headers={
            "apikey": api_key,
            "Content-Type": "application/json",
            "Prefer": "return=minimal",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=float(timeout_seconds)) as response:
            if response.status not in (200, 201, 204):
                raise FeedbackSubmissionError(
                    f"Supabase returned unexpected status {response.status}."
                )
    except FeedbackSubmissionError:
        raise
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as exc:
        raise FeedbackSubmissionError("The feedback service could not be reached.") from exc

