"""Small JSON-over-HTTP helpers shared by the chat-completions judge and the Jev client."""

from __future__ import annotations

import json
import urllib.error
import urllib.request


class JudgeUnavailable(Exception):
    """The judge can't be used for this clip (no key, key rejected, bad model...)."""


def _post_json(url: str, headers: dict, body: dict, timeout: float) -> tuple[int, dict]:
    """POST JSON; returns (status, parsed body). Raises OSError / HTTPException on network failures."""
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            status, raw = res.status, res.read()
    except urllib.error.HTTPError as exc:
        status, raw = exc.code, exc.read()
    try:
        data = json.loads(raw or b"{}")
    except ValueError:
        return (502 if status == 200 else status), {"error": {"message": raw.decode("utf-8", "replace")[:300]}}
    return status, data if isinstance(data, dict) else {"error": {"message": str(data)[:300]}}


def _error_message(data: dict) -> str:
    err = data.get("error", data) if isinstance(data, dict) else data
    if isinstance(err, list) and err:
        err = err[0]
    if isinstance(err, dict):
        return str(err.get("message") or err.get("detail") or err)[:300]
    return str(err)[:300]
