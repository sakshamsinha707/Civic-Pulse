"""
api_server.py
-------------
HTTP routing layer for the hyperlocal issue-tracking platform.

Pipeline
--------
  POST /api/issues
        │
        ▼
  Stage 1 ── HeuristicContentFilter
        │     Rejects spam, gibberish, invalid images.
        │     → 400 on failure
        ▼
  Stage 2 ── SpatialDeduplication
        │     Checks incoming report against nearby active issues.
        │     → MERGED  : upvote existing issue, return 200 with merged result
        │     → NEW     : proceed to Stage 3
        ▼
  Stage 3 ── GeminiOrchestrator
              AI analyses the new issue cluster and produces a routing plan.
              → 201 with full AnalyzedIssueReport payload

Design notes
------------
* 100 % standard-library — no third-party imports.
* Single global InMemoryIssueStore shared across requests (swap for a DB
  adapter in production without touching this file).
* MockLLMClient is the default; set GEMINI_API_KEY env var to use production.
* Requests are handled synchronously — Stage 3 LLM call blocks the thread.
  For production, wrap in a ThreadPoolExecutor or move to async.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import sys
import traceback
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

# ---------------------------------------------------------------------------
# Path setup — allow imports from the project root regardless of cwd
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# ---------------------------------------------------------------------------
# Internal module imports
# ---------------------------------------------------------------------------

from core.models import Coordinates, Issue, IssueStatus          # noqa: E402
from core.store import InMemoryIssueStore                         # noqa: E402
from modules.spatial_dedup import (                               # noqa: E402
    DeduplicationOutcome,
    deduplicate_issue,
)
from modules.gemini_orchestrator import (                         # noqa: E402
    GeminiOrchestrator,
    MockLLMClient,
    GeminiClient,
)
from content_filter import HeuristicContentFilter                 # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("api_server")

# ---------------------------------------------------------------------------
# Shared application state  (module-level singletons)
# ---------------------------------------------------------------------------

_store = InMemoryIssueStore()

_content_filter = HeuristicContentFilter()

# Use the real Gemini client only when a key is present in the environment
_gemini_api_key = os.environ.get("GEMINI_API_KEY", "").strip()
if _gemini_api_key:
    log.info("GEMINI_API_KEY detected — using GeminiClient (production mode).")
    _llm_client = GeminiClient(api_key=_gemini_api_key)
else:
    log.info("No GEMINI_API_KEY — using MockLLMClient (development mode).")
    _llm_client = MockLLMClient()

_orchestrator = GeminiOrchestrator(client=_llm_client)


# ---------------------------------------------------------------------------
# Helper: read & parse a request body as JSON
# ---------------------------------------------------------------------------

def _read_json_body(handler: BaseHTTPRequestHandler) -> dict[str, Any] | None:
    """
    Read the request body and decode it as JSON.

    Returns the parsed dict, or None if the body is missing or malformed.
    Sends an appropriate HTTP error response before returning None so the
    caller can just `return` immediately.
    """
    content_length = int(handler.headers.get("Content-Length", 0))
    if content_length == 0:
        _send_error(handler, 400, "MISSING_BODY", "Request body is empty.")
        return None

    raw = handler.rfile.read(content_length)
    try:
        return json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        _send_error(handler, 400, "INVALID_JSON", f"Could not parse JSON body: {exc}")
        return None


# ---------------------------------------------------------------------------
# Helper: send a JSON response
# ---------------------------------------------------------------------------

def _send_json(
    handler: BaseHTTPRequestHandler,
    status: int,
    payload: dict[str, Any],
) -> None:
    body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _send_error(
    handler: BaseHTTPRequestHandler,
    status: int,
    code: str,
    detail: str,
) -> None:
    _send_json(handler, status, {"error": code, "detail": detail})


# ---------------------------------------------------------------------------
# Route: POST /api/issues
# ---------------------------------------------------------------------------

def _handle_create_issue(handler: BaseHTTPRequestHandler) -> None:
    """
    Full pipeline handler for POST /api/issues.

    Expected JSON payload
    ---------------------
    {
        "text":   "<string>",          # required
        "lat":    <float>,             # required  WGS-84 latitude
        "lon":    <float>,             # required  WGS-84 longitude
        "image":  "<base64 string>"    # optional  JPEG or PNG, base64-encoded
    }

    Successful responses
    --------------------
    201  — New distinct issue created and AI-analysed.
    200  — Duplicate detected; existing issue upvoted (merged result).

    Error responses
    ---------------
    400  — Validation / content-filter failure, or malformed request.
    500  — Unexpected server-side error.
    """

    # ── Parse body ──────────────────────────────────────────────────────────
    body = _read_json_body(handler)
    if body is None:
        return  # _read_json_body already sent the error response

    # ── Extract & validate required fields ──────────────────────────────────
    text = body.get("text")
    lat  = body.get("lat")
    lon  = body.get("lon")

    if text is None or lat is None or lon is None:
        _send_error(
            handler, 400,
            "MISSING_FIELDS",
            "Required fields: 'text' (str), 'lat' (float), 'lon' (float).",
        )
        return

    if not isinstance(text, str):
        _send_error(handler, 400, "INVALID_FIELD_TYPE", "'text' must be a string.")
        return

    try:
        lat = float(lat)
        lon = float(lon)
    except (TypeError, ValueError):
        _send_error(handler, 400, "INVALID_FIELD_TYPE",
                    "'lat' and 'lon' must be numeric values.")
        return

    # ── Decode optional image ────────────────────────────────────────────────
    image_bytes: bytes | None = None
    raw_image = body.get("image")
    if raw_image is not None:
        if not isinstance(raw_image, str):
            _send_error(handler, 400, "INVALID_FIELD_TYPE",
                        "'image' must be a base64-encoded string.")
            return
        try:
            image_bytes = base64.b64decode(raw_image, validate=True)
        except Exception:
            _send_error(handler, 400, "INVALID_IMAGE_ENCODING",
                        "'image' is not valid base64.")
            return

    # ── Stage 1: HeuristicContentFilter ─────────────────────────────────────
    log.info("Stage 1 — content filter")
    filter_result = _content_filter.filter_content(text, image_bytes)

    if not filter_result.is_valid:
        log.info("Stage 1 REJECTED  reason=%s", filter_result.reason)
        _send_error(
            handler, 400,
            "CONTENT_FILTER_REJECTED",
            f"Submission rejected by content filter: {filter_result.reason}",
        )
        return

    log.info("Stage 1 PASSED")

    # ── Build an Issue object (not yet persisted) ────────────────────────────
    try:
        coords = Coordinates(latitude=lat, longitude=lon)
    except ValueError as exc:
        _send_error(handler, 400, "INVALID_COORDINATES", str(exc))
        return

    incoming_issue = Issue(
        coordinates=coords,
        text_description=text,
        timestamp=datetime.now(timezone.utc),
        status=IssueStatus.PENDING,
    )

    # ── Stage 2: Spatial Deduplication ──────────────────────────────────────
    log.info("Stage 2 — spatial dedup  issue_id=%s", incoming_issue.id)
    dedup_result = deduplicate_issue(incoming_issue, _store)

    if dedup_result.outcome == DeduplicationOutcome.MERGED:
        # ── Duplicate path: upvote the existing canonical issue ──────────────
        existing_id = dedup_result.existing_issue_id
        existing    = _store.get(existing_id)

        if existing is not None:
            # Mutate upvote count in-place (dataclass is not frozen)
            existing.upvote_count += 1
            log.info(
                "Stage 2 MERGED  existing_id=%s  upvotes=%d  score=%.3f",
                existing_id, existing.upvote_count, dedup_result.similarity_score,
            )

        incoming_issue.status         = IssueStatus.MERGED
        incoming_issue.merged_into_id = existing_id

        _send_json(handler, 200, {
            "status":          "merged",
            "message":         "Duplicate report detected. Upvoted the existing issue.",
            "merged_into_id":  existing_id,
            "similarity_score": round(dedup_result.similarity_score, 4),
            "match_reason":    dedup_result.match_reason,
            "upvote_count":    existing.upvote_count if existing else None,
        })
        return

    # ── New distinct issue ───────────────────────────────────────────────────
    log.info("Stage 2 NEW  reason=%s", dedup_result.match_reason)

    # Persist BEFORE calling the orchestrator so it exists in the store
    incoming_issue.status = IssueStatus.OPEN
    _store.add(incoming_issue)
    log.info("Issue persisted  id=%s  total_in_store=%d",
             incoming_issue.id, len(_store))

    # ── Stage 3: Gemini Orchestrator ─────────────────────────────────────────
    log.info("Stage 3 — AI orchestration  issue_id=%s", incoming_issue.id)
    try:
        ai_report = _orchestrator.analyze_cluster([incoming_issue])
    except Exception as exc:
        # AI failure is non-fatal for persistence; issue is already stored.
        log.error("Stage 3 FAILED  %s", exc)
        _send_json(handler, 201, {
            "status":    "created",
            "issue_id":  incoming_issue.id,
            "message":   "Issue created. AI analysis unavailable.",
            "ai_report": None,
            "ai_error":  str(exc),
        })
        return

    log.info(
        "Stage 3 DONE  severity=%d  dept=%s",
        ai_report.severity_level,
        ai_report.department_routed,
    )

    _send_json(handler, 201, {
        "status":   "created",
        "issue_id": incoming_issue.id,
        "message":  "New issue created and routed successfully.",
        "ai_report": {
            "severity_level":      ai_report.severity_level,
            "department_routed":   ai_report.department_routed.value if hasattr(ai_report.department_routed, "value") else str(ai_report.department_routed),
            "generated_title":     ai_report.generated_title,
            "summary_action_plan": ai_report.summary_action_plan,
        },
    })


# ---------------------------------------------------------------------------
# Helper: serve index.html (the SPA frontend)
# ---------------------------------------------------------------------------

# Resolved once at import time so the path is stable regardless of cwd.
# This matters on Cloud Run where the working directory is not guaranteed.
_HTML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")


def _serve_html(handler: BaseHTTPRequestHandler) -> None:
    """
    Read index.html from disk and stream it to the client with the correct
    Content-Type header.

    The file is read fresh on every request so edits are reflected immediately
    without restarting the server.  In production, cache the bytes in memory
    if frontend latency becomes a concern.

    Graceful degradation: returns a 500 JSON error when index.html is missing
    so the JSON API continues to work even if the frontend file is absent.
    """
    try:
        with open(_HTML_PATH, "rb") as fh:
            body = fh.read()
    except FileNotFoundError:
        log.error("index.html not found at %s", _HTML_PATH)
        _send_error(handler, 500, "FRONTEND_NOT_FOUND",
                    "index.html is missing from the deployment package.")
        return
    except OSError as exc:
        log.error("Failed to read index.html: %s", exc)
        _send_error(handler, 500, "FRONTEND_READ_ERROR", str(exc))
        return

    handler.send_response(200)
    handler.send_header("Content-Type", "text/html; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    # No aggressive caching — a redeploy should be visible immediately.
    handler.send_header("Cache-Control", "no-cache, must-revalidate")
    handler.end_headers()
    handler.wfile.write(body)
    log.info("Served index.html (%d bytes)", len(body))


# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------

class IssueTrackerHandler(BaseHTTPRequestHandler):
    """
    HTTP request handler.

    Routes
    ------
    GET  /            → serve index.html (SPA)
    GET  /health      → JSON liveness probe
    POST /api/issues  → full issue-submission pipeline
    *                 → 404
    """

    # Suppress the default per-request stdout log from BaseHTTPRequestHandler
    # and replace it with our structured logger.
    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: D102
        log.info("HTTP %s %s — %s", self.command, self.path, fmt % args)

    def do_POST(self) -> None:  # noqa: N802
        if self.path.rstrip("/") == "/api/issues":
            try:
                _handle_create_issue(self)
            except Exception:
                log.error("Unhandled exception:\n%s", traceback.format_exc())
                _send_error(
                    self, 500,
                    "INTERNAL_SERVER_ERROR",
                    "An unexpected error occurred. Please try again later.",
                )
        else:
            _send_error(self, 404, "NOT_FOUND",
                        f"No route matches POST {self.path}")

    def do_GET(self) -> None:  # noqa: N802
        """
        GET route handler.

        Routes
        ------
        /            → serve index.html (the SPA)
        /health      → JSON liveness probe (useful for Cloud Run / uptime monitors)
        *            → 404
        """
        clean_path = self.path.split("?")[0].rstrip("/")  # strip query string + trailing slash

        if clean_path in ("", "/"):
            _serve_html(self)
        elif clean_path == "/health":
            _send_json(self, 200, {
                "status":        "ok",
                "service":       "hyperlocal-issue-tracker",
                "issues_stored": len(_store),
            })
        elif clean_path == "/api/issues":
            issues = _store.all()
            _send_json(self, 200, {
                "count": len(issues),
                "issues": [
                    {
                        "id":               i.id,
                        "lat":              i.coordinates.latitude,
                        "lon":              i.coordinates.longitude,
                        "text_description": i.text_description,
                        "status":           i.status.value,
                        "category":         i.category.value if i.category else None,
                        "upvote_count":     i.upvote_count,
                        "timestamp":        i.timestamp.isoformat(),
                    }
                    for i in issues
                ],
            })
        else:
            _send_error(self, 404, "NOT_FOUND",
                        f"No route matches GET {self.path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(host: str = "0.0.0.0", port: int = 8080) -> None:
    server = HTTPServer((host, port), IssueTrackerHandler)
    log.info("Server listening on http://%s:%d", host, port)
    log.info("Endpoints:")
    log.info("  POST /api/issues  — submit a new community issue")
    log.info("  GET  /health      — liveness check")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting down.")
    finally:
        server.server_close()


if __name__ == "__main__":
    _port = int(os.environ.get("PORT", 8080))
    _host = os.environ.get("HOST", "0.0.0.0")
    run(host=_host, port=_port)
