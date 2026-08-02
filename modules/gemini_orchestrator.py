"""
modules/gemini_orchestrator.py
------------------------------
AI Orchestrator — Stage 3
==========================

Responsibility
--------------
Accept a deduplicated cluster of community Issue reports, construct a
rich context block from their metadata, call the Gemini API (or a
configurable mock), parse the JSON response, and return a validated
AnalyzedIssueReport.

Architecture
------------

  IssueCluster (list[Issue])
        │
        ▼
  ┌─────────────────────────────────────┐
  │  1. Context Assembler               │
  │     Formats timestamps, descriptions│
  │     upvote counts, coordinates into │
  │     a structured prose block.       │
  └──────────────┬──────────────────────┘
                 │  context_block: str
                 ▼
  ┌─────────────────────────────────────┐
  │  2. Prompt Builder                  │
  │     Injects context_block into the  │
  │     prompt template alongside the   │
  │     strict JSON schema definition.  │
  └──────────────┬──────────────────────┘
                 │  (system_prompt, user_prompt)
                 ▼
  ┌─────────────────────────────────────┐
  │  3. GeminiClient (or MockClient)    │
  │     Sends prompts to the model.     │
  │     Returns raw text.               │
  └──────────────┬──────────────────────┘
                 │  raw_json: str
                 ▼
  ┌─────────────────────────────────────┐
  │  4. Response Parser & Validator     │
  │     Strips markdown fences if any.  │
  │     json.loads() → dict.            │
  │     AnalyzedIssueReport.model_validate()
  └──────────────┬──────────────────────┘
                 │
                 ▼
         AnalyzedIssueReport  ✓
"""

from __future__ import annotations

import json
import re
import textwrap
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import timezone
from typing import Optional

from core.models import Issue, IssueCategory
from core.schemas import AnalyzedIssueReport, Department


# ---------------------------------------------------------------------------
# System prompt  (static — sent as the "system" role in every request)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT: str = textwrap.dedent("""
    You are CivicMind, an expert AI municipal dispatcher embedded in a
    community issue-reporting platform.

    Your job
    --------
    You will receive a CLUSTER of citizen reports that have already been
    spatially and semantically deduplicated — they all describe the same
    real-world infrastructure problem.

    You must analyse the cluster holistically and produce a single,
    authoritative assessment that helps the city operations team act fast.

    Output rules  (NON-NEGOTIABLE)
    --------------------------------
    1. Respond with ONLY a single JSON object.  No prose before or after it.
    2. Do not wrap the JSON in markdown fences (``` or ```json).
    3. Every key listed in the schema below MUST be present.
    4. Do not add any keys not listed in the schema.
    5. All string values must be in English.
    6. `severity_level` must be an INTEGER between 1 and 5 inclusive:
         1 = Cosmetic / negligible impact on daily life
         2 = Minor inconvenience, non-urgent
         3 = Moderate disruption, schedule within 72 hours
         4 = Significant hazard, respond within 24 hours
         5 = Emergency — immediate risk to life or critical infrastructure
    7. `department_routed` must be EXACTLY one of these strings:
         "Water & Sanitation"
         "Electrical"
         "Public Works"
         "Waste Management"
         "Parks & Recreation"
         "Traffic Engineering"
         "Building Inspection"
         "Emergency Services"
    8. `generated_title` must be a concise title (6–12 words) that a field
       technician can understand at a glance.
    9. `summary_action_plan` must be plain prose, ≤150 words, written as an
       ordered set of steps for the assigned department.

    JSON Schema
    -----------
    {
      "severity_level":      <integer 1–5>,
      "department_routed":   <string — one of the 8 values above>,
      "generated_title":     <string>,
      "summary_action_plan": <string>
    }
""").strip()


# ---------------------------------------------------------------------------
# Prompt template  (dynamic — assembled per cluster)
# ---------------------------------------------------------------------------

USER_PROMPT_TEMPLATE: str = textwrap.dedent("""
    CLUSTER SUMMARY
    ===============
    Total reports in cluster : {total_reports}
    Total citizen upvotes    : {total_upvotes}
    Bounding area            : {bounding_area}
    First reported (UTC)     : {first_reported}
    Most recent report (UTC) : {last_reported}
    Inferred category        : {inferred_category}

    INDIVIDUAL REPORTS
    ==================
    {individual_reports}

    ---
    Analyse the cluster above.  Return ONLY the JSON object.
""").strip()


# ---------------------------------------------------------------------------
# 1. Context Assembler
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ClusterContext:
    """Intermediate representation built from a list of Issues."""
    total_reports:      int
    total_upvotes:      int
    bounding_area:      str   # Human-readable lat/lon bounding box
    first_reported:     str   # ISO-8601 UTC
    last_reported:      str   # ISO-8601 UTC
    inferred_category:  str
    individual_reports: str   # Formatted block of per-report entries


def _format_timestamp(issue: Issue) -> str:
    """Return a clean UTC timestamp string regardless of tz-awareness."""
    ts = issue.timestamp
    if ts.tzinfo is None:
        return ts.strftime("%Y-%m-%d %H:%M UTC")
    return ts.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _infer_category(issues: list[Issue]) -> str:
    """
    Return the most common non-null category label in the cluster, or
    'Unknown' if no category has been assigned yet by the ML filter.
    """
    counts: dict[str, int] = {}
    for issue in issues:
        if issue.category is not None:
            label = issue.category.value
            counts[label] = counts.get(label, 0) + 1
    if not counts:
        return "Unknown"
    return max(counts, key=lambda k: counts[k])


def _bounding_area(issues: list[Issue]) -> str:
    """Compact human-readable lat/lon bounding box for the cluster."""
    lats = [i.coordinates.latitude  for i in issues]
    lons = [i.coordinates.longitude for i in issues]
    return (
        f"Lat [{min(lats):.5f} → {max(lats):.5f}], "
        f"Lon [{min(lons):.5f} → {max(lons):.5f}]"
    )


def assemble_cluster_context(issues: list[Issue]) -> ClusterContext:
    """
    Transform a list of Issue objects into a ClusterContext.

    The ordering of individual reports is chronological (oldest first) so
    the model sees the progression of the problem over time.

    Parameters
    ----------
    issues : At least one Issue.  Caller (orchestrator) validates this.

    Returns
    -------
    ClusterContext ready to be injected into the prompt template.
    """
    if not issues:
        raise ValueError("Cannot assemble context from an empty cluster.")

    sorted_issues = sorted(issues, key=lambda i: i.timestamp)

    # Build the per-report block
    report_lines: list[str] = []
    for idx, issue in enumerate(sorted_issues, start=1):
        upvote_note = (
            f" (+{issue.upvote_count} upvotes)" if issue.upvote_count > 0 else ""
        )
        coords = (
            f"{issue.coordinates.latitude:.5f}°N, "
            f"{issue.coordinates.longitude:.5f}°E"
        )
        description = issue.text_description.strip() or "(no description provided)"
        report_lines.append(
            f"[Report {idx}] {_format_timestamp(issue)}{upvote_note}\n"
            f"  Location   : {coords}\n"
            f"  Description: {description}"
        )

    return ClusterContext(
        total_reports=len(issues),
        total_upvotes=sum(i.upvote_count for i in issues),
        bounding_area=_bounding_area(issues),
        first_reported=_format_timestamp(sorted_issues[0]),
        last_reported=_format_timestamp(sorted_issues[-1]),
        inferred_category=_infer_category(issues),
        individual_reports="\n\n".join(report_lines),
    )


# ---------------------------------------------------------------------------
# 2. Prompt Builder
# ---------------------------------------------------------------------------

def build_user_prompt(context: ClusterContext) -> str:
    """
    Inject a ClusterContext into USER_PROMPT_TEMPLATE.

    Returns
    -------
    Fully-rendered user prompt string ready to send to the model.
    """
    return USER_PROMPT_TEMPLATE.format(
        total_reports=context.total_reports,
        total_upvotes=context.total_upvotes,
        bounding_area=context.bounding_area,
        first_reported=context.first_reported,
        last_reported=context.last_reported,
        inferred_category=context.inferred_category,
        individual_reports=context.individual_reports,
    )


# ---------------------------------------------------------------------------
# 3. LLM Client abstraction
# ---------------------------------------------------------------------------

class BaseLLMClient(ABC):
    """
    Protocol that both the real Gemini client and the mock implement.
    The orchestrator depends only on this interface — never on the
    concrete implementation.
    """

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """
        Send prompts to the model and return the raw text response.

        Parameters
        ----------
        system_prompt : The static system instructions.
        user_prompt   : The dynamically assembled cluster context.

        Returns
        -------
        Raw string from the model — expected to be a JSON object, but
        may contain markdown fences that the parser must strip.
        """


class GeminiClient(BaseLLMClient):
    """
    Production client wrapping google.generativeai.

    Usage
    -----
    Set the GEMINI_API_KEY environment variable before instantiating.

        import os
        client = GeminiClient(api_key=os.environ["GEMINI_API_KEY"])

    Model choice
    ------------
    gemini-2.5-flash        → balanced, has free tier quota  (default)
    gemini-2.0-flash-lite   → fastest, lowest cost
    gemini-2.5-pro          → highest reasoning quality (use for severity 4–5)
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.5-flash",
    ) -> None:
        try:
            import google.generativeai as genai  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "google-generativeai is not installed.  "
                "Run: pip install google-generativeai"
            ) from exc

        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=SYSTEM_PROMPT,
            generation_config=genai.GenerationConfig(
                # Force JSON-only output at the API level (Gemini 1.5+)
                response_mime_type="application/json",
                temperature=0.1,   # Low temperature = deterministic, structured
                max_output_tokens=512,
            ),
        )

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        # system_prompt is baked into the model at construction;
        # we pass user_prompt as the sole turn.
        response = self._model.generate_content(user_prompt)
        return response.text


class MockLLMClient(BaseLLMClient):
    """
    Deterministic stub for local development and CI.

    Behaviour
    ---------
    - Returns a valid AnalyzedIssueReport JSON by default.
    - Can be configured with `response_override` to test specific payloads,
      including invalid ones (for parser error-path testing).
    - Records every (system_prompt, user_prompt) pair it receives so tests
      can assert that prompts were assembled correctly.
    """

    def __init__(
        self,
        response_override: Optional[str] = None,
    ) -> None:
        self._override = response_override
        self.calls: list[dict[str, str]] = []   # Captured for test assertions

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append({
            "system_prompt": system_prompt,
            "user_prompt":   user_prompt,
        })

        if self._override is not None:
            return self._override

        # ── Deterministic default response ───────────────────────────────────
        # Infer a plausible department from keywords in the user prompt
        dept = Department.PUBLIC_WORKS.value   # safe fallback
        prompt_lower = user_prompt.lower()

        keyword_map = [
            (["water", "leak", "pipe", "drain", "sewage", "flood"],
             Department.WATER_AND_SANITATION.value),
            (["light", "electric", "power", "lamp", "streetlight"],
             Department.ELECTRICAL.value),
            (["pothole", "road", "pavement", "crack", "asphalt"],
             Department.PUBLIC_WORKS.value),
            (["garbage", "waste", "trash", "bin", "litter", "dump"],
             Department.WASTE_MANAGEMENT.value),
            (["park", "tree", "garden", "playground"],
             Department.PARKS_AND_RECREATION.value),
            (["traffic", "signal", "sign", "marking"],
             Department.TRAFFIC_ENGINEERING.value),
        ]
        for keywords, department in keyword_map:
            if any(kw in prompt_lower for kw in keywords):
                dept = department
                break

        # Severity heuristic: scale with upvote/report count in prompt
        severity = 2
        for marker, level in [("upvotes: 1", 1), ("upvotes: 0", 1),
                               ("upvotes: 2", 2), ("upvotes: 3", 3),
                               ("upvotes: 4", 4), ("upvotes: 5", 5)]:
            if marker in user_prompt.lower():
                severity = level

        return json.dumps({
            "severity_level": severity,
            "department_routed": dept,
            "generated_title": "Mock: Infrastructure Issue Requires Immediate Attention",
            "summary_action_plan": (
                "1. Dispatch field inspector within 24 hours to assess the site. "
                "2. Erect safety barriers if a hazard is confirmed. "
                "3. Schedule repair crew based on inspector's report. "
                "4. Notify affected residents via the platform. "
                "5. Mark issue as resolved once work is verified."
            ),
        })


# ---------------------------------------------------------------------------
# 4. Response Parser
# ---------------------------------------------------------------------------

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def parse_llm_response(raw: str) -> AnalyzedIssueReport:
    """
    Parse and validate the raw LLM text into an AnalyzedIssueReport.

    Handles
    -------
    - Clean JSON strings (ideal path)
    - JSON wrapped in ```json ... ``` markdown fences (common with Gemini)
    - Leading/trailing whitespace

    Raises
    ------
    ValueError  : If JSON cannot be decoded or schema validation fails.
    """
    text = raw.strip()

    # Strip markdown fences if present
    fence_match = _JSON_FENCE_RE.search(text)
    if fence_match:
        text = fence_match.group(1).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"LLM response is not valid JSON.\n"
            f"Raw response (first 400 chars): {raw[:400]!r}\n"
            f"Error: {exc}"
        ) from exc

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected a JSON object, got {type(data).__name__}."
        )

    try:
        return AnalyzedIssueReport.model_validate(data)
    except Exception as exc:
        raise ValueError(
            f"LLM response failed schema validation: {exc}\n"
            f"Parsed dict: {data}"
        ) from exc


# ---------------------------------------------------------------------------
# 5. Orchestrator  (primary public API)
# ---------------------------------------------------------------------------

@dataclass
class OrchestratorConfig:
    """
    Tunable settings injected into the orchestrator at construction.

    Attributes
    ----------
    min_cluster_size : Refuse to analyse clusters smaller than this.
                       A lone unconfirmed report warrants a different flow.
    max_cluster_size : Truncate very large clusters to the N most-upvoted
                       issues to stay within token limits.
    """
    min_cluster_size: int = 1
    max_cluster_size: int = 20


class GeminiOrchestrator:
    """
    Coordinates context assembly, prompt construction, LLM invocation,
    and response parsing into a single high-level call.

    Parameters
    ----------
    client : Any BaseLLMClient implementation (real or mock).
    config : OrchestratorConfig with tuneable limits.

    Example
    -------
    >>> orchestrator = GeminiOrchestrator(client=MockLLMClient())
    >>> report = orchestrator.analyze_cluster(issues)
    >>> print(report.severity_level, report.department_routed)
    """

    def __init__(
        self,
        client: BaseLLMClient,
        config: Optional[OrchestratorConfig] = None,
    ) -> None:
        self._client = client
        self._config = config or OrchestratorConfig()

    def analyze_cluster(self, issues: list[Issue]) -> AnalyzedIssueReport:
        """
        Full pipeline: cluster → AnalyzedIssueReport.

        Parameters
        ----------
        issues : The deduplicated cluster from the spatial dedup module.
                 Must contain at least config.min_cluster_size items.

        Returns
        -------
        AnalyzedIssueReport  — validated and ready to persist / route.

        Raises
        ------
        ValueError : Cluster too small, too large after truncation fails,
                     or LLM response invalid.
        """
        if len(issues) < self._config.min_cluster_size:
            raise ValueError(
                f"Cluster has {len(issues)} issue(s); "
                f"minimum required is {self._config.min_cluster_size}."
            )

        # Truncate oversized clusters: keep highest-upvoted reports first
        working_set = issues
        if len(issues) > self._config.max_cluster_size:
            working_set = sorted(
                issues, key=lambda i: i.upvote_count, reverse=True
            )[: self._config.max_cluster_size]

        # Stage 1 → 2: assemble context and build prompt
        context     = assemble_cluster_context(working_set)
        user_prompt = build_user_prompt(context)

        # Stage 3: call LLM
        raw_response = self._client.generate(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

        # Stage 4: parse and validate
        return parse_llm_response(raw_response)
