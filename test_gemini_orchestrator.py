"""
tests/test_gemini_orchestrator.py
----------------------------------
Covers every stage of the orchestrator pipeline using MockLLMClient.
No API key or network access required.

Tested behaviours
-----------------
  Schema              : AnalyzedIssueReport validates correct data,
                        rejects out-of-range severity, unknown departments.
  Context Assembler   : Ordering, upvote sums, bounding box, category inference.
  Prompt Builder      : All template slots populated; schema injected.
  Parser              : Clean JSON, fenced JSON, invalid JSON, schema violations.
  Orchestrator        : Happy path, keyword→department routing, cluster
                        truncation, empty cluster rejection.
  Prompt Capture      : System prompt wording; user prompt contains key phrases.
"""

from __future__ import annotations

import json
import sys
import os
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.models import Coordinates, Issue, IssueCategory, IssueStatus
from core.schemas import AnalyzedIssueReport, Department
from modules.gemini_orchestrator import (
    SYSTEM_PROMPT,
    GeminiOrchestrator,
    MockLLMClient,
    OrchestratorConfig,
    assemble_cluster_context,
    build_user_prompt,
    parse_llm_response,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BASE_LAT, BASE_LON = 12.9716, 77.5946

def _ts(offset_hours: int = 0) -> datetime:
    return datetime(2026, 6, 22, 10, 0, 0, tzinfo=timezone.utc) + timedelta(hours=offset_hours)


def _issue(
    desc: str,
    upvotes: int = 0,
    category: IssueCategory | None = None,
    lat_offset: float = 0.0,
    ts_offset: int = 0,
) -> Issue:
    return Issue(
        coordinates=Coordinates(BASE_LAT + lat_offset, BASE_LON),
        timestamp=_ts(ts_offset),
        text_description=desc,
        upvote_count=upvotes,
        category=category,
        status=IssueStatus.OPEN,
    )


POTHOLE_CLUSTER = [
    _issue("Large pothole on main road, car damaged axle.", upvotes=4, category=IssueCategory.POTHOLE, ts_offset=0),
    _issue("Pothole near junction, very dangerous at night.", upvotes=2, category=IssueCategory.POTHOLE, ts_offset=1, lat_offset=0.001),
    _issue("Road has sunk near the drain outlet.", upvotes=1, category=IssueCategory.ROAD_DAMAGE, ts_offset=3, lat_offset=0.002),
]

WATER_CLUSTER = [
    _issue("Water pipe burst on 5th street, road flooded.", upvotes=5, category=IssueCategory.WATER_LEAKAGE, ts_offset=0),
    _issue("Water leaking from underground pipe for 2 days.", upvotes=3, category=IssueCategory.WATER_LEAKAGE, ts_offset=2),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

errors: list[str] = []
passed: int = 0

def check(name: str, cond: bool, detail: str = "") -> None:
    global passed
    tag = "PASS" if cond else "FAIL"
    suffix = f"  ({detail})" if detail and not cond else ""
    print(f"  {tag}  {name}{suffix}")
    if cond:
        passed += 1
    else:
        errors.append(name)


def _valid_report(**overrides) -> dict:
    base = {
        "severity_level": 3,
        "department_routed": Department.PUBLIC_WORKS.value,
        "generated_title": "Road Damaged Near Central Junction",
        "summary_action_plan": "1. Inspect site. 2. Patch pothole. 3. Monitor.",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# 1. Schema validation
# ---------------------------------------------------------------------------

def test_schema() -> None:
    print("\n── Schema ──")

    r = AnalyzedIssueReport.model_validate(_valid_report())
    check("valid report parses", r.severity_level == 3)
    check("department is enum", r.department_routed == Department.PUBLIC_WORKS)
    check("model_dump has 4 keys", len(r.model_dump()) == 4)
    check("model_dump_json is valid JSON", json.loads(r.model_dump_json()) is not None)

    # Out-of-range severity
    try:
        AnalyzedIssueReport.model_validate(_valid_report(severity_level=6))
        check("severity 6 rejected", False)
    except (ValueError, Exception):
        check("severity 6 rejected", True)

    try:
        AnalyzedIssueReport.model_validate(_valid_report(severity_level=0))
        check("severity 0 rejected", False)
    except (ValueError, Exception):
        check("severity 0 rejected", True)

    # Unknown department
    try:
        AnalyzedIssueReport.model_validate(_valid_report(department_routed="Underwater Basket Weaving"))
        check("bad department rejected", False)
    except (ValueError, Exception):
        check("bad department rejected", True)

    check("schema has 4 required fields",
          len(AnalyzedIssueReport.model_json_schema().get("required", [])) == 4)


# ---------------------------------------------------------------------------
# 2. Context Assembler
# ---------------------------------------------------------------------------

def test_context_assembler() -> None:
    print("\n── Context Assembler ──")

    ctx = assemble_cluster_context(POTHOLE_CLUSTER)
    check("total_reports correct", ctx.total_reports == 3)
    check("total_upvotes correct", ctx.total_upvotes == 7)
    check("first_reported earliest", "10:00" in ctx.first_reported)
    check("last_reported latest", "13:00" in ctx.last_reported)
    check("category inferred as pothole", "pothole" in ctx.inferred_category.lower())
    check("bounding area has lat range", "→" in ctx.bounding_area)
    check("individual reports has all 3", ctx.individual_reports.count("[Report") == 3)
    check("reports ordered chronologically",
          ctx.individual_reports.index("[Report 1]") < ctx.individual_reports.index("[Report 2]"))
    check("upvote count appears in report block", "+4 upvotes" in ctx.individual_reports)

    # Edge: single issue
    single_ctx = assemble_cluster_context([POTHOLE_CLUSTER[0]])
    check("single issue context ok", single_ctx.total_reports == 1)

    # Edge: no category assigned
    no_cat = [_issue("mystery problem")]
    no_cat_ctx = assemble_cluster_context(no_cat)
    check("unknown category handled", no_cat_ctx.inferred_category == "Unknown")

    # Edge: empty cluster raises
    try:
        assemble_cluster_context([])
        check("empty cluster raises", False)
    except ValueError:
        check("empty cluster raises", True)


# ---------------------------------------------------------------------------
# 3. Prompt Builder
# ---------------------------------------------------------------------------

def test_prompt_builder() -> None:
    print("\n── Prompt Builder ──")

    ctx    = assemble_cluster_context(POTHOLE_CLUSTER)
    prompt = build_user_prompt(ctx)

    check("total reports in prompt", "3" in prompt)
    check("total upvotes in prompt", "7" in prompt)
    check("bounding area in prompt", "Lat [" in prompt)
    check("individual reports block present", "[Report 1]" in prompt)
    check("no unfilled template slots", "{" not in prompt and "}" not in prompt)

    # System prompt checks
    check("system prompt mentions JSON", "JSON" in SYSTEM_PROMPT)
    check("system prompt lists all 8 departments",
          all(d.value in SYSTEM_PROMPT for d in Department))
    check("system prompt defines severity scale", "1" in SYSTEM_PROMPT and "5" in SYSTEM_PROMPT)
    check("system prompt bans markdown fences", "```" in SYSTEM_PROMPT)


# ---------------------------------------------------------------------------
# 4. Response Parser
# ---------------------------------------------------------------------------

def test_parser() -> None:
    print("\n── Response Parser ──")

    # Clean JSON
    clean = json.dumps(_valid_report())
    r = parse_llm_response(clean)
    check("clean JSON parsed", r.severity_level == 3)

    # Fenced JSON (```json ... ```)
    fenced = f"```json\n{clean}\n```"
    r2 = parse_llm_response(fenced)
    check("fenced JSON parsed", r2.severity_level == 3)

    # Bare fences (``` ... ```)
    bare_fenced = f"```\n{clean}\n```"
    r3 = parse_llm_response(bare_fenced)
    check("bare-fenced JSON parsed", r3.severity_level == 3)

    # Whitespace around JSON
    r4 = parse_llm_response(f"\n\n  {clean}  \n")
    check("whitespace stripped", r4.severity_level == 3)

    # Invalid JSON → ValueError
    try:
        parse_llm_response("not json at all")
        check("invalid JSON raises ValueError", False)
    except ValueError:
        check("invalid JSON raises ValueError", True)

    # JSON array instead of object → ValueError
    try:
        parse_llm_response(json.dumps([1, 2, 3]))
        check("JSON array raises ValueError", False)
    except ValueError:
        check("JSON array raises ValueError", True)

    # Schema violation → ValueError
    try:
        parse_llm_response(json.dumps(_valid_report(severity_level=99)))
        check("schema violation raises ValueError", False)
    except (ValueError, Exception):
        check("schema violation raises ValueError", True)

    # Missing field → ValueError
    incomplete = {k: v for k, v in _valid_report().items() if k != "generated_title"}
    try:
        parse_llm_response(json.dumps(incomplete))
        check("missing field raises ValueError", False)
    except (ValueError, Exception):
        check("missing field raises ValueError", True)


# ---------------------------------------------------------------------------
# 5. Orchestrator (end-to-end with MockLLMClient)
# ---------------------------------------------------------------------------

def test_orchestrator() -> None:
    print("\n── Orchestrator ──")

    # Happy path — pothole cluster
    orch   = GeminiOrchestrator(client=MockLLMClient())
    report = orch.analyze_cluster(POTHOLE_CLUSTER)
    check("returns AnalyzedIssueReport", isinstance(report, AnalyzedIssueReport))
    check("severity is 1–5", 1 <= report.severity_level <= 5)
    check("department is Department enum", isinstance(report.department_routed, Department))
    check("generated_title non-empty", len(report.generated_title) > 0)
    check("action plan non-empty", len(report.summary_action_plan) > 0)

    # Keyword routing — water cluster
    water_report = GeminiOrchestrator(client=MockLLMClient()).analyze_cluster(WATER_CLUSTER)
    check("water cluster → Water & Sanitation",
          water_report.department_routed == Department.WATER_AND_SANITATION)

    # Prompt capture — assert prompts were assembled and sent
    mock = MockLLMClient()
    GeminiOrchestrator(client=mock).analyze_cluster(POTHOLE_CLUSTER)
    check("client.generate() called once", len(mock.calls) == 1)
    check("system_prompt in call", mock.calls[0]["system_prompt"] == SYSTEM_PROMPT)
    check("user_prompt contains report block", "[Report 1]" in mock.calls[0]["user_prompt"])
    check("user_prompt contains pothole keyword",
          "pothole" in mock.calls[0]["user_prompt"].lower())

    # Custom response override
    override_json = json.dumps({
        "severity_level": 5,
        "department_routed": Department.EMERGENCY_SERVICES.value,
        "generated_title": "Critical Water Main Burst Causing Flooding",
        "summary_action_plan": "Deploy emergency crew immediately.",
    })
    custom_mock = MockLLMClient(response_override=override_json)
    custom_report = GeminiOrchestrator(client=custom_mock).analyze_cluster(WATER_CLUSTER)
    check("override severity=5", custom_report.severity_level == 5)
    check("override department=Emergency Services",
          custom_report.department_routed == Department.EMERGENCY_SERVICES)

    # Cluster truncation
    big_cluster = [_issue(f"Issue {i}", upvotes=i) for i in range(25)]
    config = OrchestratorConfig(max_cluster_size=5)
    trunc_mock = MockLLMClient()
    GeminiOrchestrator(client=trunc_mock, config=config).analyze_cluster(big_cluster)
    # Prompt should only contain 5 report blocks
    prompt = trunc_mock.calls[0]["user_prompt"]
    check("cluster truncated to 5 reports",
          "[Report 5]" in prompt and "[Report 6]" not in prompt)
    check("highest-upvoted kept (upvotes 24)",
          "Issue 24" in prompt)

    # Empty cluster raises
    try:
        GeminiOrchestrator(client=MockLLMClient()).analyze_cluster([])
        check("empty cluster raises ValueError", False)
    except ValueError:
        check("empty cluster raises ValueError", True)

    # Mock client records multiple calls
    multi_mock = MockLLMClient()
    multi_orch = GeminiOrchestrator(client=multi_mock)
    multi_orch.analyze_cluster(POTHOLE_CLUSTER)
    multi_orch.analyze_cluster(WATER_CLUSTER)
    check("multiple calls recorded", len(multi_mock.calls) == 2)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_schema()
    test_context_assembler()
    test_prompt_builder()
    test_parser()
    test_orchestrator()

    print(f"\n{'='*50}")
    print(f"  {passed} passed, {len(errors)} failed")
    if errors:
        print("  FAILED:", errors)
        sys.exit(1)
    else:
        print("  All tests green.")
