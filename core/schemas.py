"""
core/schemas.py
---------------
Output schema for the Gemini AI Orchestrator.

Pydantic v2 is the target runtime.  When pydantic is installed the
real BaseModel is used.  When it is not (CI / offline sandbox) a
lightweight stdlib shim provides the identical public API so every
other module stays untouched.

Public API (same regardless of which backend is active)
--------------------------------------------------------
  AnalyzedIssueReport.model_validate(dict)   → AnalyzedIssueReport
  AnalyzedIssueReport.model_json_schema()    → dict
  instance.model_dump()                      → dict
  instance.model_dump_json()                 → str (compact JSON)
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Controlled vocabulary for department routing
# ---------------------------------------------------------------------------

class Department(str, Enum):
    """
    Municipal departments a reported issue can be routed to.
    Extend this enum as the city's org-chart requires.
    """
    WATER_AND_SANITATION = "Water & Sanitation"
    ELECTRICAL           = "Electrical"
    PUBLIC_WORKS         = "Public Works"
    WASTE_MANAGEMENT     = "Waste Management"
    PARKS_AND_RECREATION = "Parks & Recreation"
    TRAFFIC_ENGINEERING  = "Traffic Engineering"
    BUILDING_INSPECTION  = "Building Inspection"
    EMERGENCY_SERVICES   = "Emergency Services"


# ---------------------------------------------------------------------------
# Pydantic shim (used only when pydantic is not installed)
# ---------------------------------------------------------------------------

class _ShimBase:
    """
    Minimal Pydantic-v2-compatible base class built on stdlib.

    Supports:
      - Field-level type coercion for int / str / Enum fields
      - Range validation via optional _validators class var
      - model_validate(), model_dump(), model_dump_json()
      - model_json_schema()  (structural, not full JSON Schema)
    """

    # Subclasses declare:  _fields: dict[str, type]
    # Optional range guards: _validators: dict[str, callable]
    _fields:     dict[str, Any] = {}
    _validators: dict[str, Any] = {}

    def __init__(self, **kwargs: Any) -> None:
        for name, typ in self.__class__._fields.items():
            if name not in kwargs:
                raise ValueError(f"Missing required field: '{name}'")
            raw = kwargs[name]
            # Coerce to declared type (handles str→Enum, str→int, etc.)
            try:
                value = typ(raw) if not isinstance(raw, typ) else raw
            except (ValueError, KeyError) as exc:
                raise ValueError(
                    f"Field '{name}': cannot coerce {raw!r} to {typ.__name__}: {exc}"
                ) from exc
            # Run optional validator
            if name in self.__class__._validators:
                self.__class__._validators[name](value)
            object.__setattr__(self, name, value)

    # Prevent accidental mutation (mirrors Pydantic model behaviour)
    def __setattr__(self, key: str, value: Any) -> None:  # noqa: D105
        raise AttributeError("AnalyzedIssueReport instances are immutable.")

    @classmethod
    def model_validate(cls, obj: dict[str, Any]) -> "_ShimBase":
        """Equivalent to Pydantic's model_validate(dict)."""
        return cls(**obj)

    def model_dump(self) -> dict[str, Any]:
        """Equivalent to Pydantic's .model_dump()."""
        result = {}
        for name in self.__class__._fields:
            val = getattr(self, name)
            result[name] = val.value if isinstance(val, Enum) else val
        return result

    def model_dump_json(self) -> str:
        """Equivalent to Pydantic's .model_dump_json()."""
        return json.dumps(self.model_dump(), ensure_ascii=False)

    @classmethod
    def model_json_schema(cls) -> dict[str, Any]:
        """Return a structural description of the schema (not full JSON Schema)."""
        props: dict[str, Any] = {}
        for name, typ in cls._fields.items():
            if issubclass(typ, Enum):
                props[name] = {
                    "type": "string",
                    "enum": [m.value for m in typ],
                }
            elif typ is int:
                props[name] = {"type": "integer"}
            else:
                props[name] = {"type": "string"}
        return {
            "title": cls.__name__,
            "type": "object",
            "required": list(cls._fields.keys()),
            "properties": props,
        }

    def __repr__(self) -> str:
        pairs = ", ".join(
            f"{k}={getattr(self, k)!r}" for k in self.__class__._fields
        )
        return f"{self.__class__.__name__}({pairs})"


# ---------------------------------------------------------------------------
# Try real Pydantic; fall back to shim transparently
# ---------------------------------------------------------------------------

try:
    from pydantic import BaseModel, Field, field_validator  # type: ignore

    class AnalyzedIssueReport(BaseModel):
        """
        Structured output the Gemini orchestrator must return.

        severity_level     : 1 (cosmetic / low priority) → 5 (emergency)
        department_routed  : The municipal department responsible for resolution.
        generated_title    : Concise 6–12 word title for the issue cluster.
        summary_action_plan: Step-by-step resolution plan (plain prose, ≤150 words).
        """

        severity_level: int = Field(
            ...,
            ge=1,
            le=5,
            description="Severity on a 1–5 scale (5 = emergency).",
        )
        department_routed: Department = Field(
            ...,
            description="Municipal department responsible for resolution.",
        )
        generated_title: str = Field(
            ...,
            min_length=3,
            max_length=120,
            description="Concise human-readable title for this issue cluster.",
        )
        summary_action_plan: str = Field(
            ...,
            min_length=10,
            description="Step-by-step resolution plan in plain prose.",
        )

        @field_validator("generated_title")
        @classmethod
        def _strip_title(cls, v: str) -> str:
            return v.strip()

    _BACKEND = "pydantic"

except ModuleNotFoundError:
    # ── stdlib shim ──────────────────────────────────────────────────────────

    def _validate_severity(v: int) -> None:
        if not (1 <= v <= 5):
            raise ValueError(f"severity_level must be 1–5, got {v}")

    class AnalyzedIssueReport(_ShimBase):  # type: ignore[no-redef]
        """
        Structured output the Gemini orchestrator must return.

        severity_level     : 1 (cosmetic / low priority) → 5 (emergency)
        department_routed  : The municipal department responsible for resolution.
        generated_title    : Concise 6–12 word title for the issue cluster.
        summary_action_plan: Step-by-step resolution plan (plain prose, ≤150 words).
        """

        _fields: dict = {
            "severity_level":      int,
            "department_routed":   Department,
            "generated_title":     str,
            "summary_action_plan": str,
        }
        _validators: dict = {
            "severity_level": _validate_severity,
        }

    _BACKEND = "stdlib-shim"
