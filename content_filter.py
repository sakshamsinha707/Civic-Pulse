"""
content_filter.py
-----------------
Stage 1: Heuristic Content Filter

Rejects submissions that are clearly spam, gibberish, or structurally
invalid before the expensive AI pipeline runs.

Design
------
* Pure stdlib — no ML dependencies.
* Returns a FilterResult dataclass so callers can inspect the reason.
* Conservative by default: only hard-rejects obvious bad input.
  Edge cases are passed through to the AI for a smarter decision.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class FilterResult:
    is_valid: bool
    rejection_code: Optional[str] = None
    rejection_detail: Optional[str] = None


class HeuristicContentFilter:
    """
    Fast heuristic filter applied before spatial dedup and AI orchestration.

    Checks (in order)
    -----------------
    1. Text length  — must be between MIN_LEN and MAX_LEN characters.
    2. Gibberish    — rejects strings that are >80% non-alphanumeric / space.
    3. Repetition   — rejects strings where a single repeated character
                      makes up >60% of the text (e.g. "aaaaaaaaaa").
    4. Spam phrases — hard-coded blocklist of common spam patterns.
    5. Image size   — if provided, rejects images > MAX_IMAGE_BYTES.
    """

    MIN_LEN = 10
    MAX_LEN = 2_000
    MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB

    # Compiled once at class definition time
    _SPAM_RE = re.compile(
        r"\b(test\s*123|hello\s*world|asdf|qwerty|lorem\s*ipsum|"
        r"click\s*here|buy\s*now|free\s*money|xxx)\b",
        re.IGNORECASE,
    )

    def filter_content(
        self,
        text: str,
        image_bytes: Optional[bytes] = None,
    ) -> FilterResult:
        # ── 1. Length ────────────────────────────────────────────────────────
        stripped = text.strip()
        if len(stripped) < self.MIN_LEN:
            return FilterResult(
                is_valid=False,
                rejection_code="TEXT_TOO_SHORT",
                rejection_detail=(
                    f"Description must be at least {self.MIN_LEN} characters. "
                    f"Got {len(stripped)}."
                ),
            )
        if len(stripped) > self.MAX_LEN:
            return FilterResult(
                is_valid=False,
                rejection_code="TEXT_TOO_LONG",
                rejection_detail=(
                    f"Description must be at most {self.MAX_LEN} characters. "
                    f"Got {len(stripped)}."
                ),
            )

        # ── 2. Gibberish (non-alphanumeric ratio) ────────────────────────────
        alnum_space = sum(1 for c in stripped if c.isalnum() or c.isspace())
        if len(stripped) > 0 and (alnum_space / len(stripped)) < 0.2:
            return FilterResult(
                is_valid=False,
                rejection_code="GIBBERISH_TEXT",
                rejection_detail="Submission appears to contain mostly symbols or gibberish.",
            )

        # ── 3. Repetition ────────────────────────────────────────────────────
        if stripped:
            most_common_char_count = max(stripped.lower().count(c) for c in set(stripped.lower()))
            if most_common_char_count / len(stripped) > 0.6 and len(stripped) > 15:
                return FilterResult(
                    is_valid=False,
                    rejection_code="REPETITIVE_TEXT",
                    rejection_detail="Submission appears to be repetitive or auto-generated.",
                )

        # ── 4. Spam phrases ──────────────────────────────────────────────────
        if self._SPAM_RE.search(stripped):
            return FilterResult(
                is_valid=False,
                rejection_code="SPAM_CONTENT",
                rejection_detail="Submission matches known spam patterns.",
            )

        # ── 5. Image size ────────────────────────────────────────────────────
        if image_bytes is not None and len(image_bytes) > self.MAX_IMAGE_BYTES:
            return FilterResult(
                is_valid=False,
                rejection_code="IMAGE_TOO_LARGE",
                rejection_detail=(
                    f"Image must be under {self.MAX_IMAGE_BYTES // (1024*1024)} MB. "
                    f"Got {len(image_bytes) // (1024*1024)} MB."
                ),
            )

        return FilterResult(is_valid=True)
