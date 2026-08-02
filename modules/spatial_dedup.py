"""
modules/spatial_dedup.py
------------------------
Spatial Deduplication & Clustering Module
==========================================

Responsibility
--------------
Given a newly incoming Issue, decide whether it is:
  (a) A DUPLICATE of an existing active issue  → return that issue's ID so the
      caller can increment its upvote_count and set the new report as MERGED.
  (b) A NEW DISTINCT issue                     → return None so the caller
      creates a fresh issue record.

Data-flow
---------

  IncomingIssue
       │
       ▼
  ┌──────────────────────────────────────┐
  │  1. Geospatial Pre-filter            │
  │     Haversine distance < RADIUS_M    │  ← fast, no ML cost
  │     Keeps only "nearby" candidates   │
  └────────────────┬─────────────────────┘
                   │  candidate_issues[]
                   ▼
  ┌──────────────────────────────────────┐
  │  2. Embedding Similarity Check       │
  │     Cosine similarity on vectors     │  ← ranking step
  │     Image embedding (primary)        │
  │     Text  embedding (secondary)      │
  └────────────────┬─────────────────────┘
                   │  best_match or None
                   ▼
  ┌──────────────────────────────────────┐
  │  3. Decision                         │
  │     score >= SIMILARITY_THRESHOLD    │
  │       → DeduplicationResult(MERGED)  │
  │     score <  SIMILARITY_THRESHOLD    │
  │       → DeduplicationResult(NEW)     │
  └──────────────────────────────────────┘

Design decisions
----------------
* Haversine is computed in pure Python (no PostGIS dependency needed for the
  module's unit tests).  In production, step 1 is replaced by a PostGIS or
  pgvector bounding-box query that pushes the spatial filter into the DB index.
* Cosine similarity is implemented from scratch to keep the module dependency-
  free.  In production, swap in numpy/faiss/pgvector for batch efficiency.
* The module is stateless: it receives an issue store interface so it can be
  tested with a simple list and wired to a real DB adapter later.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Protocol, runtime_checkable

from core.models import Issue, IssueStatus


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

DEDUP_RADIUS_METERS: float = 500.0
"""Spatial pre-filter radius.  Issues farther than this cannot be duplicates."""

IMAGE_SIMILARITY_THRESHOLD: float = 0.88
"""
Cosine similarity cutoff for image embeddings.
Tune upward (→ 0.95) for stricter dedup, downward (→ 0.80) to be more
aggressive about merging.
"""

TEXT_SIMILARITY_THRESHOLD: float = 0.82
"""Cosine similarity cutoff for text embeddings (used as fallback)."""

TEXT_SIMILARITY_WEIGHT: float = 0.35
IMAGE_SIMILARITY_WEIGHT: float = 0.65
"""
When BOTH embeddings are available, the final score is a weighted average.
Image is weighted higher because it encodes physical appearance.
"""

EARTH_RADIUS_METERS: float = 6_371_000.0


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

class DeduplicationOutcome(str, Enum):
    MERGED = "merged"   # New report is a duplicate; should be merged
    NEW    = "new"      # New report is a distinct issue


@dataclass(frozen=True)
class DeduplicationResult:
    """
    Returned by `deduplicate_issue`.

    outcome          : MERGED or NEW.
    existing_issue_id: The canonical issue to merge into (MERGED only).
    similarity_score : Highest combined similarity score found (0–1).
    match_reason     : Human-readable explanation for logging / audit.
    """
    outcome: DeduplicationOutcome
    existing_issue_id: Optional[str] = None
    similarity_score: float = 0.0
    match_reason: str = ""


# ---------------------------------------------------------------------------
# Issue store protocol (dependency-injected; easily mockable)
# ---------------------------------------------------------------------------

@runtime_checkable
class IssueStore(Protocol):
    """
    Minimal interface the dedup module needs from the persistence layer.
    Concrete implementations can be PostgreSQL, Redis, in-memory list, etc.
    """
    def get_active_issues(self) -> list[Issue]:
        """Return all issues that are in an active (non-terminal) state."""
        ...


# ---------------------------------------------------------------------------
# Haversine distance
# ---------------------------------------------------------------------------

def haversine_distance(
    lat1: float, lon1: float,
    lat2: float, lon2: float,
) -> float:
    """
    Great-circle distance between two WGS-84 points, in metres.

    Formula
    -------
    a = sin²(Δlat/2) + cos(lat1)·cos(lat2)·sin²(Δlon/2)
    c = 2·atan2(√a, √(1−a))
    d = R·c

    Accuracy is ±0.5 % compared to an ellipsoidal model, sufficient for
    sub-kilometre urban deduplication.
    """
    lat1_r = math.radians(lat1)
    lat2_r = math.radians(lat2)
    d_lat  = math.radians(lat2 - lat1)
    d_lon  = math.radians(lon2 - lon1)

    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(d_lon / 2) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return EARTH_RADIUS_METERS * c


def filter_by_radius(
    incoming: Issue,
    candidates: list[Issue],
    radius_meters: float = DEDUP_RADIUS_METERS,
) -> list[tuple[Issue, float]]:
    """
    Return (issue, distance_m) pairs for all candidates within `radius_meters`
    of `incoming`, sorted nearest-first.

    Only active issues are eligible; MERGED / RESOLVED issues are skipped so
    we don't deduplicate against already-closed reports.
    """
    results: list[tuple[Issue, float]] = []

    inc_lat = incoming.coordinates.latitude
    inc_lon = incoming.coordinates.longitude

    for issue in candidates:
        if not issue.is_active():
            continue
        if issue.id == incoming.id:
            continue  # Never compare against itself

        dist = haversine_distance(
            inc_lat, inc_lon,
            issue.coordinates.latitude,
            issue.coordinates.longitude,
        )
        if dist <= radius_meters:
            results.append((issue, dist))

    results.sort(key=lambda x: x[1])  # nearest first
    return results


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Cosine similarity between two equal-length float vectors.

    Returns a value in [-1, 1]; for embedding vectors from reputable encoders
    it will typically be in [0, 1].

    Raises
    ------
    ValueError  if vectors have different lengths or are zero-norm.
    """
    if len(vec_a) != len(vec_b):
        raise ValueError(
            f"Vector length mismatch: {len(vec_a)} vs {len(vec_b)}"
        )

    dot   = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0.0 or norm_b == 0.0:
        raise ValueError("Cannot compute cosine similarity for a zero vector.")

    return dot / (norm_a * norm_b)


def combined_similarity(incoming: Issue, candidate: Issue) -> tuple[float, str]:
    """
    Compute a combined similarity score between two issues.

    Strategy
    --------
    ┌─────────────────────────────┬──────────────────────────────────────────┐
    │ Embeddings available        │ Strategy                                 │
    ├─────────────────────────────┼──────────────────────────────────────────┤
    │ Both image + text           │ Weighted average (image 65%, text 35%)   │
    │ Image only                  │ Image cosine similarity                  │
    │ Text only                   │ Text cosine similarity                   │
    │ Neither                     │ Score = 0.0 (cannot compare)             │
    └─────────────────────────────┴──────────────────────────────────────────┘

    Returns
    -------
    (score, reason_string)
    """
    has_image = (
        incoming.image_embedding is not None
        and candidate.image_embedding is not None
    )
    has_text = (
        incoming.text_embedding is not None
        and candidate.text_embedding is not None
    )

    if not has_image and not has_text:
        return 0.0, "no_embeddings_available"

    if has_image and has_text:
        img_score  = cosine_similarity(
            incoming.image_embedding,   # type: ignore[arg-type]
            candidate.image_embedding,  # type: ignore[arg-type]
        )
        text_score = cosine_similarity(
            incoming.text_embedding,    # type: ignore[arg-type]
            candidate.text_embedding,   # type: ignore[arg-type]
        )
        final = (
            IMAGE_SIMILARITY_WEIGHT * img_score
            + TEXT_SIMILARITY_WEIGHT * text_score
        )
        reason = (
            f"image={img_score:.3f} text={text_score:.3f} "
            f"combined={final:.3f}"
        )
        return final, reason

    if has_image:
        score = cosine_similarity(
            incoming.image_embedding,   # type: ignore[arg-type]
            candidate.image_embedding,  # type: ignore[arg-type]
        )
        return score, f"image_only={score:.3f}"

    # Text only
    score = cosine_similarity(
        incoming.text_embedding,    # type: ignore[arg-type]
        candidate.text_embedding,   # type: ignore[arg-type]
    )
    return score, f"text_only={score:.3f}"


def _effective_threshold(incoming: Issue, candidate: Issue) -> float:
    """
    Decide which threshold applies based on available embedding types.
    When only text embeddings are present we use the (lower) text threshold.
    """
    has_image = (
        incoming.image_embedding is not None
        and candidate.image_embedding is not None
    )
    return IMAGE_SIMILARITY_THRESHOLD if has_image else TEXT_SIMILARITY_THRESHOLD


# ---------------------------------------------------------------------------
# Primary public API
# ---------------------------------------------------------------------------

def deduplicate_issue(
    incoming: Issue,
    store: IssueStore,
    radius_meters: float = DEDUP_RADIUS_METERS,
) -> DeduplicationResult:
    """
    Determine whether `incoming` duplicates an existing active issue.

    Parameters
    ----------
    incoming      : The newly submitted issue (not yet persisted).
    store         : Provides access to existing active issues.
    radius_meters : Override the default search radius for this call.

    Returns
    -------
    DeduplicationResult
        .outcome == MERGED  → caller should increment existing issue's
                              upvote_count and set incoming.status = MERGED.
        .outcome == NEW     → caller should persist `incoming` as a new issue.

    Algorithm
    ---------
    1. Pull all active issues from the store.
    2. Spatial pre-filter: keep only those within `radius_meters`.
    3. For each spatial candidate, compute combined_similarity().
    4. Track the best (highest-scoring) candidate.
    5. If best score >= threshold → MERGED; else → NEW.
    """
    all_active = store.get_active_issues()

    # --- Step 1: Spatial pre-filter ---
    nearby: list[tuple[Issue, float]] = filter_by_radius(
        incoming, all_active, radius_meters
    )

    if not nearby:
        return DeduplicationResult(
            outcome=DeduplicationOutcome.NEW,
            match_reason="no_nearby_issues",
        )

    # --- Step 2 & 3: Similarity ranking ---
    best_issue: Optional[Issue] = None
    best_score: float = 0.0
    best_reason: str = ""

    for candidate, dist_m in nearby:
        score, reason = combined_similarity(incoming, candidate)
        if score > best_score:
            best_score  = score
            best_issue  = candidate
            best_reason = f"dist={dist_m:.1f}m {reason}"

    # --- Step 4: Decision ---
    if best_issue is None:
        # All nearby issues had no embeddings; cannot deduplicate
        return DeduplicationResult(
            outcome=DeduplicationOutcome.NEW,
            match_reason="nearby_issues_have_no_embeddings",
        )

    threshold = _effective_threshold(incoming, best_issue)

    if best_score >= threshold:
        return DeduplicationResult(
            outcome=DeduplicationOutcome.MERGED,
            existing_issue_id=best_issue.id,
            similarity_score=best_score,
            match_reason=best_reason,
        )

    return DeduplicationResult(
        outcome=DeduplicationOutcome.NEW,
        similarity_score=best_score,
        match_reason=f"below_threshold({threshold}) {best_reason}",
    )
