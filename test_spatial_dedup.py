"""
tests/test_spatial_dedup.py
---------------------------
Covers all four decision branches of deduplicate_issue() plus the
Haversine and cosine helpers.

Run with:  python -m pytest tests/ -v
"""

from __future__ import annotations

import math
import sys
import os

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from core.models import Coordinates, Issue, IssueStatus
from core.store import InMemoryIssueStore
from modules.spatial_dedup import (
    DEDUP_RADIUS_METERS,
    DeduplicationOutcome,
    cosine_similarity,
    deduplicate_issue,
    filter_by_radius,
    haversine_distance,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit_vec(dim: int, hot_index: int = 0) -> list[float]:
    """Return a unit vector with 1.0 at hot_index, 0.0 elsewhere."""
    v = [0.0] * dim
    v[hot_index] = 1.0
    return v


def _make_issue(
    lat: float,
    lon: float,
    image_emb: list[float] | None = None,
    text_emb: list[float] | None = None,
    status: IssueStatus = IssueStatus.OPEN,
) -> Issue:
    return Issue(
        coordinates=Coordinates(lat, lon),
        image_embedding=image_emb,
        text_embedding=text_emb,
        status=status,
    )


# Bangalore city centre (approximate)
BASE_LAT, BASE_LON = 12.9716, 77.5946


# ---------------------------------------------------------------------------
# Haversine tests
# ---------------------------------------------------------------------------

class TestHaversine:
    def test_same_point_is_zero(self):
        assert haversine_distance(BASE_LAT, BASE_LON, BASE_LAT, BASE_LON) == 0.0

    def test_known_distance(self):
        # ~111 km per degree of latitude
        dist = haversine_distance(0.0, 0.0, 1.0, 0.0)
        assert 111_000 < dist < 112_000

    def test_symmetry(self):
        d1 = haversine_distance(BASE_LAT, BASE_LON, BASE_LAT + 0.002, BASE_LON)
        d2 = haversine_distance(BASE_LAT + 0.002, BASE_LON, BASE_LAT, BASE_LON)
        assert math.isclose(d1, d2, rel_tol=1e-9)

    def test_within_500m(self):
        # ~0.0045° of latitude ≈ 500 m
        dist = haversine_distance(BASE_LAT, BASE_LON, BASE_LAT + 0.004, BASE_LON)
        assert dist < 500.0

    def test_outside_500m(self):
        dist = haversine_distance(BASE_LAT, BASE_LON, BASE_LAT + 0.01, BASE_LON)
        assert dist > 500.0


# ---------------------------------------------------------------------------
# Cosine similarity tests
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        assert math.isclose(cosine_similarity(v, v), 1.0, rel_tol=1e-9)

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert math.isclose(cosine_similarity(a, b), 0.0, abs_tol=1e-9)

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert math.isclose(cosine_similarity(a, b), -1.0, rel_tol=1e-9)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length mismatch"):
            cosine_similarity([1.0, 2.0], [1.0])

    def test_zero_vector_raises(self):
        with pytest.raises(ValueError, match="zero vector"):
            cosine_similarity([0.0, 0.0], [1.0, 0.0])

    def test_similar_embeddings_high_score(self):
        a = [0.9, 0.1, 0.05]
        b = [0.88, 0.12, 0.04]
        assert cosine_similarity(a, b) > 0.99


# ---------------------------------------------------------------------------
# filter_by_radius tests
# ---------------------------------------------------------------------------

class TestFilterByRadius:
    def test_nearby_issue_included(self):
        incoming  = _make_issue(BASE_LAT, BASE_LON)
        nearby    = _make_issue(BASE_LAT + 0.001, BASE_LON)  # ~111 m away
        results   = filter_by_radius(incoming, [nearby])
        assert len(results) == 1
        assert results[0][0].id == nearby.id

    def test_far_issue_excluded(self):
        incoming = _make_issue(BASE_LAT, BASE_LON)
        far      = _make_issue(BASE_LAT + 0.1, BASE_LON)    # ~11 km
        results  = filter_by_radius(incoming, [far])
        assert results == []

    def test_resolved_issue_skipped(self):
        incoming  = _make_issue(BASE_LAT, BASE_LON)
        resolved  = _make_issue(BASE_LAT + 0.001, BASE_LON, status=IssueStatus.RESOLVED)
        results   = filter_by_radius(incoming, [resolved])
        assert results == []

    def test_sorted_nearest_first(self):
        incoming = _make_issue(BASE_LAT, BASE_LON)
        far_but_still_inside  = _make_issue(BASE_LAT + 0.003, BASE_LON)
        near                  = _make_issue(BASE_LAT + 0.001, BASE_LON)
        results = filter_by_radius(incoming, [far_but_still_inside, near])
        assert results[0][0].id == near.id

    def test_self_excluded(self):
        issue   = _make_issue(BASE_LAT, BASE_LON)
        results = filter_by_radius(issue, [issue])
        assert results == []


# ---------------------------------------------------------------------------
# deduplicate_issue integration tests
# ---------------------------------------------------------------------------

class TestDeduplicateIssue:

    # --- Branch A: No nearby issues → NEW ---
    def test_no_nearby_issues_returns_new(self):
        store    = InMemoryIssueStore()
        incoming = _make_issue(BASE_LAT, BASE_LON)
        result   = deduplicate_issue(incoming, store)
        assert result.outcome == DeduplicationOutcome.NEW

    # --- Branch B: Nearby but no embeddings → NEW ---
    def test_nearby_no_embeddings_returns_new(self):
        store     = InMemoryIssueStore()
        existing  = _make_issue(BASE_LAT + 0.001, BASE_LON)   # no embeddings
        store.add(existing)
        incoming  = _make_issue(BASE_LAT, BASE_LON)
        result    = deduplicate_issue(incoming, store)
        assert result.outcome == DeduplicationOutcome.NEW
        assert "no_embeddings" in result.match_reason

    # --- Branch C: Nearby + high similarity → MERGED ---
    def test_high_similarity_returns_merged(self):
        store    = InMemoryIssueStore()
        vec      = _unit_vec(128, 0)
        existing = _make_issue(BASE_LAT + 0.001, BASE_LON, image_emb=vec)
        store.add(existing)

        # Slightly perturbed vector (still very close)
        perturbed = [v + 0.001 for v in vec]
        incoming  = _make_issue(BASE_LAT, BASE_LON, image_emb=perturbed)
        result    = deduplicate_issue(incoming, store)

        assert result.outcome == DeduplicationOutcome.MERGED
        assert result.existing_issue_id == existing.id
        assert result.similarity_score > 0.88

    # --- Branch D: Nearby + low similarity → NEW ---
    def test_low_similarity_returns_new(self):
        store    = InMemoryIssueStore()
        vec_a    = _unit_vec(128, 0)
        vec_b    = _unit_vec(128, 64)   # orthogonal → similarity = 0
        existing = _make_issue(BASE_LAT + 0.001, BASE_LON, image_emb=vec_a)
        store.add(existing)

        incoming = _make_issue(BASE_LAT, BASE_LON, image_emb=vec_b)
        result   = deduplicate_issue(incoming, store)

        assert result.outcome == DeduplicationOutcome.NEW

    # --- Weighted both embeddings ---
    def test_both_embeddings_used(self):
        store = InMemoryIssueStore()
        img   = _unit_vec(64, 0)
        txt   = _unit_vec(64, 1)

        existing = _make_issue(
            BASE_LAT + 0.001, BASE_LON,
            image_emb=img, text_emb=txt,
        )
        store.add(existing)

        incoming = _make_issue(
            BASE_LAT, BASE_LON,
            image_emb=[v + 0.001 for v in img],  # near-identical image
            text_emb=[v + 0.001 for v in txt],   # near-identical text
        )
        result = deduplicate_issue(incoming, store)
        # Combined near-identical embeddings should exceed both thresholds
        assert result.outcome == DeduplicationOutcome.MERGED

    # --- Outside radius → NEW regardless of similarity ---
    def test_outside_radius_is_new_despite_high_similarity(self):
        store = InMemoryIssueStore()
        vec   = _unit_vec(64, 0)
        existing = _make_issue(BASE_LAT + 0.1, BASE_LON, image_emb=vec)  # 11 km
        store.add(existing)

        incoming = _make_issue(BASE_LAT, BASE_LON, image_emb=vec)
        result   = deduplicate_issue(incoming, store)
        assert result.outcome == DeduplicationOutcome.NEW
