"""
core/models.py
--------------
Foundational data structures for the Community Hero platform.
All domain models live here and are imported by other modules.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


class IssueStatus(str, Enum):
    """Lifecycle state of a reported community issue."""
    PENDING    = "pending"     # Submitted, awaiting triage
    OPEN       = "open"        # Confirmed distinct issue, active
    MERGED     = "merged"      # Deduplicated into another issue
    IN_PROGRESS = "in_progress"
    RESOLVED   = "resolved"
    REJECTED   = "rejected"    # Spam / invalid


class IssueCategory(str, Enum):
    POTHOLE          = "pothole"
    WATER_LEAKAGE    = "water_leakage"
    STREETLIGHT      = "streetlight"
    WASTE            = "waste"
    ROAD_DAMAGE      = "road_damage"
    ENCROACHMENT     = "encroachment"
    OTHER            = "other"


@dataclass
class Coordinates:
    """
    WGS-84 geographic coordinates.

    latitude  : -90.0  to  90.0  (degrees)
    longitude : -180.0 to 180.0  (degrees)
    """
    latitude: float
    longitude: float

    def __post_init__(self) -> None:
        if not (-90.0 <= self.latitude <= 90.0):
            raise ValueError(f"Invalid latitude: {self.latitude}")
        if not (-180.0 <= self.longitude <= 180.0):
            raise ValueError(f"Invalid longitude: {self.longitude}")


@dataclass
class Issue:
    """
    Core domain model for a community-reported issue.

    Fields
    ------
    id               : Stable UUID, assigned on creation.
    coordinates      : Where the issue was reported.
    timestamp        : UTC creation time (auto-set if omitted).
    text_description : Raw user-supplied description text.
    image_embedding  : Float vector from a vision model (e.g. CLIP / Gemini).
                       None until the async embedding pipeline populates it.
    text_embedding   : Float vector from a text encoder.
                       None until the async embedding pipeline populates it.
    status           : Current lifecycle state.
    category         : AI-assigned or user-provided category.
    upvote_count     : Number of citizens who confirmed this issue.
    merged_into_id   : If status == MERGED, the canonical issue this maps to.
    reporter_id      : Opaque user identifier (hashed / anonymised upstream).
    """

    # --- Identity ---
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # --- Spatial ---
    coordinates: Coordinates = field(default_factory=lambda: Coordinates(0.0, 0.0))

    # --- Temporal ---
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # --- Content ---
    text_description: str = ""
    image_embedding: Optional[list[float]] = None   # e.g. 768-d or 1408-d vector
    text_embedding: Optional[list[float]] = None    # e.g. 768-d vector

    # --- Lifecycle ---
    status: IssueStatus = IssueStatus.PENDING
    category: Optional[IssueCategory] = None
    upvote_count: int = 0
    merged_into_id: Optional[str] = None

    # --- Provenance ---
    reporter_id: Optional[str] = None

    def is_active(self) -> bool:
        """True if this issue is still tracking a real open problem."""
        return self.status in (IssueStatus.PENDING, IssueStatus.OPEN, IssueStatus.IN_PROGRESS)
