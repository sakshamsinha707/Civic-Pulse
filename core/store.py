"""
core/store.py
-------------
A simple in-memory IssueStore implementation.
Used for local development and unit tests.
In production, swap this for a PostgreSQL / PostGIS adapter.
"""

from __future__ import annotations

from core.models import Issue


class InMemoryIssueStore:
    """Thread-unsafe in-memory store.  Fine for single-process dev use."""

    def __init__(self) -> None:
        self._issues: dict[str, Issue] = {}

    def add(self, issue: Issue) -> None:
        self._issues[issue.id] = issue

    def get(self, issue_id: str) -> Issue | None:
        return self._issues.get(issue_id)

    def get_active_issues(self) -> list[Issue]:
        return [i for i in self._issues.values() if i.is_active()]

    def all(self) -> list[Issue]:
        return list(self._issues.values())

    def __len__(self) -> int:
        return len(self._issues)
