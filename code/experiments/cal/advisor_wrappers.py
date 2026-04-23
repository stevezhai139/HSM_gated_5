"""Black-box wrappers for downstream index advisors — Dexter + Supabase
`index_advisor`.

These advisors are already wired into Paper 3A's `tier2/` end-to-end
pipeline. The wrappers here expose a uniform `.invoke(window)` API returning
`AdvisorOutput` — timing + recommended CREATE INDEX statements + cost
estimates — so the gate (`gate.py`) can be advisor-agnostic.

NOTE: These wrappers must not modify any module in the tier2/ subtree.
Import-only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class AdvisorOutput:
    """Result of a single advisor invocation."""

    window_index: int
    advisor_name: str
    create_index_stmts: list[str]
    estimated_plan_cost_delta: float  # higher is better; positive = improvement
    wall_seconds: float               # c_adv(X, W) — wall-clock seconds
    invoked: bool                     # False when gate short-circuited
    # If gate skipped invocation, the above fields reflect the inherited
    # (previous) recommendation — NOT zero values.


class AdvisorWrapper(Protocol):
    """Uniform advisor protocol for gate consumers."""

    name: str

    def invoke(self, window: object, *, timeout_s: float = 30.0) -> AdvisorOutput:
        ...


@dataclass
class DexterWrapper:
    """Wrapper around the Dexter CLI tool.

    Shelling out to `dexter --dsn=$DSN --min-calls=20 ...` in the simplest
    form; switch to library invocation if Dexter exposes a Python API.

    See: <https://github.com/ankane/dexter>
    """

    dsn: str
    min_calls: int = 20
    create: bool = False  # Safety: default to "log only", never auto-create
    name: str = "dexter"

    def invoke(self, window: object, *, timeout_s: float = 30.0) -> AdvisorOutput:
        raise NotImplementedError(
            "DexterWrapper.invoke: TODO — materialise the window as a "
            "pg_stat_statements snapshot, run Dexter, parse output."
        )


@dataclass
class SupabaseIndexAdvisorWrapper:
    """Wrapper around the Supabase `index_advisor` Postgres extension.

    Called via SQL: `SELECT * FROM index_advisor(query)` for each query
    in the window, then de-duplicated.

    See: <https://github.com/supabase/index_advisor>
    """

    dsn: str
    name: str = "supabase_index_advisor"

    def invoke(self, window: object, *, timeout_s: float = 30.0) -> AdvisorOutput:
        raise NotImplementedError(
            "SupabaseIndexAdvisorWrapper.invoke: TODO — iterate over window "
            "queries, call extension via psycopg, collect recommendations."
        )


@dataclass
class NullAdvisor:
    """Test double: returns a deterministic AdvisorOutput with no actual DB."""

    name: str = "null"

    def invoke(self, window: object, *, timeout_s: float = 30.0) -> AdvisorOutput:
        return AdvisorOutput(
            window_index=0,
            advisor_name=self.name,
            create_index_stmts=[],
            estimated_plan_cost_delta=0.0,
            wall_seconds=0.0,
            invoked=True,
        )


__all__ = [
    "AdvisorWrapper",
    "AdvisorOutput",
    "DexterWrapper",
    "SupabaseIndexAdvisorWrapper",
    "NullAdvisor",
]
