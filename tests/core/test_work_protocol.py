from __future__ import annotations

from ccdl_comm.collectives.work import (
    CollectiveWork as LegacyCollectiveWork,
    CompletionWork as LegacyCompletionWork,
    ImmediateWork as LegacyImmediateWork,
)
from ccdl_comm.work import CollectiveWork, CompletionWork, ImmediateWork


def test_legacy_work_module_reexports_core_implementations() -> None:
    assert LegacyCollectiveWork is CollectiveWork
    assert LegacyCompletionWork is CompletionWork
    assert LegacyImmediateWork is ImmediateWork


def test_immediate_work_returns_completed_result() -> None:
    work = ImmediateWork("done")

    assert work.query() is True
    assert work.wait() == "done"
