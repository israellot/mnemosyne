"""Tests for derived-memory ranking defaults (upstream issue #506).

Two defects made sleep-derived rows outrank the source memories they
paraphrase:

1. ``consolidate_to_episodic()`` omitted the ``tier`` column from its INSERT,
   so consolidation summaries entered at the schema default tier 1 (full 1.0x
   recall weight) for TIER2_DAYS.
2. Model-refresh proposal rows used the LLM's self-reported confidence
   (routinely 0.85-0.95) directly as ranking importance, outranking curated
   content and defeating the injection gate's importance<0.65 drop condition.

Fixes under test: ``tier`` kwarg + ``MNEMOSYNE_CONSOLIDATION_TIER`` env
default (3) on consolidate_to_episodic, and
``MNEMOSYNE_PROPOSAL_IMPORTANCE_CAP`` env cap (0.5) on proposal importance.
"""

import sqlite3
import tempfile
from pathlib import Path

import pytest

from mnemosyne.core.beam import BeamMemory


@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test.db"


def _episodic_row(db_path, memory_id):
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT tier, importance, source FROM episodic_memory WHERE id = ?",
            (memory_id,),
        ).fetchone()
        return dict(row) if row is not None else None
    finally:
        conn.close()


class TestConsolidationTier:
    def test_default_tier_is_3(self, temp_db, monkeypatch):
        monkeypatch.delenv("MNEMOSYNE_CONSOLIDATION_TIER", raising=False)
        beam = BeamMemory(db_path=str(temp_db), session_id="t506")
        mid = beam.consolidate_to_episodic(
            summary="derived summary", source_wm_ids=["a", "b"],
            source="sleep_consolidation",
        )
        row = _episodic_row(temp_db, mid)
        assert row is not None
        assert row["tier"] == 3

    def test_explicit_tier_kwarg_wins(self, temp_db, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_CONSOLIDATION_TIER", "3")
        beam = BeamMemory(db_path=str(temp_db), session_id="t506")
        mid = beam.consolidate_to_episodic(
            summary="explicit tier", source_wm_ids=["a"], tier=2,
        )
        assert _episodic_row(temp_db, mid)["tier"] == 2

    def test_env_var_overrides_default(self, temp_db, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_CONSOLIDATION_TIER", "1")
        beam = BeamMemory(db_path=str(temp_db), session_id="t506")
        mid = beam.consolidate_to_episodic(
            summary="legacy behavior", source_wm_ids=["a"],
        )
        assert _episodic_row(temp_db, mid)["tier"] == 1

    def test_tier_clamped_to_valid_range(self, temp_db, monkeypatch):
        monkeypatch.delenv("MNEMOSYNE_CONSOLIDATION_TIER", raising=False)
        beam = BeamMemory(db_path=str(temp_db), session_id="t506")
        low = beam.consolidate_to_episodic(
            summary="clamp low", source_wm_ids=["a"], tier=0,
        )
        high = beam.consolidate_to_episodic(
            summary="clamp high", source_wm_ids=["a"], tier=9,
        )
        assert _episodic_row(temp_db, low)["tier"] == 1
        assert _episodic_row(temp_db, high)["tier"] == 3

    def test_invalid_env_value_falls_back_to_3(self, temp_db, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_CONSOLIDATION_TIER", "not-a-number")
        beam = BeamMemory(db_path=str(temp_db), session_id="t506")
        mid = beam.consolidate_to_episodic(
            summary="bad env", source_wm_ids=["a"],
        )
        assert _episodic_row(temp_db, mid)["tier"] == 3


class TestProposalImportanceCap:
    """The cap logic lives inline in sleep(); test it via the same expression."""

    def _capped(self, confidence, cap_env=None, monkeypatch=None):
        # Mirrors the sleep() proposal-importance expression.
        import os
        if monkeypatch is not None:
            if cap_env is None:
                monkeypatch.delenv("MNEMOSYNE_PROPOSAL_IMPORTANCE_CAP", raising=False)
            else:
                monkeypatch.setenv("MNEMOSYNE_PROPOSAL_IMPORTANCE_CAP", cap_env)
        try:
            cap = float(os.environ.get("MNEMOSYNE_PROPOSAL_IMPORTANCE_CAP", "0.5"))
        except (TypeError, ValueError):
            cap = 0.5
        return min(float(confidence or 0.5), cap)

    def test_high_confidence_is_capped(self, monkeypatch):
        assert self._capped(0.95, monkeypatch=monkeypatch) == 0.5

    def test_low_confidence_passes_through(self, monkeypatch):
        assert self._capped(0.3, monkeypatch=monkeypatch) == 0.3

    def test_cap_configurable(self, monkeypatch):
        assert self._capped(0.95, cap_env="0.8", monkeypatch=monkeypatch) == 0.8

    def test_sleep_stores_capped_proposal_importance(self, temp_db, monkeypatch):
        """End-to-end: a sleep pass with a stubbed proposal generator stores
        proposal rows at capped importance while metadata keeps raw confidence."""
        monkeypatch.delenv("MNEMOSYNE_PROPOSAL_IMPORTANCE_CAP", raising=False)
        monkeypatch.setenv("MNEMOSYNE_MODEL_REFRESH_AUTO_APPLY", "0")

        from mnemosyne.core import model_refresh

        def fake_proposals(items):
            return [{
                "category": "project",
                "name": "test_slot",
                "body": "test proposal body",
                "confidence": 0.95,
                "evidence_ids": [],
                "action": "update",
                "reason": "test",
            }]

        monkeypatch.setattr(model_refresh, "infer_model_update_proposals", fake_proposals)

        beam = BeamMemory(db_path=str(temp_db), session_id="t506")
        # Seed old rows so sleep() has something to consolidate.
        import sqlite3 as _sq
        from datetime import datetime, timedelta
        conn = _sq.connect(str(temp_db))
        ts = (datetime.now() - timedelta(hours=200)).isoformat()
        conn.executemany(
            "INSERT INTO working_memory (id, content, source, timestamp, session_id) VALUES (?, ?, ?, ?, ?)",
            [(f"t506-{i}", f"content {i}", "conversation", ts, "t506") for i in range(6)],
        )
        conn.commit()
        conn.close()

        beam.sleep()

        conn = _sq.connect(str(temp_db))
        conn.row_factory = _sq.Row
        rows = conn.execute(
            "SELECT importance, metadata_json FROM working_memory WHERE source = 'sleep_model_refresh_proposal'"
        ).fetchall()
        conn.close()
        assert rows, "sleep() stored no proposal rows"
        import json
        for r in rows:
            assert r["importance"] <= 0.5, f"proposal importance {r['importance']} exceeds cap"
            meta = json.loads(r["metadata_json"])
            assert float(meta.get("confidence", 0)) == 0.95, "raw confidence must survive in metadata"
