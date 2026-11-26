"""Tests for time-limited acceptance of partial sessions and per-procedure input requirements.

A procedure with ``accept_partial_after_days`` (e.g. ``bids``) keeps its real
completion status (and keeps being resubmitted while incomplete), but once its
output exists and its first submission is older than N days, procedures that
depend on it may go ahead with whatever data is there.  ``requires`` stops a
procedure from being submitted for a session that lacks its input files.
"""
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from snbb_scheduler.config import SchedulerConfig
from snbb_scheduler.manifest import build_manifest
from snbb_scheduler.rules import build_rules
from tests.test_rules import (
    make_row,
    mark_bids_complete,
    mark_bids_post_complete,
    mark_dicom,
)

N_DAYS = 30


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def with_partial(cfg: SchedulerConfig, days: int | None = N_DAYS) -> SchedulerConfig:
    """Return a copy of *cfg* whose ``bids`` procedure accepts partial sessions."""
    procs = [
        replace(p, accept_partial_after_days=days) if p.name == "bids" else p
        for p in cfg.procedures
    ]
    return replace(cfg, procedures=procs)


def mark_bids_partial(row: dict, t1w: bool = True, dwi: bool = False, func: bool = False) -> None:
    """Create an incomplete BIDS session: only the requested modalities."""
    bids_dir = row["bids_path"]
    bids_dir.mkdir(parents=True, exist_ok=True)
    (bids_dir / f"{row['subject']}_{row['session']}_scans.tsv").touch()
    if t1w:
        (bids_dir / "anat").mkdir(exist_ok=True)
        (bids_dir / "anat" / "sub_T1w.nii.gz").touch()
    if dwi:
        (bids_dir / "dwi").mkdir(exist_ok=True)
        for name in ("sub_dir-AP_dwi.nii.gz", "sub_dir-AP_dwi.bvec",
                     "sub_dir-AP_dwi.bval", "sub_dir-PA_dwi.nii.gz"):
            (bids_dir / "dwi" / name).touch()
    if func:
        (bids_dir / "func").mkdir(exist_ok=True)
        (bids_dir / "func" / "sub_task-music1_bold.nii.gz").touch()


def make_state(entries: list[tuple[str, str, str, float]], tz_aware: bool = True) -> pd.DataFrame:
    """Build a state DataFrame; each entry is (subject, session, procedure, days_ago)."""
    now = datetime.now(tz=timezone.utc)
    rows = []
    for subject, session, procedure, days_ago in entries:
        ts = now - timedelta(days=days_ago)
        if not tz_aware:
            ts = ts.replace(tzinfo=None)
        rows.append({
            "subject": subject,
            "session": session,
            "procedure": procedure,
            "status": "complete",
            "submitted_at": ts,
            "job_id": "1",
        })
    return pd.DataFrame(rows)


def recent_session_label(days_ago: int) -> str:
    """Session label ``ses-YYYYMMDDHHMM`` for a scan *days_ago* days back."""
    return "ses-" + (datetime.now() - timedelta(days=days_ago)).strftime("%Y%m%d%H%M")


OLD_SES = "ses-201901011200"


def partial_row(cfg, session=OLD_SES, **kwargs) -> dict:
    row = make_row(cfg, session=session)
    mark_dicom(row)
    mark_bids_partial(row, **kwargs)
    return row


def rule_for(cfg, name, row, state=None, rows=None):
    sessions_df = pd.DataFrame(rows if rows is not None else [row])
    rules = build_rules(cfg, sessions_df=sessions_df, state=state)
    return rules[name](pd.Series(row))


# ---------------------------------------------------------------------------
# Waiting period
# ---------------------------------------------------------------------------


def test_full_session_downstream_runs_immediately(cfg):
    cfg = with_partial(cfg)
    row = make_row(cfg, session=recent_session_label(1))
    mark_dicom(row)
    mark_bids_complete(row)
    state = make_state([("sub-0001", row["session"], "bids", 1)])
    assert rule_for(cfg, "cat", row, state) is True


def test_partial_session_waits_before_period(cfg):
    cfg = with_partial(cfg)
    row = partial_row(cfg)
    state = make_state([("sub-0001", OLD_SES, "bids", 5)])
    assert rule_for(cfg, "cat", row, state) is False


def test_partial_session_goes_ahead_after_period(cfg):
    cfg = with_partial(cfg)
    row = partial_row(cfg)
    state = make_state([("sub-0001", OLD_SES, "bids", N_DAYS + 1)])
    assert rule_for(cfg, "cat", row, state) is True


def test_partial_bids_still_resubmitted_after_period(cfg):
    """Status is unchanged: bids stays incomplete and keeps being reconverted."""
    cfg = with_partial(cfg)
    row = partial_row(cfg)
    state = make_state([("sub-0001", OLD_SES, "bids", N_DAYS + 1)])
    assert rule_for(cfg, "bids", row, state) is True


def test_earliest_submission_starts_the_clock(cfg):
    cfg = with_partial(cfg)
    row = partial_row(cfg)
    state = make_state([
        ("sub-0001", OLD_SES, "bids", 2),
        ("sub-0001", OLD_SES, "bids", N_DAYS + 5),
    ])
    assert rule_for(cfg, "cat", row, state) is True


def test_tz_naive_submitted_at_is_handled(cfg):
    cfg = with_partial(cfg)
    row = partial_row(cfg)
    state = make_state([("sub-0001", OLD_SES, "bids", N_DAYS + 1)], tz_aware=False)
    assert rule_for(cfg, "cat", row, state) is True


def test_old_scan_imported_recently_waits(cfg):
    """An old scan converted for the first time recently must still wait N days."""
    cfg = with_partial(cfg)
    row = partial_row(cfg, session=OLD_SES)
    state = make_state([("sub-0001", OLD_SES, "bids", 3)])
    assert rule_for(cfg, "cat", row, state) is False


def test_no_state_row_falls_back_to_scan_date_old(cfg):
    cfg = with_partial(cfg)
    row = partial_row(cfg, session=OLD_SES)
    assert rule_for(cfg, "cat", row, state=None) is True


def test_no_state_row_falls_back_to_scan_date_recent(cfg):
    cfg = with_partial(cfg)
    session = recent_session_label(3)
    row = partial_row(cfg, session=session)
    assert rule_for(cfg, "cat", row, state=None) is False


def test_unparseable_session_without_state_is_not_accepted(cfg):
    cfg = with_partial(cfg)
    row = partial_row(cfg, session="ses-01")
    assert rule_for(cfg, "cat", row, state=None) is False


def test_nothing_converted_is_not_accepted(cfg):
    cfg = with_partial(cfg)
    row = make_row(cfg, session=OLD_SES)
    mark_dicom(row)
    state = make_state([("sub-0001", OLD_SES, "bids", N_DAYS + 1)])
    assert rule_for(cfg, "cat", row, state) is False


def test_disabled_by_default(cfg):
    """Without accept_partial_after_days the old behaviour is unchanged."""
    row = partial_row(cfg)
    state = make_state([("sub-0001", OLD_SES, "bids", N_DAYS + 100)])
    assert rule_for(cfg, "cat", row, state) is False


# ---------------------------------------------------------------------------
# requires — input guard
# ---------------------------------------------------------------------------


def test_requires_blocks_consumer_without_input(cfg):
    """An accepted session without a T1w must not get CAT12."""
    cfg = with_partial(cfg)
    row = partial_row(cfg, t1w=False, dwi=True)
    assert rule_for(cfg, "cat", row) is False


def test_requires_dwi_for_bids_post_and_qsiprep(cfg):
    cfg = with_partial(cfg)
    row = partial_row(cfg, t1w=True, dwi=False)
    assert rule_for(cfg, "bids_post", row) is False
    assert rule_for(cfg, "qsiprep", row) is False


def test_bids_post_runs_on_accepted_session_with_dwi(cfg):
    cfg = with_partial(cfg)
    row = partial_row(cfg, t1w=False, dwi=True)
    assert rule_for(cfg, "bids_post", row) is True


def test_not_applicable_dependency_counts_as_satisfied(cfg):
    """defacing depends on bids_post; no DWI → bids_post n/a → defacing may run."""
    cfg = with_partial(cfg)
    row = partial_row(cfg, t1w=True, dwi=False)
    assert rule_for(cfg, "defacing", row) is True


def test_not_applicable_needs_its_own_dependencies(cfg):
    """During the waiting period bids_post is not n/a yet → defacing waits."""
    cfg = with_partial(cfg)
    row = partial_row(cfg, t1w=True, dwi=False)
    state = make_state([("sub-0001", OLD_SES, "bids", 2)])
    assert rule_for(cfg, "defacing", row, state) is False


def test_not_applicable_does_not_pass_steps_without_requires(cfg):
    """No DWI → qsiprep n/a; qsirecon (no requires, needs qsiprep output) must not run."""
    from tests.test_rules import mark_freesurfer_complete

    cfg = with_partial(cfg)
    row = partial_row(cfg, t1w=True, dwi=False)
    mark_freesurfer_complete(row, sessions=[OLD_SES])
    assert rule_for(cfg, "qsirecon", row) is False


def test_late_dwi_is_picked_up(cfg):
    """DWI arriving after the period: bids_post (never ran) is submitted."""
    cfg = with_partial(cfg)
    row = partial_row(cfg, t1w=True, dwi=False)
    assert rule_for(cfg, "bids_post", row) is False
    mark_bids_partial(row, t1w=True, dwi=True)
    assert rule_for(cfg, "bids_post", row) is True


def test_requires_met_for_complete_session_unchanged(cfg):
    row = make_row(cfg)
    mark_dicom(row)
    mark_bids_complete(row)
    assert rule_for(cfg, "cat", row) is True
    assert rule_for(cfg, "bids_post", row) is True


# ---------------------------------------------------------------------------
# FreeSurfer cross-scope gate
# ---------------------------------------------------------------------------


def _fs_rows(cfg, ses_b_days_ago_state):
    """ses A: full + bids_post done; ses B: accepted-partial candidate without T1w/DWI."""
    row_a = make_row(cfg, session="ses-201801011200")
    mark_dicom(row_a)
    mark_bids_complete(row_a)
    mark_bids_post_complete(row_a)
    row_b = partial_row(cfg, session=OLD_SES, t1w=False, dwi=False, func=True)
    state = make_state([
        ("sub-0001", "ses-201801011200", "bids", 400),
        ("sub-0001", OLD_SES, "bids", ses_b_days_ago_state),
    ])
    return row_a, row_b, state


def test_freesurfer_not_blocked_by_session_without_t1w(cfg):
    cfg = with_partial(cfg)
    row_a, row_b, state = _fs_rows(cfg, N_DAYS + 1)
    assert rule_for(cfg, "freesurfer", row_a, state, rows=[row_a, row_b]) is True


def test_freesurfer_waits_for_partial_session_in_period(cfg):
    cfg = with_partial(cfg)
    row_a, row_b, state = _fs_rows(cfg, 2)
    assert rule_for(cfg, "freesurfer", row_a, state, rows=[row_a, row_b]) is False


def test_freesurfer_not_submitted_when_no_session_has_t1w(cfg):
    """All sessions accepted but none has a T1w → FS would only fail; don't submit."""
    cfg = with_partial(cfg)
    row = partial_row(cfg, t1w=False, dwi=True)
    state = make_state([("sub-0001", OLD_SES, "bids", N_DAYS + 1)])
    assert rule_for(cfg, "freesurfer", row, state) is False


def test_late_t1w_makes_freesurfer_incomplete_again(cfg):
    """FS done on ses A; a T1w later appears in accepted ses B → FS must rerun."""
    from tests.test_rules import mark_freesurfer_complete

    cfg = with_partial(cfg)
    row_a, row_b, state = _fs_rows(cfg, N_DAYS + 1)
    (cfg.bids_root / "sub-0001" / "ses-201801011200" / "anat" / "sub_T1w.nii.gz").touch()
    mark_freesurfer_complete(row_a, sessions=["ses-201801011200"])
    rows = [row_a, row_b]
    assert rule_for(cfg, "freesurfer", row_a, state, rows=rows) is False

    mark_bids_partial(row_b, t1w=True, func=True)
    assert rule_for(cfg, "freesurfer", row_a, state, rows=rows) is True


def test_freesurfer_still_blocked_without_feature(cfg):
    row_a, row_b, state = _fs_rows(cfg, N_DAYS + 100)
    assert rule_for(cfg, "freesurfer", row_a, state, rows=[row_a, row_b]) is False


# ---------------------------------------------------------------------------
# Manifest (incl. the cache fast path) and config
# ---------------------------------------------------------------------------


def test_manifest_includes_consumer_of_accepted_session(cfg):
    cfg = with_partial(cfg)
    row = partial_row(cfg)
    sessions = pd.DataFrame([row])
    state = make_state([("sub-0001", OLD_SES, "bids", N_DAYS + 1)])
    manifest = build_manifest(sessions, cfg, cache={}, state=state)
    procs = set(manifest["procedure"])
    assert "bids" in procs  # still reconverted
    assert "cat" in procs  # goes ahead despite bids incomplete


def test_manifest_excludes_consumer_in_waiting_period(cfg):
    cfg = with_partial(cfg)
    row = partial_row(cfg)
    sessions = pd.DataFrame([row])
    state = make_state([("sub-0001", OLD_SES, "bids", 2)])
    manifest = build_manifest(sessions, cfg, cache={}, state=state)
    assert "cat" not in set(manifest["procedure"])


def test_yaml_parses_new_fields(tmp_path):
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text(
        "procedures:\n"
        "  - name: bids\n"
        "    output_dir: ''\n"
        "    script: snbb_run_bids.sh\n"
        "    accept_partial_after_days: 30\n"
        "  - name: cat\n"
        "    output_dir: cat12\n"
        "    script: snbb_run_cat.sh\n"
        "    depends_on: [bids]\n"
        "    requires: ['anat/*_T1w.nii.gz']\n"
    )
    cfg = SchedulerConfig.from_yaml(yaml_file)
    assert cfg.get_procedure("bids").accept_partial_after_days == 30
    assert cfg.get_procedure("cat").requires == ["anat/*_T1w.nii.gz"]
    assert cfg.get_procedure("cat").accept_partial_after_days is None
