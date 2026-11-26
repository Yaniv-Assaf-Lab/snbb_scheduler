from __future__ import annotations

__all__ = ["Rule", "build_rules"]

import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from snbb_scheduler.checks import _dir_nonempty, is_complete
from snbb_scheduler.config import Procedure, SchedulerConfig

# Type alias for a rule function
Rule = Callable[[pd.Series], bool]

# Session labels produced by sessions.sanitize_session_id: ses-YYYYMMDDHHMM
_SESSION_DATE_RE = re.compile(r"^ses-(\d{8})")


def build_rules(
    config: SchedulerConfig,
    sessions_df: pd.DataFrame | None = None,
    force: bool = False,
    force_procedures: list[str] | None = None,
    cache: dict | None = None,
    state: pd.DataFrame | None = None,
) -> dict[str, Rule]:
    """Generate a rule function for every procedure in config.

    Each rule returns ``True`` when all of the following hold:

    1. DICOM data exists for the session (``dicom_exists``).
    2. All ``proc.depends_on`` are satisfied (see :func:`_dep_satisfied`).
    3. This procedure's own output is not yet complete.
    4. Its ``requires`` input files exist.
       (Skipped when *force* is ``True`` and the procedure is in
       *force_procedures*, or when *force* is ``True`` and
       *force_procedures* is ``None``.)

    Parameters
    ----------
    config:
        Scheduler configuration containing the procedure registry.
    sessions_df:
        Full sessions DataFrame produced by :func:`~snbb_scheduler.sessions.discover_sessions`.
        Required for correct evaluation of **cross-scope dependencies** —
        that is, a *subject*-scoped procedure that depends on a
        *session*-scoped one (e.g. ``freesurfer`` depends on ``bids_post``).
        When ``None``, cross-scope dependency checking is skipped (safe for
        procedures without cross-scope dependencies).
    force:
        When ``True``, skip the self-completion check for matching procedures.
    force_procedures:
        If given, limit forced re-submission to this procedure name list.
    state:
        Scheduler state; its first submission times start the
        ``accept_partial_after_days`` period (scan date when absent).
    """
    first_submitted = _first_submissions(state)
    return {
        proc.name: _make_rule(
            proc,
            config,
            sessions_df=sessions_df,
            force=force,
            force_procedures=force_procedures,
            cache=cache,
            first_submitted=first_submitted,
        )
        for proc in config.procedures
    }


# ---------------------------------------------------------------------------
# Partial acceptance and input requirements
# ---------------------------------------------------------------------------


def _first_submissions(state: pd.DataFrame | None) -> dict[tuple[str, str, str], datetime]:
    """Earliest ``submitted_at`` (UTC) per (subject, session, procedure)."""
    if state is None or state.empty or "submitted_at" not in state.columns:
        return {}
    times = pd.to_datetime(state["submitted_at"], errors="coerce", utc=True)
    df = state.assign(_t=times).dropna(subset=["_t"])
    earliest = df.groupby(["subject", "session", "procedure"])["_t"].min()
    return {key: ts.to_pydatetime() for key, ts in earliest.items()}


def _session_scan_date(session: str) -> datetime | None:
    """Scan date from a ``ses-YYYYMMDDHHMM`` label, or None."""
    m = _SESSION_DATE_RE.match(str(session))
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y%m%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _partial_accepted(
    proc: Procedure,
    row: pd.Series,
    first_submitted: dict[tuple[str, str, str], datetime],
) -> bool:
    """True if *proc* produced output and its waiting period is over.

    The period starts at the first submission of *proc* for this session, or
    at the scan date when the state file has no record of one.
    """
    days = proc.accept_partial_after_days
    if days is None or not _dir_nonempty(Path(row[f"{proc.name}_path"])):
        return False
    session = "" if proc.scope == "subject" else row["session"]
    started = first_submitted.get((row["subject"], session, proc.name))
    if started is None:
        started = _session_scan_date(row["session"])
    if started is None:
        return False
    return datetime.now(tz=timezone.utc) - started >= timedelta(days=days)


def _requires_met(
    proc: Procedure,
    row: pd.Series,
    config: SchedulerConfig,
    sessions_df: pd.DataFrame | None = None,
) -> bool:
    """True if every ``requires`` glob matches in the session's BIDS directory.

    Subject-scoped: each glob must match in at least one of the subject's sessions.
    """
    if not proc.requires:
        return True
    subject = row["subject"]
    if proc.scope == "subject" and sessions_df is not None:
        sessions = list(sessions_df.loc[sessions_df["subject"] == subject, "session"])
    else:
        sessions = [row["session"]]
    dirs = [config.bids_root / subject / ses for ses in sessions]
    return all(any(any(d.glob(pat)) for d in dirs) for pat in proc.requires)


def _dep_satisfied(
    dep_proc: Procedure,
    row: pd.Series,
    config: SchedulerConfig,
    first_submitted: dict[tuple[str, str, str], datetime],
    cache: dict | None,
    allow_not_applicable: bool,
) -> bool:
    """True if *dep_proc* does not block its dependents for *row*.

    Satisfied when *dep_proc* is complete, or accepted as partial
    (:func:`_partial_accepted`), or — only if *allow_not_applicable* —
    not applicable: its ``requires`` inputs are missing while its own
    dependencies are satisfied (e.g. ``bids_post`` for a session without DWI).
    Callers allow "not applicable" only for dependents that declare their own
    ``requires``; a dependent without them may need the skipped step's output.
    """
    dep_kwargs = _completion_kwargs(dep_proc, row, config)
    if is_complete(dep_proc, row[f"{dep_proc.name}_path"], cache=cache, **dep_kwargs):
        return True
    if _partial_accepted(dep_proc, row, first_submitted):
        return True
    if (
        allow_not_applicable
        and dep_proc.requires
        and dep_proc.scope == "session"
        and not _requires_met(dep_proc, row, config)
    ):
        return all(
            _dep_satisfied(config.get_procedure(d), row, config, first_submitted, cache, True)
            for d in dep_proc.depends_on
        )
    return False


def _completion_kwargs(
    proc: Procedure, row: pd.Series, config: SchedulerConfig
) -> dict:
    """Return extra keyword arguments for ``is_complete`` based on procedure name.

    Some specialised checks require additional context (e.g. the BIDS root
    and subject label) that cannot be derived from the output path alone.
    This helper centralises that mapping so that ``_make_rule`` stays clean.

    The same mapping is used by :func:`~snbb_scheduler.manifest.reconcile_with_filesystem`
    to avoid duplicating the per-procedure kwarg logic.
    """
    subject = row["subject"]
    if proc.name in ("freesurfer", "qsiprep"):
        return {"bids_root": config.bids_root, "subject": subject}
    if proc.name == "qsirecon":
        kwargs: dict = {
            "derivatives_root": config.derivatives_root,
            "subject": subject,
            "session": row["session"],
        }
        fs_path = row.get("freesurfer_path")
        if fs_path is not None:
            kwargs["freesurfer_path"] = fs_path
        if config.qsirecon_spec is not None:
            kwargs["recon_spec"] = config.qsirecon_spec
        return kwargs
    return {}


def _make_rule(
    proc: Procedure,
    config: SchedulerConfig,
    sessions_df: pd.DataFrame | None = None,
    force: bool = False,
    force_procedures: list[str] | None = None,
    cache: dict | None = None,
    first_submitted: dict[tuple[str, str, str], datetime] | None = None,
) -> Rule:
    """Create a rule closure that decides whether *proc* needs to run for a session.

    The returned callable accepts a session row (``pd.Series``) and returns
    ``True`` when **all** of the following hold:

    1. DICOM data exists for the session (``dicom_exists`` is ``True``).
    2. Every procedure in ``proc.depends_on`` is satisfied (:func:`_dep_satisfied`).
    3. This procedure's own output is **not** yet complete on disk.
    4. Its ``requires`` input files exist (:func:`_requires_met`).

    Cross-scope dependencies (``freesurfer`` only)
    -----------------------------------------------
    ``freesurfer`` is subject-scoped and depends on ``bids_post``
    (session-scoped).  A simple per-row check is insufficient — the
    FreeSurfer job should only start once **all** sessions of the subject
    have ``bids_post`` complete (needed for the longitudinal pipeline).

    If *sessions_df* is provided, the rule iterates every session row for
    the current subject and requires ``bids_post`` to be complete in all
    of them.  The helper script then decides whether to run cross-sectional
    only (single session) or the full 3-step longitudinal pipeline.

    Other subject-scoped procedures (e.g. ``qsiprep``, ``qsirecon``) use
    the standard per-row dependency check even when their dependency is
    session-scoped; this preserves the original one-session-at-a-time
    behaviour for those tools.
    """
    # Pre-classify dependencies as same-scope or cross-scope once, outside
    # the inner closure, to avoid repeated work on every rule evaluation.
    #
    # Cross-scope logic applies to ``freesurfer``, which must wait until
    # ALL sessions of the subject have their session-scoped dependency
    # (``bids_post``) complete before the subject-scoped run can start.
    # Other subject-scoped procedures (e.g. ``qsiprep``, ``qsirecon``)
    # check the current session row's dependency path, which is the original
    # per-row behaviour.
    if first_submitted is None:
        first_submitted = {}
    allow_na = bool(proc.requires)  # see _dep_satisfied
    cross_scope_deps: list[str] = []
    same_scope_deps: list[str] = []
    for dep_name in proc.depends_on:
        dep_proc = config.get_procedure(dep_name)
        if proc.name == "freesurfer" and dep_proc.scope == "session":
            cross_scope_deps.append(dep_name)
        else:
            same_scope_deps.append(dep_name)

    def rule(row: pd.Series) -> bool:
        if not row["dicom_exists"]:
            return False

        # ── Same-scope dependencies (existing logic) ──────────────────────
        for dep_name in same_scope_deps:
            dep_proc = config.get_procedure(dep_name)
            if not _dep_satisfied(dep_proc, row, config, first_submitted, cache, allow_na):
                return False

        # ── Cross-scope dependencies ──────────────────────────────────────
        # A subject-scoped procedure depends on a session-scoped one.
        # All sessions of the subject must have the dependency satisfied.
        if cross_scope_deps and sessions_df is not None:
            subject = row["subject"]
            subject_rows = sessions_df[sessions_df["subject"] == subject]
            for dep_name in cross_scope_deps:
                dep_proc = config.get_procedure(dep_name)
                for _, srow in subject_rows.iterrows():
                    if not srow.get("dicom_exists", False):
                        continue
                    if not _dep_satisfied(dep_proc, srow, config, first_submitted, cache, allow_na):
                        return False  # any unsatisfied session → not ready

        # ── Self-completion check ─────────────────────────────────────────
        should_force = force and (
            force_procedures is None or proc.name in force_procedures
        )
        if not should_force:
            self_kwargs = _completion_kwargs(proc, row, config)
            if is_complete(proc, row[f"{proc.name}_path"], cache=cache, **self_kwargs):
                return False

        # ── Input requirements (last: no extra I/O for complete outputs) ──
        return _requires_met(proc, row, config, sessions_df)

    rule.__name__ = f"needs_{proc.name}"
    return rule
