"""Layer-2 CI: run the whole pipeline on a few real sessions, in a scratch dir.

Runs on the lab server (GitHub self-hosted runner) after each merge to main,
or on demand. It processes a small, fixed list of real sessions from raw DICOM
through every procedure, using the code of the commit under test, and fails if
any procedure fails or does not complete.

Isolation from production
-------------------------
The run scripts in ``scripts/`` do not take their output paths from the
scheduler config: each has its own ``SNBB_*`` environment variables whose
defaults point at the production tree. ``sbatch`` passes the submitting
environment to the job, so this driver:

* rewrites the scheduler config so every writable path (BIDS root,
  derivatives, state, logs, cache) lives under the work dir, and every
  procedure script is the checked-out copy;
* sets every output ``SNBB_*`` variable to the matching path under the work
  dir, and the code inputs (heuristic, recon spec, BIDS filter, helper-script
  dir) to the checked-out copies;
* refuses to start if any of that would touch production paths, and refuses
  to wipe a work dir it did not create.

Read-only inputs (raw DICOM, containers, FreeSurfer license, atlases,
templateflow) are left at the scripts' defaults, or whatever the runner's own
environment sets.

Privacy
-------
Workflow logs on a public repository are public. Before anything else, the
driver emits ``::add-mask::`` for every subject/session identifier so GitHub
replaces them with ``***``, and the summary it writes uses anonymous labels.

Usage
-----
    python ci/pipeline_test.py --base-config /path/to/production/config.yaml \
        --sessions /path/to/ci_sessions.csv --work-dir /path/to/ci_work
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

from snbb_scheduler.checks import is_complete
from snbb_scheduler.config import SchedulerConfig
from snbb_scheduler.manifest import load_state
from snbb_scheduler.rules import _completion_kwargs
from snbb_scheduler.sessions import (
    discover_sessions,
    sanitize_session_id,
    sanitize_subject_code,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

WORK_MARKER = ".snbb_ci_workdir"

# Every SNBB_* variable a run script *writes* to, mapped to its location
# relative to the test config: ("bids", "") → bids_root,
# ("derivatives", "<dir>") → derivatives_root/<dir>, ("work", "<dir>") → work/<dir>.
_OUTPUT_LAYOUT: dict[str, tuple[str, str]] = {
    "SNBB_BIDS_ROOT": ("bids", ""),
    "SNBB_DERIVATIVES": ("derivatives", "qsiprep"),           # qsiprep output
    "SNBB_QSIPREP_DIR": ("derivatives", "qsiprep"),           # qsirecon input
    "SNBB_FS_OUTPUT": ("derivatives", "freesurfer"),
    "SNBB_FS_SUBJECTS_DIR": ("derivatives", "freesurfer"),    # fsatlas/qsiparc/qsirecon input
    "SNBB_QSIRECON_OUTPUT_DIR": ("derivatives", "qsirecon"),
    "SNBB_QSIRECON_DIR": ("derivatives", "qsirecon"),         # qsiparc input
    "SNBB_CAT_DERIVATIVES": ("derivatives", "cat12"),
    "SNBB_TABULAR_OUTPUT_DIR": ("derivatives", "tabular"),
    "SNBB_WORK_DIR": ("work", "nipype_work"),
    "SNBB_DEBUG_LOG": ("work", "logs/debug_submit.log"),
}
OUTPUT_VARS = tuple(_OUTPUT_LAYOUT)

# Production-tree defaults that are inputs only, so sharing them is safe:
# CAT12 atlases (prepared once by prepare_cat_atlases.py) and the templateflow
# cache (populated on first use, never per-subject).
READ_ONLY_PROD_VARS = ("SNBB_CAT_ATLAS_DIR", "SNBB_TEMPLATEFLOW_HOME")

# Code inputs: must come from the commit under test, not a copy elsewhere.
_CODE_LAYOUT = {
    "SNBB_HEURISTIC": "scripts/heuristic.py",
    "SNBB_RECON_SPEC": "scripts/qsirecon_full_spec.yaml",
    "SNBB_BIDS_FILTER_FILE": "examples/bids_filters.json",
    "SNBB_SCRIPT_DIR": "scripts",
}


class IsolationError(RuntimeError):
    """Raised when the test could write to, or delete, something it shouldn't."""


# ---------------------------------------------------------------------------
# Config and environment
# ---------------------------------------------------------------------------

def make_test_config(
    base: SchedulerConfig, checkout: Path, work: Path, sessions_csv: Path
) -> SchedulerConfig:
    """Return a copy of *base* whose writable paths live under *work* and whose
    scripts come from *checkout*. Inputs (DICOM root, Slurm settings, procedure
    graph) are kept from *base*."""
    scripts = Path(checkout) / "scripts"
    work = Path(work)
    procedures = [
        dataclasses.replace(p, script=str(scripts / Path(p.script).name),
                            depends_on=list(p.depends_on))
        for p in base.procedures
    ]
    return dataclasses.replace(
        base,
        bids_root=work / "bids",
        derivatives_root=work / "derivatives",
        state_file=work / "state.parquet",
        slurm_log_dir=work / "logs" / "slurm",
        log_file=work / "logs" / "scheduler.jsonl",
        completion_cache_file=work / "completion_cache.sqlite",
        sessions_file=Path(sessions_csv),
        qsirecon_spec=(scripts / base.qsirecon_spec.name) if base.qsirecon_spec else None,
        procedures=procedures,
        audit=dataclasses.replace(base.audit, report_dir=work / "reports",
                                  email_recipients=[]),
    )


def make_job_env(cfg: SchedulerConfig, checkout: Path, work: Path) -> dict[str, str]:
    """Environment overrides for the scheduler and the Slurm jobs it submits."""
    roots = {"bids": cfg.bids_root, "derivatives": cfg.derivatives_root, "work": Path(work)}
    env = {}
    for var, (root, rel) in _OUTPUT_LAYOUT.items():
        env[var] = str(roots[root] / rel) if rel else str(roots[root])
    for var, rel in _CODE_LAYOUT.items():
        env[var] = str(Path(checkout) / rel)
    return env


def _under(path: Path, root: Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def assert_isolated(
    cfg: SchedulerConfig, env: dict[str, str], base: SchedulerConfig, work: Path
) -> None:
    """Raise IsolationError unless every write target is inside *work* and
    *work* does not overlap the production output tree."""
    work = Path(work)
    prod_roots = [base.bids_root, base.derivatives_root, Path(base.state_file).parent]
    for root in prod_roots:
        if _under(work, root) or _under(root, work):
            raise IsolationError(f"work dir {work} overlaps production path {root}")
    targets = {
        "bids_root": cfg.bids_root,
        "derivatives_root": cfg.derivatives_root,
        "state_file": cfg.state_file,
        "slurm_log_dir": cfg.slurm_log_dir,
        "log_file": cfg.log_file,
        "completion_cache_file": cfg.completion_cache_file,
        **{var: Path(env[var]) for var in OUTPUT_VARS},
    }
    outside = {k: str(v) for k, v in targets.items() if v is None or not _under(v, work)}
    if outside:
        raise IsolationError(f"would write outside {work}: {outside}")


def write_config_yaml(cfg: SchedulerConfig, path: Path) -> None:
    """Serialise *cfg* so ``snbb-scheduler --config`` loads it back unchanged."""
    def plain(obj):
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, dict):
            return {k: plain(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [plain(v) for v in obj]
        return obj

    data = plain(dataclasses.asdict(cfg))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump(data, sort_keys=False))


def reset_work_dir(work: Path) -> None:
    """Empty *work* for a from-scratch run. Only deletes a dir this driver
    created (it holds WORK_MARKER) or an empty one."""
    work = Path(work)
    if work.exists():
        if (work / WORK_MARKER).exists():
            shutil.rmtree(work)
        elif any(work.iterdir()):
            raise IsolationError(
                f"{work} is not empty and was not created by this driver "
                f"(no {WORK_MARKER}); refusing to delete it"
            )
    work.mkdir(parents=True, exist_ok=True)
    (work / WORK_MARKER).write_text("Scratch dir of ci/pipeline_test.py; wiped on every run.\n")


# ---------------------------------------------------------------------------
# Privacy
# ---------------------------------------------------------------------------

def mask_commands(cfg: SchedulerConfig) -> list[str]:
    """GitHub ``::add-mask::`` lines for every form of every subject/session ID."""
    raw = pd.read_csv(cfg.sessions_file, dtype=str)
    values: set[str] = set()
    for col in (cfg.subject_col, cfg.session_col):
        if col in raw.columns:
            values.update(raw[col].dropna())
    if cfg.subject_col in raw.columns:
        for v in raw[cfg.subject_col].dropna():
            values.update({sanitize_subject_code(v), f"sub-{sanitize_subject_code(v)}"})
    if cfg.session_col in raw.columns:
        for v in raw[cfg.session_col].dropna():
            values.update({sanitize_session_id(v), f"ses-{sanitize_session_id(v)}"})
    for _, row in discover_sessions(cfg).iterrows():
        values.update({row["subject"], row["session"],
                       row["subject"].removeprefix("sub-"), row["session"].removeprefix("ses-")})
        if row["dicom_path"] is not None:
            values.add(str(row["dicom_path"]))
    return [f"::add-mask::{v}" for v in sorted(v for v in values if v)]


# ---------------------------------------------------------------------------
# Loop and report
# ---------------------------------------------------------------------------

def next_step(state: pd.DataFrame, submitted: int) -> str:
    """'failed' if any job failed, 'wait' while anything is queued/running or
    was just submitted, else 'done'."""
    if state.empty:
        return "wait" if submitted else "done"
    statuses = set(state["status"])
    if "failed" in statuses:
        return "failed"
    if submitted or statuses & {"pending", "running"}:
        return "wait"
    return "done"


def completion_report(cfg: SchedulerConfig) -> pd.DataFrame:
    """One row per (session, procedure) — per subject for subject-scoped ones —
    with anonymous labels, saying whether the output is complete on disk."""
    sessions = discover_sessions(cfg)
    subj_label = {s: f"subject{i + 1}" for i, s in enumerate(sessions["subject"].unique())}
    sess_label = {(r.subject, r.session): f"{subj_label[r.subject]}/session{i + 1}"
                  for i, r in enumerate(sessions.itertuples())}
    rows = []
    for proc in cfg.procedures:
        targets = (sessions.drop_duplicates("subject") if proc.scope == "subject"
                   else sessions)
        for _, row in targets.iterrows():
            label = (subj_label[row["subject"]] if proc.scope == "subject"
                     else sess_label[(row["subject"], row["session"])])
            done = bool(row["dicom_exists"]) and is_complete(
                proc, row[f"{proc.name}_path"], **_completion_kwargs(proc, row, cfg))
            rows.append({"label": label, "procedure": proc.name,
                         "dicom": bool(row["dicom_exists"]), "complete": done})
    return pd.DataFrame(rows, columns=["label", "procedure", "dicom", "complete"])


def cancel_in_flight(work: Path) -> list[str]:
    """scancel the pending/running jobs recorded in *work*'s state file, so an
    interrupted run (cancelled, timed out) leaves nothing writing into a work
    dir the next run is about to wipe. Returns the cancelled job IDs."""
    work = Path(work)
    state_file = work / "state.parquet"
    if not state_file.exists():
        return []
    if not (work / WORK_MARKER).exists():
        raise IsolationError(f"{work} was not created by this driver; not touching its jobs")
    state = load_state(SchedulerConfig(state_file=state_file))
    ids = [str(j) for j in state.loc[state["status"].isin(["pending", "running"]), "job_id"]
           if pd.notna(j) and str(j)]
    if ids:
        subprocess.run(["scancel", *ids], check=False)
    return ids


def _run_scheduler(cfg_path: Path, env: dict[str, str], cwd: Path) -> None:
    cmd = [sys.executable, "-c", "from snbb_scheduler.cli import main; main()",
           "--config", str(cfg_path), "run"]
    subprocess.run(cmd, env=env, cwd=cwd, check=True)


def _table(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)   # needs the optional 'tabulate' package
    except ImportError:
        return "```\n" + df.to_string(index=False) + "\n```"


def _write_summary(report: pd.DataFrame, outcome: str, elapsed_h: float, work: Path) -> None:
    report.to_csv(work / "summary_pipeline_ci.csv", index=False)
    lines = [f"## CI pipeline: {outcome}", "",
             f"Elapsed: {elapsed_h:.1f} h. "
             f"{int(report['complete'].sum())}/{len(report)} outputs complete.", "",
             _table(report)]
    text = "\n".join(lines) + "\n"
    print(text)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write(text)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base-config", type=Path,
                    help="Production config.yaml (procedures, DICOM root, Slurm settings).")
    ap.add_argument("--sessions", type=Path,
                    help="CSV with the test sessions, same columns as the production sessions file.")
    ap.add_argument("--work-dir", type=Path, required=True,
                    help="Scratch dir; wiped at the start of every run.")
    ap.add_argument("--checkout", type=Path, default=REPO_ROOT)
    ap.add_argument("--poll-minutes", type=float, default=10)
    ap.add_argument("--max-hours", type=float, default=47)
    ap.add_argument("--cancel-jobs", action="store_true",
                    help="Only scancel the in-flight jobs of the run in --work-dir, then exit.")
    args = ap.parse_args(argv)

    if args.cancel_jobs:
        ids = cancel_in_flight(args.work_dir.resolve())
        print(f"Cancelled {len(ids)} leftover job(s).")
        return 0
    if args.base_config is None or args.sessions is None:
        ap.error("--base-config and --sessions are required (unless --cancel-jobs)")

    base = SchedulerConfig.from_yaml(args.base_config)
    work = args.work_dir.resolve()
    cfg = make_test_config(base, args.checkout.resolve(), work, args.sessions.resolve())

    for line in mask_commands(cfg):      # before anything prints an ID
        print(line, flush=True)

    env_over = make_job_env(cfg, args.checkout.resolve(), work)
    assert_isolated(cfg, env_over, base, work)
    reset_work_dir(work)
    cfg_path = work / "config.yaml"
    write_config_yaml(cfg, cfg_path)
    env = {**os.environ, **env_over}
    for var in sorted(env_over):
        print(f"{var}={env_over[var]}")

    start = time.monotonic()
    outcome = "timeout"
    while (time.monotonic() - start) / 3600 < args.max_hours:
        before = len(load_state(cfg))
        _run_scheduler(cfg_path, env, cwd=args.checkout.resolve() / "scripts")
        state = load_state(cfg)
        step = next_step(state, submitted=len(state) - before)
        if step != "wait":
            outcome = step
            break
        time.sleep(args.poll_minutes * 60)

    report = completion_report(cfg)
    if outcome == "done" and not report["complete"].all():
        outcome = "incomplete"
    _write_summary(report, outcome, (time.monotonic() - start) / 3600, work)
    return 0 if outcome == "done" else 1


if __name__ == "__main__":
    sys.exit(main())
