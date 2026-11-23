"""Tests for ci/pipeline_test.py — the layer-2 pipeline CI driver.

The driver itself runs on the lab server against real data; these tests cover
its pure logic on synthetic paths: that every output is redirected into the
scratch work dir (never production), that the checked-out code is what runs,
that cleanup refuses to delete anything it did not create, that subject IDs
are masked in public logs, and that the polling loop stops correctly.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

from snbb_scheduler.config import Procedure, SchedulerConfig

_DRIVER = Path(__file__).resolve().parent.parent / "ci" / "pipeline_test.py"


@pytest.fixture(scope="module")
def drv():
    spec = importlib.util.spec_from_file_location("pipeline_test", _DRIVER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pipeline_test"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def prod(tmp_path):
    """A production-like base config: absolute script paths into someone's home."""
    home = tmp_path / "home" / "someone" / "snbb_scheduler" / "scripts"
    return SchedulerConfig(
        dicom_root=tmp_path / "raw",
        bids_root=tmp_path / "prod" / "bids",
        derivatives_root=tmp_path / "prod" / "derivatives",
        state_file=tmp_path / "prod" / ".scheduler_state.parquet",
        slurm_log_dir=tmp_path / "prod" / "logs",
        log_file=tmp_path / "prod" / "scheduler.log",
        completion_cache_file=tmp_path / "prod" / "cache.sqlite",
        qsirecon_spec=home / "qsirecon_full_spec.yaml",
        procedures=[
            Procedure(name="bids", output_dir="", script=str(home / "snbb_run_bids.sh")),
            Procedure(name="qsiprep", output_dir="qsiprep", script="snbb_run_qsiprep.sh",
                      depends_on=["bids"]),
            Procedure(name="freesurfer", output_dir="freesurfer",
                      script=str(home / "snbb_run_freesurfer.sh"), scope="subject",
                      depends_on=["bids"]),
        ],
    )


@pytest.fixture
def paths(tmp_path):
    checkout = tmp_path / "checkout"
    (checkout / "scripts").mkdir(parents=True)
    work = tmp_path / "ci_work"
    sessions = tmp_path / "ci_sessions.csv"
    sessions.write_text("UID,ScanID\nX123,20220315_0930\nX123,20250610_1415\n")
    return checkout, work, sessions


def _is_under(p, root) -> bool:
    try:
        Path(p).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Test config: everything written goes to the work dir, code comes from checkout
# ---------------------------------------------------------------------------

def test_config_redirects_all_writable_paths_into_work(drv, prod, paths):
    checkout, work, sessions = paths
    cfg = drv.make_test_config(prod, checkout, work, sessions)
    for attr in ("bids_root", "derivatives_root", "state_file", "slurm_log_dir",
                 "log_file", "completion_cache_file"):
        assert _is_under(getattr(cfg, attr), work), attr


def test_config_keeps_inputs_from_production(drv, prod, paths):
    checkout, work, sessions = paths
    cfg = drv.make_test_config(prod, checkout, work, sessions)
    assert cfg.dicom_root == prod.dicom_root
    assert cfg.sessions_file == sessions
    assert cfg.slurm_partition == prod.slurm_partition
    assert cfg.slurm_account == prod.slurm_account


def test_config_runs_scripts_from_checkout(drv, prod, paths):
    checkout, work, sessions = paths
    cfg = drv.make_test_config(prod, checkout, work, sessions)
    for proc in cfg.procedures:
        assert Path(proc.script) == checkout / "scripts" / Path(proc.script).name
    assert cfg.qsirecon_spec == checkout / "scripts" / "qsirecon_full_spec.yaml"


def test_config_keeps_procedure_graph(drv, prod, paths):
    checkout, work, sessions = paths
    cfg = drv.make_test_config(prod, checkout, work, sessions)
    assert [(p.name, p.output_dir, p.scope, p.depends_on) for p in cfg.procedures] == [
        (p.name, p.output_dir, p.scope, p.depends_on) for p in prod.procedures
    ]


def test_config_does_not_mutate_base(drv, prod, paths):
    checkout, work, sessions = paths
    before = prod.procedures[0].script
    drv.make_test_config(prod, checkout, work, sessions)
    assert prod.procedures[0].script == before
    assert prod.bids_root.name == "bids" and "prod" in prod.bids_root.parts


def test_config_yaml_roundtrip(drv, prod, paths):
    checkout, work, sessions = paths
    cfg = drv.make_test_config(prod, checkout, work, sessions)
    out = work / "config.yaml"
    drv.write_config_yaml(cfg, out)
    back = SchedulerConfig.from_yaml(out)
    assert back == cfg


# ---------------------------------------------------------------------------
# Job environment: the run scripts default to production paths, so every
# output variable must be overridden
# ---------------------------------------------------------------------------

def test_env_overrides_every_output_variable_into_work(drv, prod, paths):
    checkout, work, sessions = paths
    cfg = drv.make_test_config(prod, checkout, work, sessions)
    env = drv.make_job_env(cfg, checkout, work)
    for var in drv.OUTPUT_VARS:
        assert var in env, var
        assert _is_under(env[var], work), var


def test_env_output_dirs_match_scheduler_completion_paths(drv, prod, paths):
    """Scripts must write where the scheduler's completion checks look."""
    checkout, work, sessions = paths
    cfg = drv.make_test_config(prod, checkout, work, sessions)
    env = drv.make_job_env(cfg, checkout, work)
    assert Path(env["SNBB_BIDS_ROOT"]) == cfg.bids_root
    assert Path(env["SNBB_DERIVATIVES"]) == cfg.derivatives_root / "qsiprep"
    assert Path(env["SNBB_QSIPREP_DIR"]) == cfg.derivatives_root / "qsiprep"
    assert Path(env["SNBB_FS_OUTPUT"]) == cfg.derivatives_root / "freesurfer"
    assert Path(env["SNBB_FS_SUBJECTS_DIR"]) == cfg.derivatives_root / "freesurfer"
    assert Path(env["SNBB_QSIRECON_OUTPUT_DIR"]) == cfg.derivatives_root / "qsirecon"
    assert Path(env["SNBB_QSIRECON_DIR"]) == cfg.derivatives_root / "qsirecon"
    assert Path(env["SNBB_CAT_DERIVATIVES"]) == cfg.derivatives_root / "cat12"


def test_env_points_code_inputs_at_checkout(drv, prod, paths):
    checkout, work, sessions = paths
    cfg = drv.make_test_config(prod, checkout, work, sessions)
    env = drv.make_job_env(cfg, checkout, work)
    assert Path(env["SNBB_HEURISTIC"]) == checkout / "scripts" / "heuristic.py"
    assert Path(env["SNBB_RECON_SPEC"]) == checkout / "scripts" / "qsirecon_full_spec.yaml"
    assert Path(env["SNBB_BIDS_FILTER_FILE"]) == checkout / "examples" / "bids_filters.json"
    assert Path(env["SNBB_SCRIPT_DIR"]) == checkout / "scripts"


def test_env_leaves_read_only_inputs_alone(drv, prod, paths):
    """Containers, licenses, atlases: not set, so the scripts' defaults (or the
    runner's own environment) apply."""
    checkout, work, sessions = paths
    cfg = drv.make_test_config(prod, checkout, work, sessions)
    env = drv.make_job_env(cfg, checkout, work)
    for var in ("SNBB_DICOM_ROOT", "SNBB_FS_LICENSE", "SNBB_QSIPREP_SIF", "SNBB_ATLASES_DIR"):
        assert var not in env


def test_every_script_output_default_is_covered(drv):
    """Guard against a new script writing to a production default we don't override.

    Any `SNBB_*` variable in scripts/snbb_run_*.sh whose default points into the
    production snbb_scheduler tree must be overridden by the driver.
    """
    import re

    scripts = Path(__file__).resolve().parent.parent / "scripts"
    pat = re.compile(r'^\s*(SNBB_[A-Z0-9_]+)="\$\{\1:-([^}"]*)')
    found = set()
    for sh in scripts.glob("snbb_run_*.sh"):
        for line in sh.read_text().splitlines():
            m = pat.match(line)
            if m and "/snbb_scheduler/" in m.group(2) and "/scripts/" not in m.group(2) \
                    and "/examples/" not in m.group(2):
                found.add(m.group(1))
    covered = set(drv.OUTPUT_VARS) | set(drv.READ_ONLY_PROD_VARS)
    assert found, "pattern found no variables; the regex is out of date"
    assert found <= covered, f"not redirected: {sorted(found - covered)}"


# ---------------------------------------------------------------------------
# Isolation guard
# ---------------------------------------------------------------------------

def test_guard_accepts_clean_setup(drv, prod, paths):
    checkout, work, sessions = paths
    cfg = drv.make_test_config(prod, checkout, work, sessions)
    drv.assert_isolated(cfg, drv.make_job_env(cfg, checkout, work), prod, work)


def test_guard_rejects_output_outside_work(drv, prod, paths):
    checkout, work, sessions = paths
    cfg = drv.make_test_config(prod, checkout, work, sessions)
    env = drv.make_job_env(cfg, checkout, work)
    env["SNBB_FS_OUTPUT"] = str(prod.derivatives_root / "freesurfer")
    with pytest.raises(drv.IsolationError):
        drv.assert_isolated(cfg, env, prod, work)


@pytest.mark.parametrize("where", ["inside_bids", "contains_derivatives"])
def test_guard_rejects_work_overlapping_production(drv, prod, paths, where):
    checkout, _, sessions = paths
    work = prod.bids_root / "ci" if where == "inside_bids" else prod.derivatives_root.parent
    cfg = drv.make_test_config(prod, checkout, work, sessions)
    with pytest.raises(drv.IsolationError):
        drv.assert_isolated(cfg, drv.make_job_env(cfg, checkout, work), prod, work)


# ---------------------------------------------------------------------------
# Work-dir reset: only ever delete a dir this driver created
# ---------------------------------------------------------------------------

def test_reset_creates_marked_work_dir(drv, tmp_path):
    work = tmp_path / "w"
    drv.reset_work_dir(work)
    assert (work / drv.WORK_MARKER).exists()


def test_reset_wipes_previous_run(drv, tmp_path):
    work = tmp_path / "w"
    drv.reset_work_dir(work)
    (work / "bids" / "old.nii.gz").parent.mkdir()
    (work / "bids" / "old.nii.gz").touch()
    drv.reset_work_dir(work)
    assert not (work / "bids").exists()
    assert (work / drv.WORK_MARKER).exists()


def test_reset_refuses_unmarked_nonempty_dir(drv, tmp_path):
    work = tmp_path / "precious"
    work.mkdir()
    (work / "data.csv").write_text("keep me")
    with pytest.raises(drv.IsolationError):
        drv.reset_work_dir(work)
    assert (work / "data.csv").read_text() == "keep me"


def test_reset_accepts_existing_empty_dir(drv, tmp_path):
    work = tmp_path / "empty"
    work.mkdir()
    drv.reset_work_dir(work)
    assert (work / drv.WORK_MARKER).exists()


# ---------------------------------------------------------------------------
# Masking: subject/session IDs must never appear in (possibly public) logs
# ---------------------------------------------------------------------------

def test_mask_commands_cover_all_id_forms(drv, prod, paths):
    checkout, work, sessions = paths
    cfg = drv.make_test_config(prod, checkout, work, sessions)
    lines = drv.mask_commands(cfg)
    masked = {ln.removeprefix("::add-mask::") for ln in lines}
    assert all(ln.startswith("::add-mask::") for ln in lines)
    for value in ("X123", "sub-X123", "20220315_0930", "202203150930",
                  "ses-202203150930", "20250610_1415", "ses-202506101415"):
        assert value in masked, value


# ---------------------------------------------------------------------------
# Polling loop decision
# ---------------------------------------------------------------------------

def _state(*statuses):
    return pd.DataFrame({
        "subject": ["sub-X"] * len(statuses),
        "session": [f"ses-{i}" for i in range(len(statuses))],
        "procedure": ["bids"] * len(statuses),
        "status": list(statuses),
        "submitted_at": pd.Timestamp("2026-01-01"),
        "job_id": [str(i) for i in range(len(statuses))],
    })


def test_next_step_waits_while_jobs_in_flight(drv):
    assert drv.next_step(_state("complete", "running"), submitted=0) == "wait"
    assert drv.next_step(_state("pending"), submitted=0) == "wait"


def test_next_step_continues_after_new_submissions(drv):
    assert drv.next_step(_state("complete", "pending"), submitted=1) == "wait"


def test_next_step_done_when_quiet(drv):
    assert drv.next_step(_state("complete", "complete"), submitted=0) == "done"
    assert drv.next_step(_state(), submitted=0) == "done"


def test_next_step_fails_fast_on_failed_job(drv):
    assert drv.next_step(_state("complete", "failed", "running"), submitted=0) == "failed"


# ---------------------------------------------------------------------------
# main(): the loop end to end, with the scheduler call faked
# ---------------------------------------------------------------------------

@pytest.fixture
def e2e(drv, prod, paths, tmp_path, monkeypatch):
    """Base config on disk with one session-scoped procedure, plus a fake scheduler."""
    checkout, work, sessions = paths
    sessions.write_text("UID,ScanID\nX123,20220315_0930\n")
    (prod.dicom_root / "20220315_0930").mkdir(parents=True)
    base = dataclasses.replace(prod, procedures=[  # one procedure, no deps
        Procedure(name="qsiprep", output_dir="qsiprep", script="snbb_run_qsiprep.sh")])
    base_yaml = tmp_path / "prod_config.yaml"
    drv.write_config_yaml(base, base_yaml)
    monkeypatch.setattr(drv.time, "sleep", lambda s: None)
    calls = []

    def run(outcome):
        def fake(cfg_path, env, cwd):
            cfg = SchedulerConfig.from_yaml(cfg_path)
            calls.append(env)
            state = drv.load_state(cfg)
            if state.empty:   # first call: "submit" the job
                state = _state("pending").assign(procedure="qsiprep")
            else:             # second call: job finished
                state = state.assign(status=outcome)
                if outcome == "complete":
                    out = cfg.derivatives_root / "qsiprep" / "sub-X123" / "ses-202203150930"
                    out.mkdir(parents=True)
                    (out / "x.nii.gz").touch()
            from snbb_scheduler.manifest import save_state
            save_state(state, cfg)
        monkeypatch.setattr(drv, "_run_scheduler", fake)
        return drv.main(["--base-config", str(base_yaml), "--sessions", str(sessions),
                         "--work-dir", str(work), "--checkout", str(checkout),
                         "--poll-minutes", "0"])
    return run, calls, work


def test_main_passes_when_everything_completes(e2e, capsys):
    run, calls, work = e2e
    assert run("complete") == 0
    assert len(calls) == 2   # submit; then see it complete with nothing new to submit
    assert all(Path(c["SNBB_DERIVATIVES"]).is_relative_to(work) for c in calls)
    out = capsys.readouterr().out
    assert "::add-mask::X123" in out
    assert out.index("::add-mask::") < out.index("SNBB_")
    assert (work / "summary_pipeline_ci.csv").exists()


def test_main_fails_when_a_job_fails(e2e):
    run, _, _ = e2e
    assert run("failed") == 1


def test_main_fails_when_quiet_but_outputs_missing(e2e):
    run, _, _ = e2e
    assert run("complete_without_output") == 1


def test_main_writes_github_step_summary(e2e, tmp_path, monkeypatch):
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    run, _, _ = e2e
    run("complete")
    text = summary.read_text()
    assert "CI pipeline: done" in text
    assert "X123" not in text and "2022" not in text


# ---------------------------------------------------------------------------
# Final report
# ---------------------------------------------------------------------------

def test_report_lists_every_session_procedure_pair(drv, prod, paths):
    checkout, work, sessions = paths
    cfg = drv.make_test_config(prod, checkout, work, sessions)
    (cfg.dicom_root / "20220315_0930").mkdir(parents=True)
    (cfg.dicom_root / "20250610_1415").mkdir(parents=True)
    rep = drv.completion_report(cfg)
    # 2 sessions x 2 session-scoped procedures + 1 subject x freesurfer
    assert len(rep) == 5
    assert set(rep.columns) >= {"label", "procedure", "complete"}
    assert not rep["complete"].any()
    # labels are anonymous (no IDs)
    assert not rep["label"].str.contains("X123|2022|2025").any()


def test_report_marks_completed_output(drv, prod, paths):
    checkout, work, sessions = paths
    cfg = drv.make_test_config(prod, checkout, work, sessions)
    (cfg.dicom_root / "20220315_0930").mkdir(parents=True)
    (cfg.dicom_root / "20250610_1415").mkdir(parents=True)
    qp = cfg.derivatives_root / "qsiprep" / "sub-X123" / "ses-202203150930"
    qp.mkdir(parents=True)
    (qp / "out.nii.gz").touch()
    rep = drv.completion_report(cfg)
    done = rep[rep["complete"]]
    assert list(done["procedure"]) == ["qsiprep"]


# ---------------------------------------------------------------------------
# Cleanup: cancel leftover Slurm jobs of an interrupted run
# ---------------------------------------------------------------------------

def test_cancel_in_flight_scancels_only_pending_and_running(drv, tmp_path, monkeypatch):
    from snbb_scheduler.manifest import save_state

    work = tmp_path / "w"
    drv.reset_work_dir(work)
    cfg = SchedulerConfig(state_file=work / "state.parquet")
    save_state(_state("complete", "running", "pending", "failed"), cfg)
    seen = []
    monkeypatch.setattr(drv.subprocess, "run", lambda cmd, **kw: seen.append(cmd))
    assert drv.cancel_in_flight(work) == ["1", "2"]
    assert seen == [["scancel", "1", "2"]]


def test_cancel_in_flight_no_state_is_noop(drv, tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(drv.subprocess, "run", lambda cmd, **kw: seen.append(cmd))
    assert drv.cancel_in_flight(tmp_path / "missing") == []
    assert seen == []


def test_cancel_in_flight_refuses_unmarked_dir(drv, tmp_path):
    from snbb_scheduler.manifest import save_state

    work = tmp_path / "w"
    work.mkdir()
    save_state(_state("running"), SchedulerConfig(state_file=work / "state.parquet"))
    with pytest.raises(drv.IsolationError):
        drv.cancel_in_flight(work)
