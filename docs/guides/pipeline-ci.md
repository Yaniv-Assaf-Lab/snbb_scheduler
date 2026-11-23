# Pipeline CI

Two GitHub Actions workflows test the scheduler:

| Workflow | Runs on | When | Data |
|---|---|---|---|
| `tests.yml` | GitHub's machines | every PR and every push to `main` | synthetic (`pytest`) |
| `ci-pipeline.yml` | the lab server (self-hosted runner) | after every merge to `main`, or by hand | a few real sessions |

This page covers the second one: what it does and the one-time server setup.

---

## What it does

After a merge, the runner on the lab server checks out the new `main` and runs
[`ci/pipeline_test.py`](https://github.com/Yaniv-Assaf-Lab/snbb_scheduler/blob/main/ci/pipeline_test.py), which:

1. Wipes a scratch **work dir** and writes a test config there: the production
   config, with every output path (BIDS, derivatives, state, logs, cache) moved
   into the work dir and every procedure script taken from the checked-out commit.
2. Sets the `SNBB_*` environment variables the run scripts write to (they
   default to production paths) to the matching paths in the work dir, and
   points the heuristic, recon spec and BIDS filter at the checked-out copies.
   It refuses to start if any write target is outside the work dir, or if the
   work dir overlaps the production tree.
3. Runs `snbb-scheduler run` every 10 minutes until every procedure for the
   test sessions has finished, a job fails, or 47 hours pass.
4. Checks every (session, procedure) output on disk and fails unless all are
   complete. A table with anonymous labels (`subject1/session2`) goes to the
   run's summary page and to `summary_pipeline_ci.csv` in the work dir.

The result shows as ✅/❌ on the merge commit in `main`. Pull `main` into the
production checkout only when it is ✅.

Read-only inputs (raw DICOM, containers, FreeSurfer license, atlases,
templateflow cache) are used from their usual locations.

!!! warning "Privacy"
    Workflow logs of a public repository are public. The driver masks every
    subject/session ID (GitHub prints `***` instead) before anything is
    logged. The session list lives only on the server; never commit it or
    put IDs in a workflow file, variable, PR or issue.

---

## One-time setup

Needs a repository **admin** and someone with an account on the server.

### 1. Service user

Create (or pick) a non-root user for the runner that can:

- read the raw DICOM root and the production `config.yaml`;
- write the work dir (create it empty, outside the production BIDS/derivatives tree);
- run `sbatch`/`sacct`/`scancel` with the scheduler's Slurm account;
- read the Apptainer images and the FreeSurfer license the scripts use.

If a script's default input path is not readable by this user (e.g. a license
in someone's home directory), set the corresponding `SNBB_*` variable in the
runner's environment (`actions-runner/.env`).

### 2. Session list

Put a CSV on the server with the test sessions, in the same format as the
production `sessions_file` (by default columns `UID` and `ScanID`). Pick
subjects with several sessions spanning the protocol eras you care about, so
FreeSurfer longitudinal and older naming conventions are exercised.

### 3. Runner

1. On GitHub: **Settings → Actions → Runners → New self-hosted runner → Linux**.
   The page shows the commands with a one-time registration token (valid ~1 hour).
2. On the server, as the service user:

    ```bash
    mkdir ~/actions-runner && cd ~/actions-runner
    # download + extract: copy the two commands from the GitHub page
    ./config.sh --url https://github.com/Yaniv-Assaf-Lab/snbb_scheduler \
                --token <TOKEN> --labels snbb-server --unattended
    sudo ./svc.sh install "$USER" && sudo ./svc.sh start
    ```

The runner only makes outbound HTTPS connections to GitHub; no ports are opened.

### 4. Repository variables

**Settings → Secrets and variables → Actions → Variables**:

| Variable | Value |
|---|---|
| `SNBB_CI_BASE_CONFIG` | path to the production `config.yaml` |
| `SNBB_CI_SESSIONS_CSV` | path to the CSV from step 2 |
| `SNBB_CI_WORK_DIR` | the empty work dir from step 1 |

### 5. Safety setting

**Settings → Actions → General → Fork pull request workflows**: require
approval for all outside collaborators. The pipeline workflow never runs on
pull requests, but this keeps any future workflow from running fork code on
the server unreviewed.

### 6. First run

**Actions → CI / pipeline → Run workflow** on `main`. The first step fails
with a clear message if a variable, file or `sbatch` is missing.

---

## Running it by hand

- **Before merging a risky PR:** Actions → CI / pipeline → Run workflow →
  pick the PR's branch (needs write access to the repository).
- **On the server, without GitHub:**

    ```bash
    uv run python ci/pipeline_test.py --base-config <config.yaml> \
        --sessions <sessions.csv> --work-dir <work dir>
    ```

- **Cancel leftover jobs** of an interrupted run (the workflow does this
  automatically when cancelled or failed):

    ```bash
    uv run python ci/pipeline_test.py --cancel-jobs --work-dir <work dir>
    ```

The work dir is wiped at the start of every run. The driver only deletes a
directory it created itself (it contains `.snbb_ci_workdir`) or an empty one.
