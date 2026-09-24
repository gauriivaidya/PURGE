"""
Automated two-stage PurGE pipeline -- trial example.

Tunes a RandomForestClassifier on the bundled Banknote dataset, training on
a fresh 5% subsample each trial (see src/fitness/purge_hpo.py). Stage 1
explores the full grammar; its trial log is fed straight into purge_prune.py
to produce a pruned grammar; stage 2 runs on that pruned grammar. No manual
BNF editing in between.

Run from anywhere:
    python experiments/purge/purge_pipeline.py
"""

import csv
import json
import os
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from purge_prune import purge

REPO_ROOT        = Path(__file__).resolve().parents[2]
PONYGE_SRC       = REPO_ROOT / "src"
RESULTS_ROOT     = Path(__file__).parent / "results"
RUNS_INDEX       = RESULTS_ROOT / "runs_index.csv"   # one row per run, across all datasets

BASE_PARAMS      = str(REPO_ROOT / "parameters" / "purge_base.txt")
DATASET_NAME     = "Banknote"                        # matches datasets/Banknote/ -- change per dataset
BASE_GRAMMAR     = "purge/rf_banknote.bnf"            # relative to grammars/, as PonyGE2 expects
GENERATED_GRAMMAR_DIR = REPO_ROOT / "grammars" / "purge" / "generated"

TOTAL_TRIALS     = 80
STAGE1_TRIALS    = 50                                 # your 50/30 split, automated
STAGE2_TRIALS    = TOTAL_TRIALS - STAGE1_TRIALS
POPULATION_SIZE  = 10                                 # trials = population * generations

MIN_ACCURACY     = 0.5                                # discard trials below this VA before pruning (paper: 50%)


def run_ponyge(grammar_file: str, generations: int, stage: str, run_id: str, run_dir: Path) -> Path:
    trial_log = run_dir / f"{stage}_trials.csv"

    cmd = [
        sys.executable, "ponyge.py",
        "--parameters", BASE_PARAMS,
        "--grammar_file", grammar_file,
        "--population_size", str(POPULATION_SIZE),
        "--generations", str(generations),
        "--experiment_name", f"{run_id}_{stage}",
    ]
    env = {
        **os.environ,
        "PURGE_TRIAL_LOG": str(trial_log),   # read by purge_hpo.py -- every trial logged here
        "PURGE_RUN_ID": run_id,
        "PURGE_DATASET": DATASET_NAME,
        "PURGE_STAGE": stage,
    }
    subprocess.run(cmd, cwd=PONYGE_SRC, check=True, env=env)
    return trial_log


def _best_row(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    ok = df[df["status"] == "ok"] if "status" in df.columns else df
    if ok.empty:
        return {}
    return ok.loc[ok["accuracy"].idxmax()].to_dict()


def _append_runs_index(row: dict):
    is_new = not RUNS_INDEX.exists()
    RUNS_INDEX.parent.mkdir(parents=True, exist_ok=True)
    with open(RUNS_INDEX, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def main():
    run_id = f"{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
    run_dir = RESULTS_ROOT / DATASET_NAME / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    GENERATED_GRAMMAR_DIR.mkdir(parents=True, exist_ok=True)

    stage1_gens = round(STAGE1_TRIALS / POPULATION_SIZE)
    stage2_gens = round(STAGE2_TRIALS / POPULATION_SIZE)

    print(f"--- Run {run_id} on {DATASET_NAME} ---")
    print(f"--- Stage 1: {STAGE1_TRIALS} trials, full search space ---")
    stage1_csv = run_ponyge(BASE_GRAMMAR, stage1_gens, "stage1", run_id, run_dir)

    print("--- Auto-pruning search space from stage 1 results ---")
    # grammar filename identifies dataset + run, so concurrent/sequential runs
    # never overwrite each other's pruned grammar
    pruned_grammar_relpath = f"purge/generated/{DATASET_NAME}_{run_id}_pruned.bnf"
    pruned_grammar_path = REPO_ROOT / "grammars" / pruned_grammar_relpath
    ranges = purge(
        csv_file=stage1_csv,
        accuracy_column="accuracy",
        min_accuracy=MIN_ACCURACY,
        output_bnf_file=pruned_grammar_path,
    )
    shutil.copy(pruned_grammar_path, run_dir / "pruned_grammar.bnf")  # archived with this run's results

    print(f"--- Stage 2: {STAGE2_TRIALS} trials, pruned search space ---")
    stage2_csv = run_ponyge(pruned_grammar_relpath, stage2_gens, "stage2", run_id, run_dir)

    best1, best2 = _best_row(stage1_csv), _best_row(stage2_csv)

    manifest = {
        "run_id": run_id,
        "dataset": DATASET_NAME,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "stage1_trials": STAGE1_TRIALS,
        "stage2_trials": STAGE2_TRIALS,
        "min_accuracy": MIN_ACCURACY,
        "population_size": POPULATION_SIZE,
        "pruned_search_space": ranges,
        "best_stage1": best1,
        "best_stage2": best2,
        "stage1_log": str(stage1_csv),
        "stage2_log": str(stage2_csv),
        "pruned_grammar": str(pruned_grammar_path),
    }
    with open(run_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    _append_runs_index({
        "run_id": run_id,
        "dataset": DATASET_NAME,
        "timestamp_utc": manifest["timestamp_utc"],
        "stage1_trials": STAGE1_TRIALS,
        "stage2_trials": STAGE2_TRIALS,
        "best_stage1_accuracy": best1.get("accuracy", ""),
        "best_stage2_accuracy": best2.get("accuracy", ""),
        "run_dir": str(run_dir),
    })

    print(f"Done. Results: {run_dir}")
    print(f"      Master index: {RUNS_INDEX}")


if __name__ == "__main__":
    main()
