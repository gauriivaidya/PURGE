"""
PurGE fitness function: trains the actual model (not a proxy) on a 5%
subsample of the training data, per trial -- the "ML" case from the paper
(NN tuning would instead swap in a smaller proxy architecture + few epochs;
see purge_nn_hpo.py for that variant).

Grammar must produce a phenotype that's a literal dict, e.g.
    "{'n_estimators': 120, 'max_depth': 7, 'min_samples_split': 4,
      'min_samples_leaf': 2, 'criterion': 'entropy'}"
See grammars/purge/rf_banknote.bnf.
"""

import ast
import csv
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from fitness.base_ff_classes.base_ff import base_ff
from algorithm.parameters import params

DATA_SUBSAMPLE_FRAC = 0.05  # the "5% of the data" budget

# Fixed column order for the trial log. Hyperparameter names are specific to
# the RF/Banknote grammar (grammars/purge/rf_banknote.bnf) -- if you swap in
# a different grammar, update HP_FIELDS to match its keys. Kept fixed (rather
# than derived from each row's dict) so every row -- including a failed one
# where the phenotype never parsed -- writes the same columns; a partial/
# ragged row would otherwise silently misalign the CSV.
HP_FIELDS = ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", "criterion"]
LOG_FIELDS = ["run_id", "dataset", "stage", "trial_index", "timestamp",
              "status", *HP_FIELDS, "accuracy", "error", "phenotype"]


def _load_ponyge_dataset(rel_path):
    """PonyGE2's bundled datasets are whitespace-delimited with a '#'
    comment header and the label in the last column (Banknote: +-1).
    Paths in DATASET_TRAIN/TEST are relative to the repo's datasets/ dir."""
    full_path = Path(__file__).resolve().parents[2] / "datasets" / rel_path
    df = pd.read_csv(full_path, comment="#", sep=r"\s+", header=None)
    X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values
    return X, y


def _log_trial(csv_path: Path, row: dict):
    """Append one row, every trial attempted -- success or failure alike --
    so results/ never has a gap between how many trials ran and how many
    got logged."""
    is_new = not csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in LOG_FIELDS})


class purge_hpo(base_ff):

    maximise = True  # higher accuracy = better

    def __init__(self):
        super().__init__()
        self.csv_path = Path(os.environ.get("PURGE_TRIAL_LOG", "experiments/purge/results/trials.csv"))
        self.run_id = os.environ.get("PURGE_RUN_ID", "unknown_run")
        self.dataset = os.environ.get("PURGE_DATASET", Path(params["DATASET_TRAIN"]).parent.name)
        self.stage = os.environ.get("PURGE_STAGE", "unknown_stage")
        self.trial_index = 0
        self.X_train, self.y_train = _load_ponyge_dataset(params["DATASET_TRAIN"])
        self.X_test, self.y_test = _load_ponyge_dataset(params["DATASET_TEST"])

    def evaluate(self, ind, **kwargs):
        self.trial_index += 1
        base_row = {
            "run_id": self.run_id, "dataset": self.dataset, "stage": self.stage,
            "trial_index": self.trial_index, "timestamp": time.time(),
            "phenotype": ind.phenotype,
        }

        try:
            hp = ast.literal_eval(ind.phenotype)

            # 5% subsample each trial -- cheap signal, per the PurGE data
            # subsampling strategy (fresh split per trial, not a fixed subset)
            X_sub, _, y_sub, _ = train_test_split(
                self.X_train, self.y_train,
                train_size=DATA_SUBSAMPLE_FRAC, stratify=self.y_train,
            )
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_sub, y_sub, test_size=0.2, stratify=y_sub,
            )

            model = RandomForestClassifier(**hp, n_jobs=-1)
            model.fit(X_tr, y_tr)
            accuracy = model.score(X_val, y_val)

            _log_trial(self.csv_path, {**base_row, "status": "ok", **hp, "accuracy": accuracy})
            return accuracy

        except Exception as err:
            # Log the failed trial too (bad phenotype, degenerate split,
            # invalid sklearn params, etc.) instead of letting it vanish --
            # then hand PonyGE2 the worst possible fitness for it.
            _log_trial(self.csv_path, {**base_row, "status": "error", "error": str(err), "accuracy": 0.0})
            return 0.0
