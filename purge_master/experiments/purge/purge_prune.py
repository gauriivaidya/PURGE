"""
PurGE search-space pruning -- Algorithm 1 from the paper (Section 3.1),
not the ICA-based approach from the original purge.py.

Pipeline (mirrors Algorithm 1 / Fig. 3-5):
  1. Discard trials with VA < min_accuracy.
  2. H1: hyperparameters whose Pearson correlation with VA exceeds
     corr_threshold -- these get pruned individually via ICE/partial
     dependence (Fig. 4).
  3. H2: hyperparameter PAIRS whose mutual Pearson correlation exceeds
     pair_corr_threshold -- these get pruned jointly via a 2D VA heatmap
     over their value combinations (Fig. 5), on top of whatever H1 already
     did for each member.
  4. A hyperparameter in neither H1 nor any H2 pair is left unpruned
     (paper's batch_size vs. lr/layers/dropout example: batch_size was
     pruned via H1 alone; lr/layers/dropout only got pruned because they
     showed up in an H2 pair, not because they were individually
     correlated with VA).
  5. Write the surviving values/ranges as a BNF grammar (same dict-literal
     phenotype format as before, same quote-wrapping fix for PonyGE2's
     grammar parser).

Categorical hyperparameters go through the exact same PDP-based selection
as numeric ones (step 2) -- this is what the paper's ICE plot for
`optimizer` is doing (Adam/RMSProp kept, SGD dropped based on partial
dependence, not just "appeared in the top N%").
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import partial_dependence
from sklearn.preprocessing import LabelEncoder

N_STEPS = 12  # cap on how many explicit values a pruned numeric range is discretised into

# Bookkeeping columns purge_hpo.py adds to every trial row -- never treated
# as hyperparameters to prune.
METADATA_FIELDS = ["run_id", "dataset", "stage", "trial_index", "timestamp",
                    "status", "error", "phenotype"]


def _encode(df: pd.DataFrame):
    """Label-encode categorical columns for correlation/surrogate-model use.
    Returns the encoded frame plus {col: LabelEncoder} for decoding later."""
    encoded = df.copy()
    encoders = {}
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            le = LabelEncoder()
            encoded[col] = le.fit_transform(df[col])
            encoders[col] = le
    return encoded, encoders


def _partial_dependence_keep(model, X_enc, feature_idx, col, encoders, is_int):
    """ICE/partial dependence for one hyperparameter: average predicted VA
    across the observed distribution of the other hyperparameters, at each
    of this hyperparameter's observed values. Keep values whose partial
    dependence is at or above the column's own mean PD -- this is the
    'Optimal Dependence' cutoff line in Fig. 4."""
    X_float = X_enc.astype(float)  # sklearn deprecates PDP on integer-dtype columns
    pd_result = partial_dependence(
        model, X_float, features=[feature_idx], kind="average",
        grid_resolution=min(20, X_enc.iloc[:, feature_idx].nunique()),
    )
    grid = pd_result["grid_values"][0]
    avg_pd = pd_result["average"][0]

    threshold = avg_pd.mean()
    kept_encoded = grid[avg_pd >= threshold]

    if col in encoders:
        # map encoded ints back to original category strings
        kept = encoders[col].inverse_transform(kept_encoded.astype(int)).tolist()
    elif is_int:
        kept = sorted(set(int(round(v)) for v in kept_encoded))
    else:
        kept = sorted(set(round(float(v), 4) for v in kept_encoded))
    return kept


def _pairwise_refine(df, col_a, col_b, accuracy_column, va_threshold):
    """Joint VA heatmap over (col_a, col_b) combinations (Fig. 5): keep the
    value of each column that appears in at least one combination whose
    mean VA is at/above va_threshold."""
    grouped = df.groupby([col_a, col_b])[accuracy_column].mean().reset_index()
    good = grouped[grouped[accuracy_column] >= va_threshold]
    return set(good[col_a].unique()), set(good[col_b].unique())


def _discretise_range(values, is_int):
    lo, hi = min(values), max(values)
    if lo == hi:
        return [lo]
    steps = np.linspace(lo, hi, N_STEPS)
    if is_int:
        return sorted(set(int(round(s)) for s in steps))
    return sorted(set(round(float(s), 4) for s in steps))


def purge(csv_file, accuracy_column="accuracy", min_accuracy=0.5,
          corr_threshold=0.1, pair_corr_threshold=0.3,
          output_bnf_file="hyperparameters.bnf"):
    """
    Returns: dict {param_name: [kept values]} -- the pruned search space.
    """
    data = pd.read_csv(csv_file)
    if accuracy_column not in data.columns:
        raise ValueError(f"The CSV file must contain a '{accuracy_column}' column.")

    param_cols = [c for c in data.columns if c != accuracy_column and c not in METADATA_FIELDS]

    # Drop failed trials (status == 'error') before anything else -- they
    # have no valid hyperparameter values to correlate or prune with, but
    # they're still in the CSV so nothing about the run gets lost.
    if "status" in data.columns:
        n_failed = (data["status"] == "error").sum()
        if n_failed:
            print(f"Excluding {n_failed} failed trial(s) from pruning (still present in the CSV).")
        data = data[data["status"] == "ok"].reset_index(drop=True)

    # Step 1: discard low-performing configurations (paper: VA < 50%)
    df = data[data[accuracy_column] >= min_accuracy].reset_index(drop=True)
    if len(df) < 5:
        raise ValueError(
            f"Only {len(df)} trials scored >= min_accuracy={min_accuracy}; "
            "not enough data to prune reliably. Lower min_accuracy or run more trials."
        )

    encoded, encoders = _encode(df[param_cols])
    is_int_col = {c: pd.api.types.is_integer_dtype(df[c]) for c in param_cols}

    # Step 2: H1 -- hyperparameters correlated with VA
    h1 = []
    for col in param_cols:
        r, _ = pearsonr(encoded[col], df[accuracy_column])
        if abs(r) >= corr_threshold:
            h1.append(col)

    # Step 3: H2 -- hyperparameter PAIRS correlated with each other
    h2_pairs = []
    for i, col_a in enumerate(param_cols):
        for col_b in param_cols[i + 1:]:
            r, _ = pearsonr(encoded[col_a], encoded[col_b])
            if abs(r) >= pair_corr_threshold:
                h2_pairs.append((col_a, col_b))

    pruned_cols = set(h1) | {c for pair in h2_pairs for c in pair}

    # Surrogate model for partial dependence (fit once, reused per feature)
    model = RandomForestRegressor(n_estimators=200, random_state=0)
    model.fit(encoded.astype(float), df[accuracy_column])

    kept_values = {}

    # Step 4a: individual PDP-based pruning for every H1 member
    for col in h1:
        idx = param_cols.index(col)
        kept_values[col] = set(_partial_dependence_keep(
            model, encoded, idx, col, encoders, is_int_col[col]
        ))

    # Step 4b: joint refinement for every H2 pair
    va_threshold = df[accuracy_column].mean()
    for col_a, col_b in h2_pairs:
        keep_a, keep_b = _pairwise_refine(df, col_a, col_b, accuracy_column, va_threshold)
        kept_values[col_a] = (kept_values.get(col_a, set(df[col_a].unique())) & keep_a) \
            or kept_values.get(col_a, keep_a)
        kept_values[col_b] = (kept_values.get(col_b, set(df[col_b].unique())) & keep_b) \
            or kept_values.get(col_b, keep_b)

    # Step 5: hyperparameters outside H1/H2 stay unpruned (full observed range)
    ranges = {}
    for col in param_cols:
        if col in pruned_cols:
            vals = sorted(kept_values[col], key=str) if col in encoders else sorted(kept_values[col])
        else:
            vals = sorted(df[col].unique().tolist(), key=str) if col in encoders else sorted(df[col].unique().tolist())

        if col not in encoders and len(vals) > N_STEPS:
            vals = _discretise_range(vals, is_int_col[col])
        ranges[col] = vals

    _write_grammar(ranges, param_cols, output_bnf_file)
    print(f"H1 (individually pruned): {h1}")
    print(f"H2 (jointly pruned pairs): {h2_pairs}")
    print(f"Pruned BNF grammar saved to {output_bnf_file}")
    return ranges


def _write_grammar(ranges, param_order, output_bnf_file):
    dict_body = ", ".join(f'"\'{c}\'": <{c}>' for c in param_order)
    lines = [f"<hyperparameters> ::= {{{dict_body}}}"]
    for col in param_order:
        values = ranges[col]
        lines.append(
            f"<{col}> ::= "
            + " | ".join(f'"\'{v}\'"' if isinstance(v, str) else str(v) for v in values)
        )
    with open(output_bnf_file, "w") as f:
        f.write("\n".join(lines) + "\n")
