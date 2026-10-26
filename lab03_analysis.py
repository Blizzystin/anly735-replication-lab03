"""
ANLY 735 - Replication Laboratory #3
When the Test Environment Changes

Partial replication of the WILDS distribution-shift evaluation logic
using CivilComments-WILDS.

Researcher: Arvind Natarajan
"""

import json
import platform
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

import datasets
import pyarrow
import sklearn

warnings.filterwarnings("ignore", category=UserWarning)

SEED = 735
np.random.seed(SEED)

OUT = Path("analysis")
OUT.mkdir(exist_ok=True)

IDENTITIES = [
    "male",
    "female",
    "LGBTQ",
    "christian",
    "muslim",
    "other_religions",
    "black",
    "white",
]


# ------------------------------------------------------------
# Load CivilComments-WILDS
# ------------------------------------------------------------
try:
    ds = load_dataset("pietrolesci/civilcomments-wilds", "default")
    train_full = ds["train"].to_pandas()
    wilds_val = ds["validation"].to_pandas()
    ood_test = ds["test"].to_pandas()
    DATA_SOURCE = "Hugging Face mirror: pietrolesci/civilcomments-wilds"
except Exception:
    # Direct parquet fallback if the Hugging Face loader changes.
    base = (
        "https://huggingface.co/datasets/"
        "pietrolesci/civilcomments-wilds/resolve/main/data"
    )
    train_full = pd.read_parquet(f"{base}/train-00000-of-00001.parquet")
    wilds_val = pd.read_parquet(
        f"{base}/validation-00000-of-00001.parquet"
    )
    ood_test = pd.read_parquet(f"{base}/test-00000-of-00001.parquet")
    DATA_SOURCE = "Direct Parquet mirror: pietrolesci/civilcomments-wilds"


required = {"comment_text", "toxicity", *IDENTITIES}
for name, df in {
    "train": train_full,
    "validation": wilds_val,
    "test": ood_test,
}.items():
    missing = required.difference(df.columns)
    assert not missing, f"{name} missing columns: {missing}"
    assert df["comment_text"].notna().all()
    assert set(df["toxicity"].unique()).issubset({0, 1})


# ------------------------------------------------------------
# Define development, ID, and shifted environments
# ------------------------------------------------------------
train_dev, id_test = train_test_split(
    train_full,
    test_size=0.20,
    stratify=train_full["toxicity"],
    random_state=SEED,
)

train_dev = train_dev.reset_index(drop=True)
id_test = id_test.reset_index(drop=True)
ood_test = ood_test.reset_index(drop=True)


# ------------------------------------------------------------
# Fit one fixed modeling pipeline
# ------------------------------------------------------------
vectorizer = TfidfVectorizer(
    lowercase=True,
    strip_accents="unicode",
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.98,
    max_features=30_000,
    sublinear_tf=True,
    dtype=np.float32,
)

X_train = vectorizer.fit_transform(train_dev["comment_text"])
X_id = vectorizer.transform(id_test["comment_text"])
X_ood = vectorizer.transform(ood_test["comment_text"])

y_train = train_dev["toxicity"].to_numpy()
y_id = id_test["toxicity"].to_numpy()
y_ood = ood_test["toxicity"].to_numpy()

model = LogisticRegression(
    penalty="l2",
    C=1.0,
    solver="liblinear",
    max_iter=1000,
    random_state=SEED,
)
model.fit(X_train, y_train)

p_id = model.predict_proba(X_id)[:, 1]
p_ood = model.predict_proba(X_ood)[:, 1]


# ------------------------------------------------------------
# Overall evaluation
# ------------------------------------------------------------
def expected_calibration_error(y_true, prob, n_bins=10):
    y_true = np.asarray(y_true)
    prob = np.asarray(prob)
    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(prob, bins[1:-1], right=True)

    ece = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.any():
            ece += (
                mask.mean()
                * abs(y_true[mask].mean() - prob[mask].mean())
            )
    return float(ece)


def evaluate(y_true, prob, threshold=0.50):
    pred = (prob >= threshold).astype(int)
    return {
        "ROC-AUC": roc_auc_score(y_true, prob),
        "PR-AUC": average_precision_score(y_true, prob),
        "Brier": brier_score_loss(y_true, prob),
        "ECE": expected_calibration_error(y_true, prob),
        "Accuracy": accuracy_score(y_true, pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, pred),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
    }


id_metrics = evaluate(y_id, p_id)
ood_metrics = evaluate(y_ood, p_ood)

results = pd.DataFrame(
    [
        {
            "Metric": metric,
            "In-Distribution": id_metrics[metric],
            "Shifted Environment": ood_metrics[metric],
            "Change (OOD - ID)": ood_metrics[metric] - id_metrics[metric],
        }
        for metric in id_metrics
    ]
)


# ------------------------------------------------------------
# Subgroup evaluation
# ------------------------------------------------------------
def safe_auc(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan


def subgroup_metrics(df, y, p, environment):
    pred = (p >= 0.50).astype(int)
    rows = []

    for identity in IDENTITIES:
        mask = df[identity].to_numpy() == 1
        if mask.sum() == 0:
            continue

        yy = y[mask]
        pp = p[mask]
        pr = pred[mask]

        tn, fp, fn, tp = confusion_matrix(
            yy, pr, labels=[0, 1]
        ).ravel()

        rows.append(
            {
                "Environment": environment,
                "Identity": identity,
                "n": int(mask.sum()),
                "Prevalence": yy.mean(),
                "ROC-AUC": safe_auc(yy, pp),
                "PR-AUC": (
                    average_precision_score(yy, pp)
                    if yy.sum() > 0
                    else np.nan
                ),
                "Accuracy": accuracy_score(yy, pr),
                "Recall / TPR": (
                    tp / (tp + fn) if (tp + fn) else np.nan
                ),
                "Specificity / TNR": (
                    tn / (tn + fp) if (tn + fp) else np.nan
                ),
                "Brier": brier_score_loss(yy, pp),
            }
        )

    return pd.DataFrame(rows)


subgroups = pd.concat(
    [
        subgroup_metrics(id_test, y_id, p_id, "ID"),
        subgroup_metrics(ood_test, y_ood, p_ood, "Shifted"),
    ],
    ignore_index=True,
)


def identity_label_groups(df, y, p, environment):
    pred = (p >= 0.50).astype(int)
    rows = []

    for identity in IDENTITIES:
        ident = df[identity].to_numpy() == 1
        for label in (0, 1):
            mask = ident & (y == label)
            if mask.sum() == 0:
                continue

            rows.append(
                {
                    "Environment": environment,
                    "Identity": identity,
                    "Label": label,
                    "n": int(mask.sum()),
                    "Accuracy": accuracy_score(y[mask], pred[mask]),
                }
            )

    return pd.DataFrame(rows)


identity_label = pd.concat(
    [
        identity_label_groups(id_test, y_id, p_id, "ID"),
        identity_label_groups(
            ood_test, y_ood, p_ood, "Shifted"
        ),
    ],
    ignore_index=True,
)

worst = (
    identity_label.sort_values("Accuracy")
    .groupby("Environment", as_index=False)
    .first()
)


# ------------------------------------------------------------
# Environment diagnostics
# ------------------------------------------------------------
def environment_diagnostics(df, name):
    lengths = df["comment_text"].str.split().str.len()

    return {
        "Environment": name,
        "n": len(df),
        "Toxicity prevalence": df["toxicity"].mean(),
        "Any identity mention": df[IDENTITIES].any(axis=1).mean(),
        "Median words": lengths.median(),
        "Mean words": lengths.mean(),
        "90th pct words": lengths.quantile(0.90),
    }


diagnostics = pd.DataFrame(
    [
        environment_diagnostics(train_dev, "Development"),
        environment_diagnostics(id_test, "ID"),
        environment_diagnostics(ood_test, "Shifted"),
    ]
)

identity_shift = pd.DataFrame(
    [
        {
            "Identity": identity,
            "ID proportion": id_test[identity].mean(),
            "Shifted proportion": ood_test[identity].mean(),
            "Change": (
                ood_test[identity].mean()
                - id_test[identity].mean()
            ),
        }
        for identity in IDENTITIES
    ]
).sort_values("Change", key=np.abs, ascending=False)

id_mean = np.asarray(X_id.mean(axis=0)).ravel()
ood_mean = np.asarray(X_ood.mean(axis=0)).ravel()
terms = np.asarray(vectorizer.get_feature_names_out())
delta = ood_mean - id_mean
idx = np.argsort(np.abs(delta))[::-1][:20]

term_shift = pd.DataFrame(
    {
        "Term": terms[idx],
        "ID mean TF-IDF": id_mean[idx],
        "Shifted mean TF-IDF": ood_mean[idx],
        "Difference": delta[idx],
    }
)


# ------------------------------------------------------------
# Deployment judgment
# ------------------------------------------------------------
id_worst = float(
    identity_label[
        identity_label["Environment"] == "ID"
    ]["Accuracy"].min()
)
ood_worst = float(
    identity_label[
        identity_label["Environment"] == "Shifted"
    ]["Accuracy"].min()
)

auc_drop = id_metrics["ROC-AUC"] - ood_metrics["ROC-AUC"]
brier_worsen = ood_metrics["Brier"] - id_metrics["Brier"]

if (
    auc_drop >= 0.03
    or brier_worsen >= 0.02
    or ood_worst < 0.70
):
    deployment_judgment = "REVALIDATE"
elif any(
    ood_metrics[m] < id_metrics[m]
    for m in [
        "ROC-AUC",
        "PR-AUC",
        "Accuracy",
        "Balanced Accuracy",
    ]
):
    deployment_judgment = "DEPLOY WITH SAFEGUARDS"
else:
    deployment_judgment = "DEPLOY"


# ------------------------------------------------------------
# Save evidence
# ------------------------------------------------------------
results_with_worst = pd.concat(
    [
        results,
        pd.DataFrame(
            [
                {
                    "Metric": "Worst-group accuracy",
                    "In-Distribution": id_worst,
                    "Shifted Environment": ood_worst,
                    "Change (OOD - ID)": ood_worst - id_worst,
                }
            ]
        ),
    ],
    ignore_index=True,
)

results_with_worst.to_csv(
    OUT / "lab03_results.csv", index=False
)
subgroups.to_csv(
    OUT / "lab03_subgroup_metrics.csv", index=False
)
identity_label.to_csv(
    OUT / "lab03_identity_label_groups.csv", index=False
)
diagnostics.to_csv(
    OUT / "lab03_environment_diagnostics.csv", index=False
)
identity_shift.to_csv(
    OUT / "lab03_identity_composition_shift.csv", index=False
)
term_shift.to_csv(
    OUT / "lab03_top_shifted_terms.csv", index=False
)

session_info = {
    "python": sys.version,
    "platform": platform.platform(),
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "scikit_learn": sklearn.__version__,
    "datasets": datasets.__version__,
    "pyarrow": pyarrow.__version__,
    "seed": SEED,
    "data_source": DATA_SOURCE,
    "deployment_judgment": deployment_judgment,
}

with open(OUT / "session_info.json", "w") as f:
    json.dump(session_info, f, indent=2)

print("\nOverall results")
print(results_with_worst.round(4).to_string(index=False))

print("\nWorst identity-label group by environment")
print(worst.round(4).to_string(index=False))

print("\nDeployment judgment:", deployment_judgment)
