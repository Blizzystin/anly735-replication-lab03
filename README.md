# ANLY 735 Replication Laboratory #3

## When the Test Environment Changes

**Researcher:** Arvind Natarajan  
**Replication type:** Partial replication of the WILDS distribution-shift evaluation logic

## Replication Question

What happens to the conclusions about a fixed toxicity classifier when evaluation moves from a source-like in-distribution holdout to the shifted CivilComments-WILDS test environment?

## Design

The analysis uses **CivilComments-WILDS** with one fixed modeling pipeline:

- TF-IDF text features
- word unigrams and bigrams
- maximum vocabulary of 30,000 features
- L2-regularized logistic regression
- `C = 1.0`
- `liblinear` solver
- probability threshold = `0.50`
- random seed = `735`

The official WILDS training split is the development environment. A stratified 20% holdout from that split is used for **in-distribution (ID)** evaluation. The official WILDS test split is used as the **shifted** evaluation environment.

The model is fitted once. It is not retrained or retuned after shifted performance is observed.

## Main Result

Aggregate performance is essentially unchanged across the two evaluation environments.

| Metric | ID | Shifted |
|---|---:|---:|
| ROC-AUC | 0.9027 | 0.9030 |
| PR-AUC | 0.6289 | 0.6302 |
| Accuracy | 0.9124 | 0.9127 |
| Worst-group accuracy | 0.2203 | 0.2552 |

The important result appears below the average: the weakest identity-label subgroup performs poorly in both environments. The shifted worst-group accuracy is **0.2552**.

**Replication verdict:** Partially Reproduced  
**Deployment judgment:** REVALIDATE

## Repository Structure

```text
.
├── analysis/
│   ├── lab03_results.csv
│   └── lab03_subgroup_metrics.csv
├── python/
│   └── lab03_analysis.py
├── ANLY735_Replication_Lab_03_CivilComments_WILDS.ipynb
├── Replication_Lab_3_Final.docx
├── replication-lab.qmd
├── requirements.txt
├── .gitignore
└── README.md
```

## Reproduce the Analysis

Install the required packages:

```bash
pip install -r requirements.txt
```

Then run:

```bash
python python/lab03_analysis.py
```

The script downloads the public processed CivilComments-WILDS mirror and writes the analytical evidence to the `analysis/` directory.

The executed notebook is also included so that the complete computational workflow and outputs can be inspected directly.

## Data

The repository does **not** store the CivilComments dataset.

The analysis loads the public processed mirror:

`pietrolesci/civilcomments-wilds`

This keeps the repository small and makes the data source explicit.

## Reproducibility

The executed analysis used:

- Python 3.13.16
- NumPy 2.1.3
- pandas 2.2.3
- scikit-learn 1.6.1
- datasets 4.8.5
- pyarrow 23.0.1
- random seed 735

## Reference

Koh, P. W., Sagawa, S., Marklund, H., et al. (2021). *WILDS: A benchmark of in-the-wild distribution shifts*. Proceedings of the 38th International Conference on Machine Learning, 139, 5637-5664.
