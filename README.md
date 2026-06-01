# Retrieval-Augmented Antibiotic Ranking from Clinical Narratives

This repository contains code and aggregate results for a project exploring **retrieval-augmented antibiotic ranking** from electronic health record (EHR)-derived clinical narratives.

The project builds on the idea that biomedical language model embeddings can represent patient clinical context. A supervised classifier predicts antibiotic effectiveness probabilities, while a retrieval module finds similar historical patients and uses their observed antibiotic labels to provide an interpretable, case-based ranking.

The main goal is not to replace clinical judgment or antimicrobial susceptibility testing, but to explore how similar-patient retrieval can improve **interpretability, provenance, and physician trust** in antibiotic prediction models.

---

## Project Overview

The pipeline has two main components:

### 1. Frozen BioClinicalBERT + LightGBM Classifier

Each patient narrative is encoded using BioClinicalBERT. The transformer model is used as a **frozen feature extractor**, meaning BioClinicalBERT weights are not fine-tuned. The resulting patient embeddings are used to train one LightGBM binary classifier per antibiotic.

For each test patient, the classifier outputs a predicted probability for each antibiotic:

```text
patient_paragraph
→ BioClinicalBERT embedding
→ LightGBM classifiers
→ antibiotic probabilities
```

The eight antibiotics evaluated are:

* Clindamycin
* Erythromycin
* Gentamicin
* Levofloxacin
* Oxacillin
* Tetracycline
* Trimethoprim/Sulfa
* Vancomycin

### 2. Similar-Patient Retrieval Ranking

The retrieval system uses BioClinicalBERT embeddings to identify the top-k most similar historical patients for each test patient. Antibiotics are then ranked based on how often each antibiotic was labeled effective among those retrieved neighbors.

```text
query patient embedding
→ cosine similarity against training patient embeddings
→ top-10 similar patients
→ count neighbor antibiotic labels
→ retrieval-based antibiotic ranking
```

For each antibiotic, retrieval produces:

* `retrieval_count`: number of top-10 neighbors with positive label
* `retrieval_fraction`: retrieval_count / 10
* `retrieval_weighted_score`: similarity-weighted neighbor evidence

The retrieval layer is intended to provide a clinician-facing explanation such as:

> “The model predicts this antibiotic is likely effective, and 8 of the 10 most similar historical patients also had this antibiotic labeled effective.”

---

## Key Results

The main reported results use the row-level train/test split consistent with the original benchmark-style workflow.

### Classifier Performance

The frozen BioClinicalBERT + LightGBM classifier achieved:

| Metric | Mean Across Antibiotics |
| ------ | ----------------------: |
| AUROC  |                   0.731 |
| AUPRC  |                   0.829 |

Approximate classifier AUROC by antibiotic:

| Antibiotic         | AUROC |
| ------------------ | ----: |
| Clindamycin        | 0.733 |
| Erythromycin       | 0.775 |
| Gentamicin         | 0.620 |
| Levofloxacin       | 0.802 |
| Oxacillin          | 0.799 |
| Tetracycline       | 0.639 |
| Trimethoprim/Sulfa | 0.695 |
| Vancomycin         | 0.782 |

### Mean-Pooled Retrieval Performance

Mean-pooled BioClinicalBERT retrieval showed meaningful but weaker predictive performance than the supervised classifier:

| Method                        | Mean AUROC | Mean AUPRC |
| ----------------------------- | ---------: | ---------: |
| Retrieval fraction            |      0.626 |      0.723 |
| Similarity-weighted retrieval |      0.652 |      0.751 |
| Classifier probability        |      0.731 |      0.829 |

### Ranking Evaluation

Retrieval rankings successfully placed truly effective antibiotics higher in the ranked list:

| Retrieval Rank | True-Label Rate |
| -------------: | --------------: |
|              1 |           0.907 |
|              2 |           0.838 |
|              3 |           0.768 |
|              4 |           0.703 |
|              5 |           0.599 |
|              6 |           0.541 |
|              7 |           0.482 |
|              8 |           0.352 |

This supports the main retrieval hypothesis: antibiotics ranked higher by similar-patient evidence were more likely to be truly effective.

### Agreement Analysis

When the classifier and retrieval system agreed on the top antibiotic, the top recommendation was correct in 94.6% of cases.

| Group              | Patients | Retrieval Top-1 Hit | Classifier Top-1 Hit |
| ------------------ | -------: | ------------------: | -------------------: |
| All patients       |      599 |               0.907 |                0.958 |
| Agreement cases    |      349 |               0.946 |                0.946 |
| Disagreement cases |      250 |               0.852 |                0.976 |

These results suggest a practical design: use the classifier as the primary prediction engine, and use retrieval as an interpretable support layer that provides similar-patient provenance.

---

## Repository Structure

This public repository is organized to include code, aggregate summary results, figures, and the final report while excluding patient-level data and derived patient-level outputs.

```text
.
├── README.md
├── requirements.txt
├── scripts/
│   ├── run_frozen_validation_with_predictions.py
│   ├── encode_mean_embeddings_for_retrieval.py
│   ├── retrieval_followup_from_cached_embeddings.py
│   ├── evaluate_retrieval_followup.py
│   ├── evaluate_agreement_disagreement.py
│   └── run_grouped_frozen_classifier_and_retrieval.py
├── results_summary/
│   ├── frozen_validation_metrics_comprehensive.csv
│   ├── retrieval_vs_classifier_by_antibiotic.csv
│   ├── ranking_hit_rates.csv
│   ├── rank_position_true_label_analysis.csv
│   ├── neighborhood_similarity_analysis.csv
│   ├── combined_score_weight_sweep.csv
│   ├── agreement_disagreement_summary.csv
│   └── qsub/
│       ├── run_frozen_validation_gpu.sh
│       └── run_mean_embeddings_gpu.sh
├── figures/
└── paper/
<<<<<<< HEAD
```

---

## Privacy and Data Availability

This repository does **not** include raw patient-level data or patient-level model outputs.

The following files are intentionally excluded from the public repository:

* raw EHR-derived data files
* `antibiotics_labels.csv`
* patient-level prediction CSVs
* patient-level retrieval summaries
* detailed neighbor JSON files
* subject IDs, hospital admission IDs, and stay IDs
* cached embedding matrices
* trained model files
* cluster logs

Only aggregate summary metrics are included publicly.

Researchers with authorized access to the underlying dataset can rerun the full pipeline using the scripts in this repository.

---

## Setup

Create and activate a Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```

Core Python packages include:

* `numpy`
* `pandas`
* `scikit-learn`
* `torch`
* `transformers`
* `lightgbm`
* `tqdm`
* `matplotlib`

GPU access is recommended for BioClinicalBERT embedding extraction, but the retrieval and evaluation scripts can run on CPU once embeddings are cached.

---

## How to Run the Pipeline

The full pipeline requires authorized access to `antibiotics_labels.csv`.

### 1. Train frozen BioClinicalBERT + LightGBM classifier

```bash
python scripts/run_frozen_validation_with_predictions.py
```

This script:

1. loads `antibiotics_labels.csv`
2. extracts or loads BioClinicalBERT embeddings
3. trains one LightGBM classifier per antibiotic
4. saves aggregate metrics
5. saves private patient-level predictions

Public-safe aggregate output:

```text
outputs/frozen_validation_metrics_comprehensive.csv
```

Private outputs should not be committed.

### 2. Generate mean-pooled BioClinicalBERT embeddings

```bash
python scripts/encode_mean_embeddings_for_retrieval.py
```

On a GPU cluster, the provided qsub script can be used:

```bash
qsub qsub/run_mean_embeddings_gpu.sh
```

This creates a cached mean-pooled embedding matrix. The cache is derived from patient data and should not be committed.

### 3. Run similar-patient retrieval

```bash
python scripts/retrieval_followup_from_cached_embeddings.py
```

This script:

1. loads cached embeddings
2. recreates the train/test split
3. retrieves the top-10 similar training patients for each test patient
4. ranks antibiotics by neighbor effectiveness
5. saves private patient-level retrieval outputs

Detailed retrieval outputs contain patient identifiers and should not be made public.

### 4. Evaluate retrieval and classifier rankings

```bash
python scripts/evaluate_retrieval_followup.py
python scripts/evaluate_agreement_disagreement.py
```

These scripts generate aggregate evaluation tables, including:

* retrieval vs classifier AUROC/AUPRC
* ranking hit rates
* rank-position true-label analysis
* combined classifier-retrieval weight sweep
* agreement/disagreement analysis

Aggregate summary outputs may be placed in `results_summary/` for public release.

---

## Interpretation

The main conclusion is that retrieval is best used as an **interpretability layer**, not as a replacement for the classifier.

The classifier provides the strongest primary predictive signal. Retrieval provides complementary case-based evidence by showing similar historical patients and their observed antibiotic outcomes.

A clinician-facing version of this system could display:

* classifier probability for each antibiotic
* retrieval count, such as “8/10 similar patients effective”
* similarity-weighted retrieval support
* top neighbor provenance, if available in a secure clinical environment

This could help physicians understand why a model-ranked antibiotic is being suggested and inspect similar cases when needed.

---

## Limitations

This project has several important limitations:

1. The main results use a row-level split for consistency with the original benchmark-style workflow. Row-level splits may allow repeated patients or stays to appear across train and test.
2. Grouped split sensitivity analyses showed lower but still meaningful retrieval signal, suggesting some row-split performance may be inflated by repeated patient/stay structure.
3. BioClinicalBERT embeddings may capture documentation style or repeated note structure in addition to clinical similarity.
4. Antibiotic labels were treated as binary outcomes, but real antibiotic selection depends on infection source, organism, allergies, renal function, drug interactions, and stewardship guidelines.
5. This project is retrospective and should not be used for clinical decision-making without prospective validation.

---

## AI Tool Use Disclosure

ChatGPT and Claude Code were used to support code development, debugging, and writing organization. All final results were reviewed by the authors.

---

## Citation / Acknowledgment

If using or extending this work, please cite the original benchmark or dataset source where appropriate and acknowledge that this repository contains an exploratory student research project focused on retrieval-augmented interpretability for antibiotic ranking.

---

## Disclaimer

This repository is for research and educational purposes only. It is not intended for clinical deployment or medical decision-making.
=======
```
>>>>>>> ba4863a (update)
