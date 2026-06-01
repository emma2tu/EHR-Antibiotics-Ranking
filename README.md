# Using Similar-Patient Retrieval to Rank Antibiotic Susceptibility 

**Abstract**
**Introduction:** Antibiotic selection in the Emergency Department is time-sensitive. Researchers are able to build machine learning models to predict the right antibiotics using Electronic Health Records (EHR). However, current machine learning model predictions of selecting antibiotics can be difficult for clinicians to interpret. This project evaluated whether similar-patient retrieval can provide interpretable support through antibiotic ranking.  
**Method:** Using a cohort of 4,185 patients with Staphylococcus aureus infections from the MIMIC IV dataset, patient features were first converted into clinical narratives and embedded with frozen BioClinicalBERT. A LightGBM classifier was trained separately for each of eight antibiotics to predict susceptibility, and classifier probabilities were sorted to generate antibiotic rankings. In addition, a retrieval method identified the ten most similar training patients for each test patient using cosine similarity over BioClinicalBERT embeddings, then ranked antibiotics based on neighbor susceptibility.  
**Results:** The supervised BioClinicalBERT + LightGBM classifier achieved the strongest overall predictive performance, with a mean AUROC of 0.731 and mean AUPRC of 0.829 across antibiotics. Retrieval-only methods had lower AUROC/AUPRC, but produced clinically meaningful rankings, where the top ranked antibiotic was truly effective in 90.7% of test cases. 
**Discussion:** These results suggest that retrieval should not replace the classifier approach for antibiotic susceptibility prediction, but can serve as an interpretable layer by showing similar historical patients whose outcomes support the recommendation.

The eight antibiotics evaluated are:

* Clindamycin
* Erythromycin
* Gentamicin
* Levofloxacin
* Oxacillin
* Tetracycline
* Trimethoprim/Sulfa
* Vancomycin

---

## Project Overview

The pipeline has two main components:

### 1. Frozen BioClinicalBERT + LightGBM Classifier (Recreating Lee et. al's work)

Each patient narrative is encoded using BioClinicalBERT. The transformer model is used as a **frozen feature extractor**, meaning BioClinicalBERT weights are not fine-tuned. The resulting patient embeddings are used to train one LightGBM binary classifier per antibiotic.

For each test patient, the classifier outputs a predicted probability for each antibiotic:

For each patient, there is:
* BioClinicalBERT embedding
* LightGBM classifiers
* antibiotic probabilities

### 2. Similar-Patient Retrieval Ranking

The retrieval system uses BioClinicalBERT embeddings to identify the top-k most similar historical patients for each test patient. Antibiotics are then ranked based on how often each antibiotic was labeled effective among those retrieved neighbors.


For each patient, we find:
* cosine similarity against training patient embeddings
* top-10 similar patients
* count neighbor antibiotic labels
* retrieval-based antibiotic ranking

For each antibiotic, retrieval produces:
* `retrieval_count`: number of top-10 neighbors with positive label
* `retrieval_fraction`: retrieval_count / 10
* `retrieval_weighted_score`: similarity-weighted neighbor evidence

The retrieval layer is intended to provide a clinician-facing explanation such as:

> “The model predicts this antibiotic is likely effective, and 8 of the 10 most similar historical patients also had this antibiotic labeled effective.”

---

## Key Results

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

These results suggest that retrieval could be used as an interpretable support layer for patient antibiotic susceptibility prediction.

