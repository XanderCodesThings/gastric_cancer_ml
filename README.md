# Repository Overview
*Note: This project originated as part of a university AI course. The original project was developed collaboratively in a private repository. This repository contains only my independently written code, experiments, documentation, and analysis.*

Research Area: Predictive Analytics for Healthcare

Project Title: Diagnosis of Gastric Cancer with Machine Learning

## Problem Statement

Historically, diagnosing gastric cancer has been an extremely invasive process. These invasive procedures expose patients to more risks, possibly complicating diagnosis and treatment. The purpose of this project is to evaluate whether machine learning can predict the presence of gastric cancer using non-invasive biomarkers.

## Methods (Literature Review Summary)

This study evaluates the following models:

1. Gradient Boosting Decision Tree (GBDT)
2. Random Forests

## Results

| Model | ROC-AUC |
|--------|--------:|
| Random Forest | 0.885 |
| Gradient Boosting Decision Tree | 0.879 |

Both models successfully discriminated between gastric cancer and control patients using only non-invasive clinical biomarkers.

## Experimental Setup

- Dataset: 709 patient records (705 after dropping 4 duplicate rows, see Dataset notes) with 22 non-invasive clinical features, labeled as gastric cancer versus control. The dataset is available from the publication listed below. All 22 features are used as model inputs (the full routine blood panel: demographics, complete blood count, biochemistry and lipids, and tumor markers).
- Input Modalities: Variables covering age, gender, blood cell count, liver function, kidney function, blood lipids, and tumor markers.
- Evaluation Metrics: ROC, AUC, accuracy, sensitivity, and specificity (as used in the source publication).
- Implementation Framework: Scikit-Learn.
- Validation: Stratified cross-validation, with checks for duplicate rows and possible data leakage before comparing model results.

## Dataset / Paper for Replication

Zhu et al., *PLOS ONE* 2020. "Application of machine learning in the diagnosis of gastric cancer based on noninvasive characteristics."

https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0244869

## Dataset notes

File: `data/gastric.csv`, 709 rows by 23 columns (`label` plus 22 features). One row per patient.

The paper's open-access S1 Data is a 28-page PDF table. We extracted it by word position into two horizontal panels and joined them by patient order. We dropped one redundant column from the published table, `f6_bin`, which was just `lymphocytes >= 2.0` and added no information beyond `lym`, leaving 22 features.

Label: `label`, where 1 = gastric cancer and 0 = control. There are 398 cancer and 311 control patients (about 56% positive). The 311 controls are 202 benign gastric disease patients plus 109 healthy people, so "control" means non-cancer, not only healthy.

Features (all numeric, all from a routine blood draw):

- Demographics: `gender` (0/1), `age` (binned by decade, values 2 to 9 meaning 20s to 90s).
- Complete blood count: `neu`, `lym`, `neu_lym` (neutrophil-to-lymphocyte ratio), `hb`, `hct`, `rdw`, `plt`.
- Biochemistry and lipids: `alb` (albumin), `alt`, `tb` (total bilirubin), `cr` (creatinine), `tg` (triglycerides), `chol`, `hdl`, `ldl`, `lpa` (lipoprotein).
- Tumor markers: `cea`, `ca125`, `ca199`, `ca724`.

The data has no missing values, all 22 features are numeric, and every value was checked against the source PDF. EDA found 4 exact duplicate rows (all cancer cases), most likely introduced when the two PDF panels were joined by patient order, so `model_comparison.py` drops them before training, leaving 705 patients (394 cancer, 311 control) for modeling. A few lab values are extreme, but most are real (the tumor markers are genuinely high in cancer patients), so we keep the data as published and let the models handle scaling. There is no separate "cleaned" file; `data/gastric.csv` is the single source of truth and the duplicate drop happens in code.

## Scripts
