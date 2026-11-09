# Financial Risk Profiling — tabular ML and financial-language experiments

**An exploratory notebook investigating two complementary signals for company bankruptcy risk: financial statements and financial text.** The tabular track engineers balance-sheet ratios, builds classical risk-score features, and compares classification approaches under class imbalance. The language track explores FinBERT sentiment inference and includes a fine-tuning workflow using labeled financial sentences.

The repository is a **research notebook**, not a deployed lending decision system. Its strength is the breadth of the investigation: financial-ratio engineering, baseline comparison, resampling strategies, and a transformer-based text track in one place.

[Open the notebook in Google Colab](https://colab.research.google.com/github/vldzio/Multimodal-Risk-Profiling/blob/main/US_bankrupt.ipynb)

## What is in the notebook

### 1. Financial-risk features

- Reads `american_bankruptcy.csv` and selects the latest available record per company.
- Renames statement fields and computes features based on working capital, retained earnings, EBIT, revenue, leverage, liquidity, and profitability.
- Explores **Altman Z-score**, **Zmijewski-style**, and **Ohlson-style** risk indicators and combines their risk categories into a rule-based profile.

### 2. Classifier and imbalance experiments

- Prepares `alive` versus `failed` labels and explores random undersampling and SMOTE-based oversampling.
- Includes comparison code for **Logistic Regression, Random Forest, XGBoost, SVM, and KNN**.
- Calculates classification metrics including accuracy, precision, recall, F1, ROC-AUC, confusion matrices, and ROC curves.

### 3. Financial-text exploration

- Loads the `yiyanghkust/finbert-tone` model for example financial-sentence sentiment inference.
- Prepares labeled financial sentences from a separate `phrasebank_data.csv` file.
- Contains Hugging Face `Trainer` configurations and training calls for experimenting with FinBERT fine-tuning, including an F1-oriented configuration.

These are **two research tracks in the current notebook**. It does not yet show a trained, evaluated model that fuses a company's financial ratios with matched text-derived sentiment features.

## Reproducing the exploration

This repository contains `US_bankrupt.ipynb` and this README. The source CSVs and trained checkpoints are **not included**; you need compatible data you are authorized to use.

1. Open the [notebook in Colab](https://colab.research.google.com/github/vldzio/Multimodal-Risk-Profiling/blob/main/US_bankrupt.ipynb), save a copy, and mount Google Drive when prompted.
2. Provide the financial dataset as `american_bankruptcy.csv` (with the notebook's expected `company_name`, `year`, `status_label`, and `X1`–`X18` fields) and, for the NLP section, labeled financial sentences as `phrasebank_data.csv` (with `Sentence` and `Sentiment` fields).
3. Update the notebook's absolute Drive paths for your environment. The current cells mix `/content/drive/MyDrive/...` and `/content/drive/My Drive/...`; make them consistent **before** running cells. The original workflow expects the finance file inside a `USBDT` folder and the text file inside `Phrasebank`.
4. Install missing notebook libraries in Colab if needed (the notebook already contains some `pip install` cells):

   ```python
   %pip install scikit-learn imbalanced-learn xgboost pandas numpy matplotlib seaborn scipy transformers datasets accelerate evaluate torch
   ```

5. Run and review the finance and NLP sections separately, adapting any dataset-specific labels or file paths. Google Colab or an environment with sufficient CPU/GPU memory is recommended for the fine-tuning cells.

## Evaluating responsibly

The checked-in notebook contains **no saved execution outputs or benchmark table**, so this README does not invent an accuracy or claim an improvement over a baseline. Before citing results:

- Split into training and held-out data **before** fitting scalers, computing any train-derived transforms, or applying SMOTE; the current SMOTE section scales and resamples before splitting, which can leak information into evaluation.
- Report class support and failed-class precision/recall/F1 alongside ROC-AUC; overall accuracy alone can conceal performance on the minority class.
- Keep the financial-ratio and text experiments separate until the data can be matched by company and time, a fusion method is implemented, and a fair baseline comparison is recorded.
- Clearly identify the dataset's population and time period; a general company-bankruptcy dataset should not automatically be described as an SME-only dataset.

## Next milestone

A compelling next extension is a **reproducible, leakage-controlled benchmark** for tabular risk, followed by time-aligned text features and an explicit tabular-only versus tabular-plus-sentiment comparison. The existing notebook provides feature engineering, model families, imbalance experiments, and financial NLP groundwork for that step.

## Live demo

An interactive demo lives in [`demo/`](demo) (static UI, deploys to Vercel) backed by [`demo_api/`](demo_api) (FastAPI, deploys as a Hugging Face Space together with a private [FreeLLMAPI](https://github.com/tashfeenahmed/freellmapi) gateway):

| Feature | How it works |
| --- | --- |
| Altman Z · Zmijewski · Ohlson scores | The notebook's exact feature definitions and coefficients, as plain Python (`demo_api/risk_core.py`, unit-tested). Instant, deterministic, no LLM. |
| AI analyst narrative | LLM output generated live from the computed numbers (schema-validated JSON). |
| Sentence sentiment | LLM labels financial sentences positive / negative / neutral (the notebook's FinBERT track lives in Colab). |

Every LLM call is traced to a LangSmith project (`risk-demo`). Visitors get a small daily allowance and can bring their own key. Deploy recipe: [`deploy/space/`](deploy/space) and [`vercel.json`](vercel.json) (set `DEMO_API_URL`). Run tests with `pip install -r demo_api/requirements.txt pytest && pytest tests`.
