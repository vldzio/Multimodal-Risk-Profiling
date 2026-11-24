# Multimodal Risk Profiling — Company Bankruptcy Prediction

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vldzio/Multimodal-Risk-Profiling/blob/main/US_bankrupt.ipynb)

Predicting company bankruptcy risk from two complementary signals: **financial statements** and **financial language**. The tabular track engineers balance-sheet ratios, computes the classic Altman, Zmijewski and Ohlson risk scores, and benchmarks five classifiers under class imbalance. The language track runs FinBERT sentiment inference and fine-tunes FinBERT on labeled financial sentences. The two tracks are explored independently — they are not combined into one trained model; fusing them is future work (see [Roadmap](#roadmap)).

The research lives in a single Colab notebook, [`US_bankrupt.ipynb`](US_bankrupt.ipynb). The **RiskLens** web demo puts the scoring models and FinBERT in front of anyone with a browser.

**Live demo:** [risk-profiling-demo.vercel.app](https://risk-profiling-demo.vercel.app)

## What's in the notebook

### 1. Financial-ratio features and classic risk scores

- Loads `american_bankruptcy.csv`, keeps the most recent filing per company, and maps the raw `X1`–`X18` fields to readable statement names.
- Engineers ratios: working capital / total assets, retained earnings / total assets, EBIT / total assets, market value / total liabilities, sales / total assets, ROA, leverage, current ratio, log total assets and a negative-net-income flag.
- Computes the **Altman Z-score** (Safe / Gray / Distress zones), the **Zmijewski score** (probit probability) and the **Ohlson O-score** (logistic probability), then combines them into a rule-based overall risk profile.

### 2. Classifier comparison with imbalance handling

- One stratified train/test split; the scaler is fit on the training data, and the held-out test set keeps the real class balance.
- Two training-set balancing strategies: **random undersampling** and **SMOTE oversampling**.
- Five models trained under each strategy: **Logistic Regression, Random Forest, XGBoost, SVM and KNN**.
- Every model reports failed-class precision, recall, F1 and PR-AUC, a per-class report with support, ROC-AUC, a confusion matrix and a ROC curve.

### 3. FinBERT sentiment and fine-tuning

- Sentiment inference with [`yiyanghkust/finbert-tone`](https://huggingface.co/yiyanghkust/finbert-tone), for single sentences and batches.
- Cleans and labels the Financial PhraseBank sentences, aligned with the model's own label mapping.
- Fine-tunes FinBERT with the Hugging Face `Trainer`, including early stopping and a weighted-F1 evaluation metric.

## Data

Place both files in Google Drive under `DATA_DIR` (default `/content/drive/MyDrive`):

```
DATA_DIR/
├── USBDT/american_bankruptcy.csv
└── Phrasebank/phrasebank_data.csv
```

| File | Fields |
| --- | --- |
| `american_bankruptcy.csv` | `company_name`, `status_label` (`alive` / `failed`), `year`, `X1`–`X18` (current assets, COGS, D&A, EBITDA, inventory, net income, receivables, market value, net sales, total assets, long-term debt, EBIT, gross profit, current liabilities, retained earnings, revenue, total liabilities, operating expenses) |
| `phrasebank_data.csv` | `Sentence`, `Sentiment` (`positive` / `neutral` / `negative`) |

The notebook writes intermediate files (`preprocessed_df.csv` and the resampled training sets) back to `DATA_DIR/USBDT/`.

## How to run

1. Click the **Open in Colab** badge above and save a copy to your Drive.
2. Run the first cell to mount Google Drive. If your data lives elsewhere, change `DATA_DIR` in that cell — every path in the notebook is built from it.
3. Install the dependencies (the notebook also includes the install cells it needs):

   ```python
   %pip install scikit-learn imbalanced-learn xgboost pandas numpy matplotlib seaborn scipy transformers datasets accelerate evaluate torch
   ```

4. Run the cells top to bottom. The financial and FinBERT sections are independent and can be run separately; a GPU runtime is recommended for fine-tuning.

## RiskLens web demo

A static UI in [`demo/`](demo) (Vercel) backed by a FastAPI service in [`demo_api/`](demo_api) (Render):

| Feature | How it works |
| --- | --- |
| Altman Z, Zmijewski and Ohlson scores | The notebook's feature definitions and coefficients as plain Python (`demo_api/risk_core.py`, unit-tested). Instant and deterministic. |
| Sentence sentiment | The notebook's `yiyanghkust/finbert-tone`, int8-quantized ONNX on `onnxruntime`. |
| Analyst narrative | An LLM explains the computed scores in plain language (schema-validated JSON, traced in LangSmith). |

Run it locally:

```bash
pip install -r demo_api/requirements.txt pytest
python scripts/export_finbert_onnx.py          # builds demo_api/model/ for the sentiment endpoint
pytest tests
uvicorn demo_api.app:app --port 8140
```

Set `API_URL` in `demo/config.js` to `http://localhost:8140` and serve `demo/` with `python -m http.server`. Deployment recipes: `render.yaml`, `deploy/space/` and `vercel.json`.

## Tech stack

Python · pandas · NumPy · scikit-learn · imbalanced-learn · XGBoost · SciPy · Matplotlib · seaborn · PyTorch · Hugging Face Transformers / Datasets / Evaluate · FastAPI · ONNX Runtime

## Roadmap

Fuse the two signals: match FinBERT sentiment from company filings and news to each firm's financial ratios by company and year, and train a joint tabular-plus-text risk model.
