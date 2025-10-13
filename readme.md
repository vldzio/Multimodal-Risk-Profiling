# Multimodal Financial Risk Profiling

A hybrid machine learning system that predicts SME bankruptcy risk by combining quantitative financial metrics with qualitative sentiment analysis from financial texts. Addresses the limitations of traditional models by integrating both structured financial data and market sentiment signals.

## Overview

Traditional bankruptcy prediction models rely solely on financial ratios, missing critical qualitative signals. This project builds an end-to-end solution that:
- Analyzes structured financial data using classical risk metrics (Altman Z-score, Ohlson O-score, Zmijewski Score)
- Extracts sentiment from financial news and reports using **FinBERT** (transformer-based NLP)
- Combines both data sources into a unified risk assessment model

## Key Features

**Financial Risk Metrics**
- Altman Z-score, Ohlson O-score, and Zmijewski Score calculation
- Automated financial ratio computation and normalization

**Machine Learning Pipeline**
- Handles severe class imbalance with **SMOTE** and resampling techniques
- Multiple classifiers: Logistic Regression, Random Forest, XGBoost, SVM, KNN
- Performance tuned for imbalanced data using Precision, Recall, F1-score, AUC-ROC

**Sentiment Analysis**
- Fine-tuned **FinBERT** model for financial domain text
- Processes news articles, earnings reports, and market commentary
- Extracts sentiment scores as features for risk prediction

**Hybrid Model**
- Fuses quantitative (financial ratios) and qualitative (sentiment) data
- Improved prediction accuracy over single-modality approaches
- Robust evaluation with confusion matrices and ROC curves

## Tech Stack

- **Python** - Core implementation
- **FinBERT** - Financial sentiment analysis (transformer-based)
- **Scikit-learn** - ML classifiers and evaluation
- **XGBoost** - Gradient boosting classifier
- **SMOTE (imbalanced-learn)** - Class balancing
- **Pandas/NumPy** - Data processing
- **Matplotlib/Seaborn** - Visualizations

## Use Case

Predicts bankruptcy risk for SMEs by analyzing:
1. **Structured data**: Balance sheets, income statements, cash flow
2. **Unstructured data**: Financial news, analyst reports, market sentiment

The hybrid approach captures both financial health indicators and market perception, providing a more comprehensive risk profile.

## Performance Metrics

- Precision, Recall, F1-score for imbalanced classification
- AUC-ROC curves for model comparison
- Confusion matrices for error analysis
- Feature importance rankings

## Why This Matters

SME bankruptcies have significant economic impact. Early prediction enables:
- Proactive risk management for lenders
- Better investment decisions
- Targeted interventions for struggling businesses

This project demonstrates how combining traditional financial analysis with modern NLP techniques creates more robust risk assessment tools.
