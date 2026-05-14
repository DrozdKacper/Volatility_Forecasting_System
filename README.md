#  Cryptocurrency Volatility Forecasting (Time Series ML)

An end-to-end **time-series machine learning pipeline** for forecasting future Bitcoin (BTC/USDT) volatility using deep learning.

The project combines:
- time-series forecasting,
- ETL-style data pipelines,
- feature engineering,
- realistic backtesting,
- experiment tracking,
- and production-oriented ML engineering practices.

---

#  Project Overview

This project implements a modular machine learning system for predicting future cryptocurrency volatility using a **GRU (Gated Recurrent Unit)** neural network built with PyTorch.

The system follows an **ETL-style pipeline architecture**:

1. **Extract** — collect historical OHLCV market data from Binance via the CCXT API  
2. **Transform** — preprocess data, engineer time-series features, generate targets, and prepare sequential datasets  
3. **Load** — store processed datasets and feed transformed sequences into the training and evaluation pipeline  

The project focuses on building a realistic and reproducible workflow for financial time-series forecasting.

---

#  Features

## ETL-Style Data Pipeline

### Extract
- Automated historical OHLCV data collection from Binance
- API integration using CCXT
- Incremental time-series retrieval logic

### Transform
- Data cleaning and preprocessing
- Financial feature engineering
- Rolling volatility calculations
- Time-series sequence generation
- Scaling and normalization
- Leakage-safe target engineering

### Load
- Processed datasets prepared for model training
- Sequence-based dataloaders for PyTorch
- Structured outputs for evaluation and predictions

### Pipeline Reliability
- Structured logging across all ETL stages
- Robust exception handling for data ingestion and processing
- Unit tests covering core data processing logic using pytest

---

##  Feature Engineering

Engineered financial time-series features including:

- logarithmic returns (`log_return`)
- logarithmic price range (`log_range`)
- rolling volatility features
- volatility surge indicators
- volume anomaly detection
- intraday price positioning
- moving average distance
- lag-based temporal features

---

##  Target Engineering

The model predicts **future market volatility** using a forward-looking target:

- rolling standard deviation of returns
- shifted target construction to avoid data leakage
- configurable forecasting horizon

---

#  Deep Learning Model

## GRU Neural Network (PyTorch)

Model architecture:
- GRU-based sequential neural network
- configurable hidden size and number of layers
- dropout regularization
- sequence-based forecasting

Input:
- engineered time-series features

Output:
- predicted future log-volatility

---

#  Evaluation Strategy

## Walk-Forward Backtesting

The project uses **rolling window validation** to simulate realistic financial forecasting conditions.

For each split:
1. Train on historical data
2. Predict on future unseen data
3. Roll the window forward
4. Repeat evaluation

This approach:
- prevents look-ahead bias,
- avoids data leakage,
- and provides realistic performance estimation for time-series forecasting.

---

#  Benchmarking

The GRU model is evaluated against multiple statistical baselines:

- EWMA volatility model
- rolling standard deviation baseline
- persistence baseline

Evaluation metrics:
- MAE
- MSE
- correlation coefficient
- skill score vs baseline

---

#  Experiment Tracking

## MLflow Integration

The project uses MLflow for:
- hyperparameter tracking
- metric logging
- experiment management
- model artifact versioning
- reproducibility

Tracked parameters include:
- sequence length
- learning rate
- hidden size
- dropout
- batch size
- training epochs

---

#  Training Pipeline

Pipeline workflow:

1. Extract historical OHLCV data from Binance
2. Transform raw data into engineered features
3. Generate leakage-safe forecasting targets
4. Split data using rolling validation
5. Scale features with `StandardScaler`
6. Create time-series sequences
7. Train GRU model
8. Evaluate against baselines
9. Log experiments with MLflow

---

#  Tech Stack

- Python
- PyTorch
- Pandas
- NumPy
- scikit-learn
- MLflow
- CCXT

---

#  Key Concepts Demonstrated

- Time-series forecasting
- Deep learning for sequential data
- ETL-style ML pipelines
- Financial feature engineering
- Rolling-window backtesting
- Leakage prevention
- Experiment tracking
- Reproducible ML workflows
- Statistical baseline benchmarking

---

#  Planned Improvements

Future enhancements may include:

- FastAPI model serving
- Docker containerization
- CI/CD pipelines
- DVC integration
- Great Expectations validation
- NannyML monitoring
- SHAP explainability

---

#  Motivation

Financial markets are highly dynamic, noisy, and non-stationary.

This project explores whether deep learning architectures such as GRU networks can improve volatility forecasting performance compared to traditional statistical approaches while following reproducible ML engineering and ETL-style pipeline practices.
