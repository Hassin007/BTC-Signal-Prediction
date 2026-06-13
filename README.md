# BTC Direction Classifier

A machine learning project that predicts the **next-day direction of Bitcoin (BTC/USD)** as one of three signals: **BUY**, **HOLD**, or **SELL**.

Built as a Complex Engineering Problem for **CS 324 – Machine Learning**, NED University of Engineering & Technology.

**Team:** Azen Sajid (CS-23103), Showzab Mehdi (CS-23128), M. Hassin Saghir (CS-23129)
**Instructor:** Dr. Maria Waqas

---

## 📌 Project Overview

The goal is to predict whether BTC will make a significant **upward move (BUY)**, **downward move (SELL)**, or **stay flat (HOLD)** the next day, using daily price, technical, macro, and sentiment data.

- **Dataset:** Daily BTC/USD data, July 2018 – May 2026 (~2,843 days)
- **Features:** 22–28 engineered features (price, volatility, technical indicators, S&P 500, NASDAQ, VIX, Fear & Greed Index, USD Index)
- **Validation:** 5-fold walk-forward (no look-ahead bias)
- **Models compared:** LightGBM, Logistic Regression, LSTM

---

## 🏷️ How Labels Are Made

A day is labeled BUY/SELL only if the next day's return exceeds **1.0 standard deviation** of recent volatility — otherwise it's HOLD.

> 🔑 **Key Insight:** Switching the threshold from 0.5σ to 1.0σ was the single biggest improvement in the whole project — bigger than any model tuning. It dropped log-loss from 1.16 (worse than random) to 0.83.

Label split: ~33% BUY, 33% SELL, 34% HOLD (nicely balanced).

---

## 🤖 The Three Models

### 1. LightGBM (Winner 🏆)
- Gradient-boosted decision trees
- Handles non-linear, irregular patterns in the data
- Tuned with Optuna (60 trials, 9 hyperparameters)

### 2. Logistic Regression
- Simple linear model with ElasticNet regularization
- Fully interpretable (shows exactly which features matter)
- Confirms BTC's behavior is **not linear**

### 3. LSTM (Deep Learning)
- Recurrent neural network for sequential/time-series patterns
- Started broken, fixed through 3 targeted changes:
  1. Shorter sequence length (30 → 7 days) → more training data
  2. Removed pre-computed lag features → forces real temporal learning
  3. Added jitter data augmentation → fixed class imbalance

---

## 📊 Results Summary

| Metric | Random | Logistic Regression | LSTM (Best) | **LightGBM (Best)** |
|---|---|---|---|---|
| Avg. Log-Loss | 1.099 | ~0.92–0.96 | ~1.05–1.09 | **~0.79** |
| Avg. Sharpe Ratio | — | Moderate | Positive | **~0.84 (Best)** |
| Avg. Return/Fold | — | Positive, modest | Positive | **~+28%** |
| Max Drawdown | — | Moderate | ~-10 to -15% | **Lowest** |
| Interpretability | — | Full | None | Partial |

---

## ✅ Key Achievements

- ✅ Built and compared 3 very different ML architectures fairly using the same data and validation
- ✅ Discovered that **label design matters more than model choice**
- ✅ Used strict **walk-forward validation** to avoid look-ahead bias (a common mistake in finance ML)
- ✅ Tuned LightGBM with Optuna to reach the best log-loss (~0.79) and best Sharpe ratio (~0.84)
- ✅ Took a **broken LSTM** (negative Sharpe, 2/5 folds with zero trades) and fixed it into a profitable model through 3 targeted interventions
- ✅ Used Logistic Regression to interpret *why* the models make certain predictions (e.g., Bollinger Band breakouts → BUY signal)
- ✅ Applied temperature scaling to calibrate LSTM probability outputs
- ✅ Always evaluated models on **financial metrics (Sharpe, drawdown)**, not just accuracy/log-loss — because a statistically good model can still lose money

---

## 🧠 Main Takeaways

1. **Label quality > model sophistication.** The 1.0σ threshold decision mattered more than any tuning.
2. **Walk-forward validation is mandatory** for any time-series financial model.
3. **Bitcoin's behavior is non-linear** — that's why LightGBM beats Logistic Regression.
4. **LSTM wasn't a bad idea, just data-starved.** With more data (e.g., hourly instead of daily), it could close the gap.
5. **Low log-loss ≠ profitable.** Always check Sharpe ratio and drawdown alongside accuracy.

---

## 🚀 Future Improvements

- Combine all 3 models into a stacking ensemble
- Add on-chain data (whale activity, exchange flows, funding rates)
- Train LSTM on hourly data for more training sequences
- Use margin-based signals (P(BUY) − P(HOLD) ≥ δ) instead of simple thresholds
- Add a separate "market regime" detector to handle crash periods (like the 2022 bear market) better

---

## 📁 Source Notebooks

- `BTC_Direction_Classifier.ipynb` — LightGBM & Logistic Regression (core models)
- `BTC_LR_and_LGBM.ipynb` — Extended variants, split ratio tests, interactive interface
- `BTC_LSTM_Notebook.ipynb` — LSTM design, debugging, and improvement journey

---

*Submitted: May 17, 2026*
