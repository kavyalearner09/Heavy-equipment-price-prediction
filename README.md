# Heavy Equipment Price Prediction

A regression pipeline that predicts the resale price of used heavy construction equipment (bulldozers, excavators, loaders, etc.) from historical auction data. Built as my Machine Learning Practice (MLP) capstone project for the BS Degree in Data Science, IIT Madras.

**Final result:** ~0.199 RMSLE on the leaderboard, clearing the 0.20 evaluation cutoff, using a blended XGBoost/LightGBM ensemble.

---

## Problem Statement

Given historical auction records of heavy equipment sales (machine specs, sale date, usage hours, configuration, etc.), predict the sale price of each unit. The target is evaluated using **RMSLE (Root Mean Squared Log Error)**, which penalizes relative rather than absolute error — appropriate here since equipment prices span a very wide range (a few thousand dollars to hundreds of thousands).

## Dataset

- Historical auction sale records with machine identifiers, sale dates, equipment configuration fields, and usage metrics
- Target variable: sale price (log-transformed for training, per RMSLE's requirements)

## Approach

### 1. Exploratory Data Analysis
- Examined the target variable's distribution and skew, which motivated the log-transform
- Detected and handled bad/implausible values in `ManufactureYear` (e.g. placeholder years like 1000)
- Ran bivariate scatter plots to understand relationships between key numeric features and price

### 2. Feature Engineering
- `AssetAge` — derived from sale date minus manufacture year
- `TransactionQuarter` — extracted from the sale date to capture seasonality
- `DescriptorLength` — text-length feature from the equipment description field
- Binary missingness flags for columns with informative "missing-ness" (i.e. where the fact that a value is missing is itself predictive)

### 3. Preprocessing
- Compared **OrdinalEncoder** (RMSLE 0.2053) vs. **OneHotEncoder** (RMSLE 0.2338) for categorical features — OrdinalEncoder performed better here, likely because many categorical fields had a natural or high-cardinality structure that one-hot encoding fragmented

### 4. Modeling
Trained and compared three gradient-boosting models on the same preprocessing pipeline:
- **LightGBM**
- **XGBoost**
- **CatBoost**

XGBoost and LightGBM were further tuned using **RandomizedSearchCV**. The final submission blends the two tuned models:

```
final_prediction = 0.63 * XGBoost_prediction + 0.37 * LightGBM_prediction
```

### 5. Debugging Along the Way
A few real issues caught and fixed during development (documented here because debugging is half of the job):
- A double log-transform bug that silently inflated predictions
- `DummyRegressor` accidentally overwriting the real model's submission during pipeline testing
- Pipeline syntax errors and inconsistent `verbosity` parameter conventions across LightGBM, XGBoost, and CatBoost

## Results

| Model | Validation RMSE |
|---|---|
| XGBoost | 0.232410 |
| LightGBM | 0.233857 |
| CatBoost | 0.253665 |
| **Blended (0.63 XGB / 0.37 LGBM)** | **~0.199 (leaderboard RMSLE)** |

## Tech Stack

- **Language:** Python
- **Core libraries:** pandas, NumPy, scikit-learn
- **Models:** LightGBM, XGBoost, CatBoost
- **Environment:** Kaggle Notebooks

## Repository Structure

```
.
├── heavy_equipment_price_prediction.ipynb   # Main notebook: EDA, feature engineering, modeling
├── requirements.txt                          # Python dependencies
└── README.md
```

## How to Run

1. Clone this repo:
   ```
   git clone https://github.com/kavyalearner09/Heavy-equipment-price-prediction.git
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Open `heavy_equipment_price_prediction.ipynb` in Jupyter or VS Code and run all cells.

> Note: the original dataset was accessed via Kaggle's competition data. You'll need to download it from the competition page and place it in the expected input path referenced in the notebook.

## Key Learnings

- How target skew and evaluation metric (RMSLE) should directly shape preprocessing choices (log-transform) rather than being an afterthought
- Encoding strategy isn't one-size-fits-all — OrdinalEncoder beat OneHotEncoder here, which isn't the "default" assumption most tutorials lead with
- Blending multiple models, even a simple weighted average, gave a measurable improvement over any single model
- The value of systematic debugging — several silent bugs (double log-transform, an overwritten submission) would have gone unnoticed without deliberate validation checks.

---

*Built as part of the BS Degree in Data Science, IIT Madras — Machine Learning Practice (MLP) capstone.*
