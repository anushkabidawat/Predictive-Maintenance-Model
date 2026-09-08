# Machine Learning for Industrial Predictive Maintenance

Predicting machine failures before they happen, using sensor telemetry from an industrial milling process — so maintenance can be scheduled proactively instead of reactively.

## Problem Statement

Unplanned machine failures cost manufacturers downtime and money. This project builds a classification model that predicts machine failure in advance, using real-time sensor readings (temperature, rotational speed, torque, tool wear), so that maintenance teams can intervene before a breakdown occurs rather than after.

## Dataset

**AI4I 2020 Predictive Maintenance Dataset** — 10,000 records simulating a real industrial milling machine, with the following raw features:

| Feature | Description |
|---|---|
| Type | Product quality variant (Low / Medium / High) |
| Air temperature [K] | Ambient air temperature |
| Process temperature [K] | Process temperature |
| Rotational speed [rpm] | Spindle rotational speed |
| Torque [Nm] | Spindle torque |
| Tool wear [min] | Cumulative tool wear time |
| Machine failure | Target label (0 = Normal, 1 = Failure) |

Source: [UCI Machine Learning Repository — AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)

> The dataset is highly imbalanced: only **3.39%** of records are failure cases, which directly shapes the modeling approach below.

## Project Workflow

### 1. Exploratory Data Analysis
- Distribution analysis of spindle torque across normal vs. failure states, revealing that failures cluster at both extremes (very low torque → mechanical breakdown, very high torque → tool jam/seizure) rather than at a single threshold.
- Correlation analysis exposing strong collinearity between Air and Process temperature (0.88) and a strong inverse relationship between Rotational speed and Torque (-0.88).
- Class imbalance quantified: 96.61% Normal vs. 3.39% Failure.

### 2. Feature Engineering
Rather than passing raw, collinear sensor readings directly into the model, three domain-driven features were engineered:
- **Delta_T** — Process temperature − Air temperature (isolates thermal stress, removing the redundancy between the two raw temperature readings)
- **Power** — Torque × angular velocity (captures mechanical load directly)
- **Overstrain** — Tool wear × Torque (captures cumulative mechanical stress on a worn tool)

Columns that would cause data leakage (failure-mode indicator flags such as TWF, HDF, PWF, OSF, RNF, which are only known *after* a failure occurs) were excluded from the feature set.

### 3. Preprocessing
- Stratified 80/20 train-test split to preserve the failure class ratio in both sets.
- Feature scaling via `StandardScaler`.
- 2D PCA visualization to inspect class separability after scaling.

### 4. Modeling
Given the severe class imbalance, **accuracy was treated as a misleading metric** and **recall** (the ability to catch actual failures) was used as the primary evaluation criterion — in this domain, missing a real failure is far costlier than a false alarm.

| Model | Accuracy (%) | Recall (%) |
|---|---|---|
| Logistic Regression | 96.75 | 17.65 |
| Random Forest | 99.00 | 73.53 |
| Tuned Random Forest (GridSearchCV) | 99.20 | 79.41 |
| XGBoost (class-weighted) | 99.10 | 82.35 |
| **Tuned XGBoost (GridSearchCV)** | **93.70** | **92.65** |

The final model was selected by optimizing hyperparameters against recall via `GridSearchCV`, deliberately trading some accuracy for substantially better failure detection.

### 5. Feature Importance
Torque, Rotational Speed, and Tool Wear emerged as the top three predictors, together accounting for roughly 70% of the model's predictive power. Notably, the **engineered features (Power, Delta_T) outperformed the raw temperature readings**, confirming that domain-driven feature engineering added genuine predictive signal rather than just extra columns.

## Key Results

The final tuned XGBoost model catches **~93 out of every 100 real machine failures** (92.65% recall) at 93.70% accuracy. This comes with a deliberate tradeoff of roughly 121 false alarms per 2,000 machines (~6.3% of normal machines flagged for unnecessary inspection). In an industrial maintenance context, this tradeoff is intentional: an unnecessary inspection is inexpensive relative to the cost of an unplanned breakdown.

## Tech Stack

- Python
- pandas, numpy
- scikit-learn (Logistic Regression, Random Forest, GridSearchCV, PCA, StandardScaler)
- XGBoost
- matplotlib, seaborn

## Project Structure

```
├── Resumeproject.ipynb    # Full analysis: EDA → feature engineering → modeling → evaluation
├── ai4i2020.csv            # Dataset (AI4I 2020, UCI ML Repository)
├── requirements.txt        # Python dependencies
└── README.md
```

## How to Run

```bash
git clone https://github.com/anushkabidawat/Predictive-Maintenance-Model.git
cd Predictive-Maintenance-Model
pip install -r requirements.txt
jupyter notebook Resumeproject.ipynb
```

Run all cells top to bottom — the notebook is self-contained and reproducible.

## Conclusion

This project demonstrates an end-to-end predictive maintenance workflow: identifying and correcting for class imbalance, engineering physically meaningful features from raw sensor data, comparing multiple model families, and tuning toward the metric that actually matters for the business problem (recall) rather than the metric that looks best on paper (accuracy). The result is a model that reliably flags at-risk machines while keeping the false-alarm rate manageable.

## License

This project is licensed under the MIT License.
