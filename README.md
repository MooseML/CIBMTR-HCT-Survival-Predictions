# 🧬 CIBMTR - Equity in Post-HCT Survival Predictions

This repo contains my full solution for the [CIBMTR Kaggle Competition](https://www.kaggle.com/competitions/equity-post-HCT-survival-predictions), which focuses on predicting equitable survival rates for allogeneic Hematopoietic Cell Transplantation (HCT) patients.

📈 Final Private Leaderboard Score: **0.69261**  
🏅 Rank: **773 / 3327**  
📅 Competition Timeline: Dec 4, 2024 – Mar 5, 2025

---

## Local Environment Setup (GPU)

This solution is designed to run **locally** on a GPU-enabled system with the following specs:

```
Python 3.8.10
CUDA 11.8
NVIDIA RTX 3070 Ti
PyTorch 2.4.1 + cu118
```

## How to Run

# Clone the repository
```
git clone https://github.com/MooseML/CIBMTR-HCT-Survival-Predictions.git
cd CIBMTR-HCT-Survival-Predictions
```
# (Optional) Create a virtual environment
```
python -m venv .venv
.\.venv\Scripts\activate      # or: source .venv/bin/activate
```
# Install requirements
```
pip install -r requirements.txt
```
### 3. Generate submission

Output file format:

```
data/            # CSV files 
notebooks/       # Jupyter notebooks (main pipeline in here)
src/             # All Python source code (config, models, training, etc.)
    ├── config.py
    ├── train.py
    ├── feature_engineering.py
    ├── ensemble.py
    └── ...
requirements.txt
README.md
submission.csv   # final submission format
```

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/MooseML/CIBMTR-HCT-Survival-Predictions.git
cd CIBMTR-HCT-Survival-Predictions
```

### 2. Install requirements

Use the conda instructions above to ensure PyTorch and GPU support.

### 3. Run the pipeline

The pipeline is executed through the notebook:

```python
notebooks/cibmtr-equitable-post-hct-survival-predictions.ipynb
```

You can run the full pipeline there, including feature engineering, model training, ensembling, and submission generation.

---

## Notes on Kaggle Execution

If you want to run this **on Kaggle**, do the following:

1. **Upload the entire `src/` folder** as part of a custom Kaggle Dataset (e.g. `woohoo`).
2. In the notebook, prepend the dataset path to `sys.path`:
   ```python
   import sys
   sys.path.append("/kaggle/input/woohoo")
   ```

3. Then change imports to remove the `src.` prefix:
   ```python
   from config import CFG
   from feature_engineering import FE
   from train import MD, train_and_predict
   from ensemble import optimize_merge_params
   ```

Alternatively, if you leave `src.` in the imports, ensure your dataset includes the full `src/` structure and use:

```python
sys.path.append("/kaggle/input/woohoo/src")
```

---

## Strategy Overview

### 1. **Feature Engineering**
- Ordinal encodes categorical variables for tree models
- OneHot encodes categorical variables for NN based models
- Scales continuous features

### 2. **Stage 1 - Classification**
- Binary model: predicts `efs = 0/1`
- Handles censoring with stratified sampling
- Outputs risk probabilities

### 3. **Stage 2 - Regression**
- Models time-to-event using:
  - Log transformed efs_time 
  - Kaplan-Meier survival probability
  - Nelson-Aalen cumulative hazard
  - Ranked normalization for efs_time

### 4. **Merging / Aggregation**
- Combines classification and regression outputs
- Optimizes final risk score using `optuna` on stratified C-index

---

## Evaluation Metric: Stratified Concordance Index (C-index)

The competition uses a **race-stratified concordance index**, which:
- Measures the ability to rank survival times correctly
- Penalizes models that are unfair across racial groups
- Rewards models that perform **consistently** across all subgroups

Predictions must be risk scores: **higher = higher risk = shorter survival**.

### Why Predictions Are Negated (`-prediction`)

The regression models output **survival time estimates**, meaning larger = longer survival. But the competition scoring expects **larger = riskier**, so we apply `-prediction` before scoring.

If you don’t do this, your C-index will look inverted or broken — especially in cross-validation plots.

**It's not a bug**, just a sign flip, so be consistent.

---

## Submission Format

Your final CSV should look like:

```
id,prediction
28800,0.57
28801,0.89
...
```

Provided is a sample submission in `submission.csv`.

---

## Final Notes

- This repo is meant to be **educational**: feel free to reuse/adapt for future survival modeling work.
- For best results, run locally on GPU with the provided environment.
- Reach out or open an issue if you need help replicating the results!
