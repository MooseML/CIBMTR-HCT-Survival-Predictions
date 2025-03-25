# 🧬 CIBMTR - Equity in Post-HCT Survival Predictions

This repo contains my full solution for the [CIBMTR Kaggle Competition](https://www.kaggle.com/competitions/equity-post-HCT-survival-predictions), which focuses on predicting equitable survival rates for allogeneic Hematopoietic Cell Transplantation (HCT) patients.

Final Private Leaderboard Score: **0.69261**  
Rank: **773 / 3327**  
Competition Timeline: Dec 4, 2024 – Mar 5, 2025

---

## High-Level Strategy

1. **Feature Engineering**: Preprocess and transform input features.
2. **Stage 1 - Classification**: Predict event probability (efs = 0 or 1).
3. **Stage 2 - Regression**: Predict time-to-event using models like Cox, KM, or direct regression.
4. **Aggregation**: Combine predictions and optimize final output with Optuna.

---

## Scoring Metric

The competition uses the **Stratified Concordance Index (C-index)**, which evaluates model accuracy and fairness across race groups:

```python
import pandas as pd
import pandas.api.types
import numpy as np
from lifelines.utils import concordance_index

class ParticipantVisibleError(Exception):
    pass

def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    """
    >>> import pandas as pd
    >>> row_id_column_name = "id"
    >>> y_pred = {'prediction': {0: 1.0, 1: 0.0, 2: 1.0}}
    >>> y_pred = pd.DataFrame(y_pred)
    >>> y_pred.insert(0, row_id_column_name, range(len(y_pred)))
    >>> y_true = { 'efs': {0: 1.0, 1: 0.0, 2: 0.0}, 'efs_time': {0: 25.1234,1: 250.1234,2: 2500.1234}, 'race_group': {0: 'race_group_1', 1: 'race_group_1', 2: 'race_group_1'}}
    >>> y_true = pd.DataFrame(y_true)
    >>> y_true.insert(0, row_id_column_name, range(len(y_true)))
    >>> score(y_true.copy(), y_pred.copy(), row_id_column_name)
    0.75
    """
    
    del solution[row_id_column_name]
    del submission[row_id_column_name]
    
    event_label = 'efs'
    interval_label = 'efs_time'
    prediction_label = 'prediction'
    for col in submission.columns:
        if not pandas.api.types.is_numeric_dtype(submission[col]):
            raise ParticipantVisibleError(f'Submission column {col} must be a number')
    # Merging solution and submission dfs on ID
    merged_df = pd.concat([solution, submission], axis=1)
    merged_df.reset_index(inplace=True)

    merged_df_race_dict = dict(merged_df.groupby(['race_group'], observed=True).groups)
    metric_list = []
    for race in merged_df_race_dict.keys():
        # Retrieving values from y_test based on index
        indices = sorted(merged_df_race_dict[race])
        merged_df_race = merged_df.iloc[indices]
        # Calculate the concordance index
        c_index_race = concordance_index(
                        merged_df_race[interval_label],
                        -merged_df_race[prediction_label], 
                        merged_df_race[event_label])
        metric_list.append(c_index_race)
    return float(np.mean(metric_list)-np.sqrt(np.var(metric_list)))
```
Larger risk scores imply higher hazard and earlier events.

## How to Run

# Clone the repository
git clone https://github.com/MooseML/CIBMTR-HCT-Survival-Predictions.git
cd CIBMTR-HCT-Survival-Predictions

# (Optional) Create a virtual environment
python -m venv .venv
.\.venv\Scripts\activate      # or: source .venv/bin/activate

# Install requirements
pip install -r requirements.txt

### 3. Generate submission

Output file format:

```
ID,prediction
28800,0.57
28801,0.89
...
```

---

## Repo Structure

```
data/            ← Training and test data (not in Git)
notebooks/       ← Jupyter Notebooks
src/             ← Source code (models, metrics, training, etc.)
requirements.txt
README.md
submission.csv   
```

---

## Competition Summary

Participants must build models that are not only accurate but fair across racial demographics. Predictions are evaluated using a race-stratified C-index to ensure equity in survival modeling.

Your models will help make medical AI systems more fair and robust for transplant patients from all backgrounds.
