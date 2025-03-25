import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from lifelines import KaplanMeierFitter, NelsonAalenFitter
from src.metric import score  

class Targets:
    def __init__(self, data, n_splits=5):
        """
        Initialize target generation with EFS-time ranking, Kaplan-Meier survival probabilities,
        and Nelson-Aalen cumulative hazard.
        """
        self.data = data.copy()
        self.n_splits = n_splits

    def create_norm_targets(self):
        """Creates the Log EFS Time, Kaplan-Meier, and Nelson-Aalen based targets"""
        df = self.data.copy()

        # 1. log EFS time 
        df['log_efs_time'] = np.log1p(df['efs_time'])

        # 2. Kaplan-Meier survival probability
        kmf = KaplanMeierFitter()
        kmf.fit(df['efs_time'], df['efs'])
        df['km_survival_prob'] = kmf.survival_function_at_times(df['efs_time']).values
        # normalize Kaplan-Meier probability
        df['km_survival_prob'] = (df['km_survival_prob'] - df['km_survival_prob'].min()) / (df['km_survival_prob'].max() - df['km_survival_prob'].min())

        # 3. Nelson-Aalen cumulative hazard
        naf = NelsonAalenFitter()
        naf.fit(df['efs_time'], df['efs'])
        df['na_cum_hazard'] = naf.cumulative_hazard_at_times(df['efs_time']).values
        # transform Nelson-Aalen to neg exponent
        df['neg_exp_na_cum_hazard'] = np.exp(-df['na_cum_hazard'])

        # 4. ranked normalization for efs_time
        df['ranked_efs_time'] = df['efs_time'].rank(pct=True, method="first")

        # 5. final normalized target
        df['target_norm'] = np.where(df['efs'] == 1, df['log_efs_time'], 1.45)
        df['target_norm'] += np.random.normal(0, 1e-6, size=len(df)) # tiny noise to stop LightGBM from dropping 
        return df
    
    def get_stratified_cv(self):
        """Creates Stratified KFold CV Splits based on efs & race_group"""
        race_group_str = self.data['race_group'].astype(str).iloc[:, 0].str.strip() if isinstance(self.data['race_group'], pd.DataFrame) else self.data['race_group'].astype(str).str.strip()
        efs_str = self.data['efs'].astype(str).iloc[:, 0].str.strip() if isinstance(self.data['efs'], pd.DataFrame) else self.data['efs'].astype(str).str.strip()
        strat_col = race_group_str + "_" + efs_str
        cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        return cv, strat_col

    def validate_model(self, preds, title):
        """Validate model predictions using Concordance Index"""
        y_true = self.data[['ID', 'efs', 'efs_time', 'race_group']].copy()
        y_pred = pd.DataFrame({'ID': self.data['ID'], 'prediction': preds})
        c_index = score(y_true, y_pred, 'ID')
        print(f'Stratified C-Index for {title}: {c_index:.4f}')
