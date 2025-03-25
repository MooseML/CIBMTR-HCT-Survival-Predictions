import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from torch.utils.data import DataLoader
from pathlib import Path
from config import CFG

class FE:
    def __init__(self, batch_size=32768, transformer_directory="transformers/"):
        self.batch_size = batch_size
        self.scaler = StandardScaler()
        self.transformer_directory = Path(transformer_directory)
        self.transformer_directory.mkdir(parents=True, exist_ok=True)
        
    def add_derived_features(self, df):
        df['age_karnofsky'] = df['age_at_hct'] * df['karnofsky_score']
        df['donor_recipient_age_diff'] = abs(df['donor_age'] - df['age_at_hct'])
        df['age_comorbidity'] = df['age_at_hct'] * df['comorbidity_score']
        df['donor_age_comorbidity'] = df['donor_age'] * df['comorbidity_score']
        return df

    def prepare_targets(self, df, model_type="tree", train=True):
        if train:
            # log transform of survival time
            df['log_efs_time'] = np.log1p(df['efs_time'])

            # ranked survival time
            df['ranked_efs_time'] = df.groupby('efs')['efs_time'].rank(pct=True, method="first")

            # Kaplan-Meier survival probability (normalized)
            df['km_survival_prob'] = 1 - (df['efs_time'] / df['efs_time'].max())

            # Nelson-Aalen cumulative hazard w/ 
            df['na_cum_hazard'] = -np.exp(-(df['efs_time'] / df['efs_time'].max()))
            df['neg_exp_na_cum_hazard'] = np.exp(-df['na_cum_hazard'])

        # normalized target w noise to avoid issues in tree models
        df['target_norm'] = np.where(df.get('efs', 0) == 1, df.get('log_efs_time', 0), 1.35)
        df['target_norm'] += np.random.normal(0, 1e-6, size=len(df))

        return df


    def feature_engineering_pipeline(self, data, model_type='tree', train=True):
        if isinstance(data, str):
            df = pd.read_csv(data)
        else:
            df = data.copy()

        num_cols = [
            'hla_high_res_8', 'hla_low_res_8', 'hla_high_res_6', 'hla_low_res_6',
            'hla_high_res_10', 'hla_low_res_10', 'hla_match_dqb1_high',
            'hla_match_dqb1_low', 'hla_match_drb1_high', 'hla_match_drb1_low',
            'hla_nmdp_6', 'year_hct', 'hla_match_a_high', 'hla_match_a_low',
            'hla_match_b_high', 'hla_match_b_low', 'hla_match_c_high',
            'hla_match_c_low', 'donor_age', 'age_at_hct', 'comorbidity_score',
            'karnofsky_score', 'efs', 'efs_time'
        ]
        
        feature_cat = set(df.select_dtypes(include=['object', 'category', 'string']).columns)

        for col in df.columns:
            if col in num_cols:
                df[col] = df[col].fillna(df[col].median()).astype(np.float32)  
            elif col in feature_cat:
                df[col] = df[col].fillna('Unknown').astype(str)

        if 'ID' in df.columns:
            df['ID'] = df['ID'].astype(np.int32)

        # df = self.add_derived_features(df)

        # process targets before encoding to avoid feature misalignment
        df = self.prepare_targets(df, model_type, train)

        # categorical encoding
        feature_cat = df.select_dtypes(include=['object', 'string']).columns.tolist()
        if model_type == "tree":
            encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            if train:
                df[feature_cat] = encoder.fit_transform(df[feature_cat]).astype(int)
                with open(self.transformer_directory / "ordinal_encoder.pkl", "wb") as f:
                    pickle.dump(encoder, f)
            else:
                with open(self.transformer_directory / "ordinal_encoder.pkl", "rb") as f:
                    encoder = pickle.load(f)
                df[feature_cat] = encoder.transform(df[feature_cat]).astype(int)

            # convert categorical features back to category dtype for LightGBM
            for col in feature_cat:
                df[col] = df[col].astype("category")  

        elif model_type == "coxresnet":
            if 'race_group' in feature_cat:
                feature_cat.remove('race_group')
            encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            if train:
                encoded = encoder.fit_transform(df[feature_cat])
                with open(self.transformer_directory / "onehot_encoder.pkl", "wb") as f:
                    pickle.dump(encoder, f)
            else:
                with open(self.transformer_directory / "onehot_encoder.pkl", "rb") as f:
                    encoder = pickle.load(f)
                encoded = encoder.transform(df[feature_cat])

            encoded_cols = encoder.get_feature_names_out(feature_cat)
            encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=df.index)
            # df = df.drop(columns=feature_cat)
            df = pd.concat([df, encoded_df], axis=1)
            # convert og categorical columns to integer 
            for col in feature_cat:
                df[col] = df[col].astype('category').cat.codes
            
            if 'race_group' in df.columns:
                df['race_group'] = df['race_group'].astype('category').cat.codes # convert to numeric

        # normalize 
        if model_type == "coxresnet":
            num_cols_cox = [
            'hla_high_res_8', 'hla_low_res_8', 'hla_high_res_6', 'hla_low_res_6',
            'hla_high_res_10', 'hla_low_res_10', 'hla_match_dqb1_high',
            'hla_match_dqb1_low', 'hla_match_drb1_high', 'hla_match_drb1_low',
            'hla_nmdp_6', 'year_hct', 'hla_match_a_high', 'hla_match_a_low',
            'hla_match_b_high', 'hla_match_b_low', 'hla_match_c_high',
            'hla_match_c_low', 'donor_age', 'age_at_hct', 'comorbidity_score',
            'karnofsky_score', 'efs', 'efs_time'
        ]
       
            num_cols_cox = [col for col in num_cols_cox if col not in ('efs', 'efs_time', 'ID', 'race_group', 'km_survival_prob', 'log_efs_time', 'na_cum_hazard', 'neg_exp_na_cum_hazard', 'ranked_efs_time')]

            scaler_path = self.transformer_directory / "scaler.pkl"
            if train:
                df[num_cols_cox] = self.scaler.fit_transform(df[num_cols_cox])
                with open(scaler_path, "wb") as f:
                    pickle.dump(self.scaler, f)
            else:
                with open(scaler_path, "rb") as f:
                    self.scaler = pickle.load(f)
                df[num_cols_cox] = self.scaler.transform(df[num_cols_cox])
        
        # final dataframe ready
        output_dict = {'X': df.drop(columns=['efs', 'efs_time'], errors='ignore'), 'efs': df.get('efs'), 'efs_time': df.get('efs_time'), 'ID': df['ID'], 'race_group': df['race_group'], 'feature_cat': feature_cat}

        return output_dict

