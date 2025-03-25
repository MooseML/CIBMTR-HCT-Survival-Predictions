import pandas as pd
import numpy as np
import torch
from catboost import CatBoostClassifier, CatBoostRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from eda import EDA
from targets import Targets
from metric import score
from config import CFG
from models import CoxResNet, CoxPHLoss, train_cox_resnet_fullbatch
from feature_engineering import FE
from sklearn.metrics import roc_auc_score, log_loss


class MD:
    def __init__(self, color, data, cat_cols, early_stop, n_splits):
        self.eda = EDA(color, data)
        self.targets = Targets(data, n_splits)
        self.data = self.targets.create_norm_targets()
        self.cat_cols = cat_cols
        self.early_stop = early_stop
        self.cv, self.strat_col = self.targets.get_stratified_cv()
        self.fe = FE()
    
    def train_tree_model(self, model_type, params, target, is_classifier, title):
        """Train tree-based models (XGBoost, LightGBM, CatBoost, HistGBM)"""
        drop_cols = ['efs', 'efs_time', 'log_efs_time', 'ranked_efs_time', 'target_norm', 
                     'km_survival_prob', 'na_cum_hazard', 'neg_exp_na_cum_hazard']
        existing_drop_cols = [col for col in drop_cols if col in self.data.columns]

        # data based on model type
        if is_classifier:
            data = self.data.copy()
        else:
            data = self.data.copy()
            data["efs_as_feat"] = data["efs"].astype(int)
        
        # ID, efs, efs_time, race_group must always be present
        for col in ['ID', 'efs', 'efs_time', 'race_group']:
            if col not in data.columns:
                data[col] = self.data[col]

        # sample weights to emphasize when events happen 
        sample_weights = np.where(data["efs"] == 1, 0.6, 0.4) if not is_classifier else None

        X = data.drop(columns=existing_drop_cols, errors='ignore').copy()
        y = data.loc[:, target].reset_index(drop=True)
        
        # remove ID column from features (not relevant)
        if 'ID' in X.columns:
            X = X.drop(columns=['ID'])
        
        # drop leakage columns
        leakage_cols = ['efs_time', 'efs'] if is_classifier else []
        X = X.drop(columns=leakage_cols, errors='ignore')
        
        oof_preds = np.zeros(len(data))
        models, fold_scores, auc_scores, logloss_scores = [], [], [], []

        for fold, (tr_idx, val_idx) in enumerate(self.cv.split(X, self.strat_col.loc[data.index])):
            X_train, X_val = X.iloc[tr_idx].copy(), X.iloc[val_idx].copy()
            y_train, y_val = y.iloc[tr_idx], y.iloc[val_idx]
            
            # remove duplicates
            X_train = X_train.loc[:, ~X_train.columns.duplicated()]
            X_val = X_val.loc[:, ~X_val.columns.duplicated()]
            
            print(f"\nFold {fold + 1}")
            
            sample_weights_train = sample_weights[tr_idx] if not is_classifier else None
            
            # small noise to numerical columns
            num_cols = X_train.select_dtypes(include=[np.number]).columns
            X_train.loc[:, num_cols] = X_train.loc[:, num_cols].astype(np.float32).fillna(0)
            X_val.loc[:, num_cols] = X_val.loc[:, num_cols].astype(np.float32).fillna(0)
            X_train.loc[:, num_cols] += np.random.normal(0, 1e-5, X_train.loc[:, num_cols].shape)
            X_val.loc[:, num_cols] += np.random.normal(0, 1e-5, X_val.loc[:, num_cols].shape)
            
            # handle categorical features properly for each model type
            if model_type == "CatBoost":
                for cat_col in self.cat_cols:
                    if cat_col in X_train.columns:
                        X_train[cat_col] = X_train[cat_col].astype(str)
                        X_val[cat_col] = X_val[cat_col].astype(str)

                # model initialization and training
                model = (CatBoostClassifier(**params, cat_features=self.cat_cols, verbose=0) if is_classifier else CatBoostRegressor(**params, cat_features=self.cat_cols, verbose=0))
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], sample_weight=sample_weights_train)
                
            elif model_type == "XGBoost":
                params["enable_categorical"] = True
                for cat_col in self.cat_cols:
                    if cat_col in X_train.columns:
                        X_train[cat_col] = X_train[cat_col].astype("category")
                        X_val[cat_col] = X_val[cat_col].astype("category")
                model = XGBClassifier(**params) if is_classifier else XGBRegressor(**params)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], sample_weight=sample_weights_train)
                
            elif model_type == "LightGBM":
                # convert categorical columns for both train and val sets consistently
                cat_indices = []
                for i, cat_col in enumerate(self.cat_cols):
                    if cat_col in X_train.columns:
                        cat_indices.append(X_train.columns.get_loc(cat_col))
                        # make sure categories are consistent btw train and val
                        all_categories = sorted(list(set(X_train[cat_col].astype(str).unique()).union(
                            set(X_val[cat_col].astype(str).unique()))))
                        
                        # use same encoding for both sets
                        X_train[cat_col] = pd.Categorical(X_train[cat_col].astype(str), categories=all_categories)
                        X_val[cat_col] = pd.Categorical(X_val[cat_col].astype(str), categories=all_categories)
                
                # init and train model
                model = LGBMClassifier(**params) if is_classifier else LGBMRegressor(**params)
                # use categorical_feature as indices not column names
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                        categorical_feature=cat_indices, 
                        sample_weight=sample_weights_train)
            
            elif model_type == "HistGBM":
                model = (HistGradientBoostingClassifier(**params) if is_classifier else HistGradientBoostingRegressor(**params))
                model.fit(X_train, y_train, sample_weight=sample_weights_train)

            # prediction step
            if model_type == "LightGBM" and is_classifier:
                # LightGBM classifier: use raw scores and convert to probability
                raw_preds = model.predict(X_val, raw_score=True)
                pred_efs_1 = 1 / (1 + np.exp(-raw_preds)) # conversion to probability
            elif is_classifier:
                pred_efs_1 = model.predict_proba(X_val)[:, 1]
            else:
                pred_efs_1 = model.predict(X_val)
                
            oof_preds[val_idx] = pred_efs_1
            
            # check for required columns before metric calculation
            required_cols = ['ID', 'efs', 'efs_time', 'race_group']
            if all(col in data.columns for col in required_cols):
                fold_true = data.iloc[val_idx][required_cols].reset_index(drop=True)
            else:
                fold_true = data.iloc[val_idx].copy()
                for col in required_cols:
                    if col not in fold_true.columns:
                        fold_true[col] = self.data.iloc[val_idx][col].values
                fold_true = fold_true[required_cols].reset_index(drop=True)

            fold_pred = pd.DataFrame({'ID': fold_true['ID'], 'prediction': pred_efs_1})
            # remove duplicate race_group columns
            if 'race_group' in fold_true.columns and isinstance(fold_true['race_group'], pd.DataFrame):
                print("Warning: Duplicate 'race_group' detected, selecting first column!")
                fold_true = fold_true.loc[:, ~fold_true.columns.duplicated()]

            # make sure race_group is a single column Series
            fold_true['race_group'] = fold_true['race_group'].squeeze()

            # remove duplicate IDs from fold_true and fold_pred
            fold_true = fold_true.drop_duplicates(subset=['ID'])
            fold_pred = fold_pred.drop_duplicates(subset=['ID'])

            if is_classifier:
                fold_auc = roc_auc_score(y_val, pred_efs_1)
                fold_logloss = log_loss(y_val, pred_efs_1)
                auc_scores.append(fold_auc)
                logloss_scores.append(fold_logloss)
                fold_scores.append(fold_auc)
                print(f"Fold {fold + 1} ROC AUC: {fold_auc:.4f}, Log Loss: {fold_logloss:.4f}")
            else:
                fold_score = score(fold_true, fold_pred, 'ID')
                fold_scores.append(fold_score)
                print(f"Fold {fold + 1} Stratified C-Index: {fold_score:.4f}")

            models.append(model)

        # plot CV scores
        if fold_scores:
            if is_classifier:
                self.eda.plot_roc_auc_logloss(auc_scores, logloss_scores, title=title)
            else:
                self.eda.plot_cv(fold_scores, title=title)

        return models, oof_preds
  

    def infer_tree_model(self, data, models, is_classifier):
        """Inference for tree models with automatic feature alignment"""
        X_test = data.copy() 

        if not is_classifier:
            X_test["efs_as_feat"] = 1 # force efs=1 for test rows

        first_model = models[0]
        model_type = first_model.__class__.__name__
        
        # prep categorical features based on model type
        if "LGBMClassifier" in model_type or "LGBMRegressor" in model_type:
            # check categories match what was used in training
            for cat_col in self.cat_cols:
                if cat_col in X_test.columns:
                    # convert to the same category type w the same categories
                    X_test[cat_col] = X_test[cat_col].astype(str).astype('category')
        
        # identify model features dynamically
        if hasattr(first_model, 'get_booster'): # XGBoost
            model_features = first_model.get_booster().feature_names
        elif hasattr(first_model, 'feature_names_'): # LightGBM
            model_features = first_model.feature_names_
        elif hasattr(first_model, 'feature_names_in_'): # HistGBM 
            model_features = first_model.feature_names_in_
        else:  # CatBoost or other models
            model_features = X_test.columns.tolist()

        # check for missing and extra features
        missing_features = [col for col in model_features if col not in X_test.columns]
        extra_features = [col for col in X_test.columns if col not in model_features]

        if missing_features:
            print(f"missing features in test data: {missing_features}")
        if extra_features:
            print(f"extra features in test data (will be dropped): {extra_features}")

        # handle missing columns
        for col in missing_features:
            X_test[col] = 0  # add missing features with placeholder values
        
        # select only the columns needed for pred
        X_test = X_test[model_features]

        if "LGBMClassifier" in model_type and is_classifier:
            # use predict method directly and specify that raw scores
            raw_preds = first_model.predict(X_test, raw_score=True)
            # change raw scores to probabilities
            pred_probs_efs_1 = 1 / (1 + np.exp(-raw_preds))
            return pred_probs_efs_1
        # standard prediction
        elif is_classifier:
            return first_model.predict_proba(X_test)[:, 1]
        else:
            return first_model.predict(X_test)
    

    def train_cox_resnet(self):
        """Train CoxResNet with stratified CV using race & efs, return OOF predictions"""
        data_cox = self.data.copy()
        data_cox["efs_as_feat"] = data_cox["efs"].astype(int)
        # select training data and drop non-feature columns
        drop_cols = ['efs', 'efs_time', 'log_efs_time', 'ranked_efs_time', 'target_norm', 
                    'km_survival_prob', 'na_cum_hazard', 'neg_exp_na_cum_hazard', 'ID', 'race_group']
        
        X = data_cox.drop(columns=drop_cols, errors='ignore').copy()
        y_log_efs_time = np.log1p(data_cox['efs_time']) # log transform survival time
        y_event = data_cox['efs'] # event indicator

        # store OOF preds for the full dataset (initialize with zeros)
        oof_preds = np.zeros(len(self.data))  
        models, fold_scores = [], []

        for fold, (train_idx, val_idx) in enumerate(self.cv.split(X, self.strat_col.loc[data_cox.index])):
            print(f"Training Fold {fold + 1}")

            # train/val split 
            X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
            y_train_log = y_log_efs_time.iloc[train_idx]
            y_train_event = y_event.iloc[train_idx]
            sample_weights = np.where(y_train_event == 1, 0.6, 0.4)

            X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32, device='cuda')
            y_train_log_tensor = torch.tensor(y_train_log.values, dtype=torch.float32, device='cuda')
            y_train_event_tensor = torch.tensor(y_train_event.values, dtype=torch.float32, device='cuda')
            sample_weights_tensor = torch.tensor(sample_weights, dtype=torch.float32, device='cuda')

            # init model
            model = CoxResNet(input_size=X_train.shape[1], hidden_size=4096, dropout=0.05).cuda()
            optimizer = torch.optim.RAdam(model.parameters(), lr=CFG.cox_resnet_params['learning_rate'])
            criterion = CoxPHLoss()

            # train the model
            train_cox_resnet_fullbatch(model, X_train_tensor, y_train_log_tensor, y_train_event_tensor,
                                       optimizer, criterion, epochs=CFG.cox_resnet_params['epochs'],
                                       device='cuda', sample_weights=sample_weights_tensor)

            models.append(model)  
            # OOF preds
            X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32, device='cuda')
            model.eval()
            with torch.no_grad():
                oof_preds[val_idx] = model(X_val_tensor).cpu().numpy().squeeze()
            
            # race_group needs to be a single column before scoring
            fold_true = data_cox.iloc[val_idx].copy()

            # remove duplicate race_group columns if necessary
            if 'race_group' in fold_true.columns and isinstance(fold_true['race_group'], pd.DataFrame):
                print("Warning: Duplicate 'race_group' detected, selecting first column!")
                fold_true = fold_true.loc[:, ~fold_true.columns.duplicated()]

            # enforce race_group to a single column series
            fold_true['race_group'] = fold_true['race_group'].squeeze()

            # eval performance (like tree model)
            fold_score = score(fold_true, pd.DataFrame({'ID': fold_true['ID'], 'prediction': oof_preds[val_idx]}), 'ID')
            fold_scores.append(fold_score)
            print(f"Fold {fold + 1} Stratified C-Index: {fold_score:.4f}")

        self.eda.plot_cv(fold_scores, title="CoxResNet CV Results")
        print("\nFinal OOF Predictions:", oof_preds)

        return models, oof_preds
    
    def infer_cox_resnet(self, models, test_data, device='cuda'):
        """Inference for CoxResNet with feature alignment before prediction"""
        test_data_cox = test_data.copy()
        test_data_cox["efs_as_feat"] = 1 # force efs to 1 for inference
        # test_data to tensor format
        if 'race_group' in test_data_cox.columns:
            print("converting race_group to int before inference")
            test_data_cox['race_group'] = test_data_cox['race_group'].astype(int)
 
        test_tensor = torch.tensor(test_data_cox.values, dtype=torch.float32, device=device)

        # init test_preds with zeros
        test_preds = np.zeros(len(test_data_cox))

        # feature names from the trained models
        model_features = test_data_cox.drop(columns=['efs', 'efs_time', 'log_efs_time', 'ranked_efs_time', 
                                                     'target_norm', 'km_survival_prob', 'na_cum_hazard', 
                                                     'neg_exp_na_cum_hazard', 'ID', 'race_group'], errors='ignore').columns.tolist()

        test_features = test_data_cox.columns.tolist() # get feature names from test data

        # check for missing and extra features
        missing_features = [col for col in model_features if col not in test_features]
        extra_features = [col for col in test_features if col not in model_features]

        if missing_features:
            print(f"missing features in test data: {missing_features}")
        if extra_features:
            print(f"extra features in test data (will be dropped): {extra_features}")

        # handle missing columns
        for col in missing_features:
            test_data_cox[col] = 0 # add missing features w zero
        test_data_cox = test_data_cox[model_features] # enforce col order matches the model
        test_tensor = torch.tensor(test_data_cox.values, dtype=torch.float32, device=device) # change updated test data to tensor

        # make preds using averaging
        for model in models:
            model.eval()
            with torch.no_grad():
                test_preds += model(test_tensor).cpu().numpy().squeeze() / len(models)

        print("\nFinal Test Predictions:", test_preds)
        return test_preds


def train_and_predict(self, model_configs, test_data):
    models, oof_preds, test_preds = [], [], []

    for config in model_configs:
        model_type = config["model_type"]

        if model_type == "CoxResNet":
            trained_models, oof_pred = self.train_cox_resnet() # returns only models and OOF preds
            test_pred = self.infer_cox_resnet(trained_models, test_data) # infer on test separately
        else:
            if "params" not in config:
                raise ValueError(f"missing params in model config: {config}")

            trained_model, oof_pred = self.train_tree_model(**config)
            test_pred = self.infer_tree_model(test_data, trained_model, config["is_classifier"])

        models.append(trained_models if model_type == "CoxResNet" else trained_model)
        oof_preds.append(oof_pred)
        test_preds.append(test_pred)

    return models, oof_preds, test_preds



