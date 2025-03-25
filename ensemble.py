import pandas as pd
import numpy as np
import optuna
from metric import score
from config import CFG

def model_merge(y_hat_reg_tree,  y_hat_reg_km, y_hat_reg_na, y_hat_reg_rank, y_hat_reg_cox, y_hat_cls, weights):
    """
    Merge 17 models: 4 classifiers + 3 log EFS time + 3 KM survival prob + 3 NA cumulative hazard + 3 ranked EFS time + 1 CoxResNet
    using this weight strategy: a1-a4 -> 4 classifiers; b1-b3 -> log EFS time; k1-k3 -> KM; n1-n3 -> NA; r1-r3 -> ranked EFS;
    c -> overall tree regression weight factor; d -> CoxResNet weight
    """
    a1, a2, a3, a4 = weights["a1"], weights["a2"], weights["a3"], weights["a4"] # classifier weights
    b1, b2, b3 = weights["b1"], weights["b2"], weights["b3"] # log EFS time weights
    k1, k2, k3 = weights["k1"], weights["k2"], weights["k3"] # KM survival prob weights
    n1, n2, n3 = weights["n1"], weights["n2"], weights["n3"] # NA cumulative hazard weights
    r1, r2, r3 = weights["r1"], weights["r2"], weights["r3"] # ranked EFS time weights
    c = weights["c"] # overall tree regression factor
    d = weights.get("d", 0) # CoxResNet weight

    # 1. weighted classifiers: y_hat_cls is a 4-element list: [oof_preds_tree[0], oof_preds_tree[1], oof_preds_tree[2], oof_preds_tree[3]]
    # combine them with a1-a4
    y_cls_ensemble = (a1 * y_hat_cls[0] + a2 * y_hat_cls[1] + a3 * y_hat_cls[2] + a4 * y_hat_cls[3])

    # 2. weighted tree regressors: each is a 3 elt list: [CatBoost, LightGBM, HistGBM]
    y_tree_efs = [y_hat_reg_tree[0], y_hat_reg_tree[1], y_hat_reg_tree[2]] 
    # weighted sum of log EFS time
    y_hat_efs_time = b1 * y_tree_efs[0] + b2 * y_tree_efs[1] + b3 * y_tree_efs[2]

    # weighted sum of KM (flipped so aggregator doesnt have to fight the reversed signal)
    y_hat_km = - (k1 * y_hat_reg_km[0] + k2 * y_hat_reg_km[1] + k3 * y_hat_reg_km[2])

    # weighted sum of NA 
    y_hat_na = (n1 * y_hat_reg_na[0] + n2 * y_hat_reg_na[1] + n3 * y_hat_reg_na[2])

    # weighted sum of ranked EFS 
    y_hat_rank = (r1 * y_hat_reg_rank[0] + r2 * y_hat_reg_rank[1] + r3 * y_hat_reg_rank[2])

    # 3. weighted CoxResNet (flipped)
    y_hat_cox = - y_hat_reg_cox # single array, shape (N,)

    # 4. weighted contributions: apply "c" to each tree-based regression block (efs_time, km, na, rank) "d" is for CoxResNet
    y_fun_efs = (y_hat_efs_time > 0) * c * np.abs(y_hat_efs_time)
    y_fun_km  = (y_hat_km > 0) * c * np.abs(y_hat_km)
    y_fun_na  = (y_hat_na > 0) * c * np.abs(y_hat_na)
    y_fun_rank= (y_hat_rank > 0) * c * np.abs(y_hat_rank)
    y_fun_cox = (y_hat_cox > 0) * d * np.abs(y_hat_cox)

    # classifier contribution -> x_fun
    x_fun = (y_cls_ensemble > 0) * np.abs(y_cls_ensemble)

    # sum for all regressors
    total_reg = (y_fun_efs + y_fun_km + y_fun_na + y_fun_rank + y_fun_cox)

    # final merging:
    res = (1 - total_reg) * x_fun + total_reg

    print("Merged Prediction Shape Before Ranking:", res.shape)

    # return rank transform
    return pd.Series(res).rank(pct=True)


def ensemble_objective(trial, oof_preds_tree, oof_preds_cox, train_data):
    """
    Uses the following weighting: a1-a4 -> 4 classifiers;  b1-b3 -> log EFS time; k1-k3 -> KM; 
    n1-n3 -> NA; r1-r3 -> ranked EFS; c -> overall tree factor; d -> CoxResNet
    and merges them with model_merge
    """
    # classifier weights
    a1 = trial.suggest_float("a1", 0.85, 1.0) # CatBoost EFS
    a2 = trial.suggest_float("a2", 0.85, 1.0) # XGBoost EFS
    a3 = trial.suggest_float("a3", 0.85, 1.0) # LightGBM EFS
    a4 = trial.suggest_float("a4", 0.85, 1.0) # HistGBM EFS

    # log EFS time weights
    b1 = trial.suggest_float("b1", 0.05, 0.7) # CatBoost log EFS
    b2 = trial.suggest_float("b2", 0.01, 0.7) # LightGBM log EFS
    b3 = trial.suggest_float("b3", 0.1, 0.7) # HistGBM  log EFS

    # KM weights
    k1 = trial.suggest_float("k1", 0.1, 1.0) # CatBoost KM
    k2 = trial.suggest_float("k2", 0.1, 1.0) # LightGBM KM
    k3 = trial.suggest_float("k3", 0.08, 1.0) # HistGBM  KM

    # NA weights
    n1 = trial.suggest_float("n1", 0.2, 1.0) # CatBoost NA
    n2 = trial.suggest_float("n2", 0.2, 1.0) # LightGBM NA
    n3 = trial.suggest_float("n3", 0.2, 1.0) # HistGBM  NA

    # ranked EFS time weights
    r1 = trial.suggest_float("r1", 0.1, 1.0) # CatBoost EFS time ranked
    r2 = trial.suggest_float("r2", 0.1, 1.0) # LightGBM EFS time ranked
    r3 = trial.suggest_float("r3", 0.1, 1.0) # HistGBM EFS time ranked

    # overall tree factor
    c  = trial.suggest_float("c", 0.25, 0.50)
    # CoxResNet factor
    d  = trial.suggest_float("d", 0.001, 0.1)

    # get preds in the order in which they're called in notebook:
    # 0..3 = classifiers; 4..6 = log EFS time; 7..9 = KM; 10..12 = NA; 13..15 = ranked
    y_hat_cls = [oof_preds_tree[0], # CatBoost EFS
                 oof_preds_tree[1], # XGBoost EFS
                 oof_preds_tree[2], # LightGBM EFS
                 oof_preds_tree[3]] # HistGBM EFS
    
    y_hat_reg_tree = [oof_preds_tree[4], # CatBoost log EFS
                      oof_preds_tree[5], # LightGBM log EFS
                      oof_preds_tree[6]] # HistGBM  log EFS
    
    y_hat_reg_km = [oof_preds_tree[7], # CatBoost KM
                    oof_preds_tree[8], # LightGBM KM
                    oof_preds_tree[9]] # HistGBM  KM
    
    y_hat_reg_na = [oof_preds_tree[10], # CatBoost NA
                    oof_preds_tree[11], # LightGBM NA
                    oof_preds_tree[12]] # HistGBM  NA
    
    y_hat_reg_rank = [oof_preds_tree[13], # CatBoost ranked
                      oof_preds_tree[14], # LightGBM ranked
                      oof_preds_tree[15]] # HistGBM  ranked
   
    y_hat_reg_cox = np.array(oof_preds_cox).squeeze()

    # pass to model_merge
    merged_preds = model_merge(y_hat_reg_tree, y_hat_reg_km, y_hat_reg_na, y_hat_reg_rank, y_hat_reg_cox, y_hat_cls,
        {"a1": a1, "a2": a2, "a3": a3, "a4": a4, "b1": b1, "b2": b2, "b3": b3, "k1": k1, "k2": k2, "k3": k3, "n1": n1, "n2": n2, "n3": n3, "r1": r1, "r2": r2, "r3": r3, "c":  c, "d":  d})

    # eval c-index on train_data
    if isinstance(train_data, dict):
        # handle as dictionary
        print("Handling `train_data` as dict in ensemble_objective...")
        if 'ID' not in train_data:
            raise KeyError("Missing 'ID' in train_data dictionary!")
        submission_df = pd.DataFrame({'ID': train_data['ID'], 'prediction': merged_preds})
        required_keys = ['ID','efs','efs_time','race_group']
        missing_keys = [k for k in required_keys if k not in train_data]
        if missing_keys:
            raise KeyError(f"Missing keys: {missing_keys}")
        eval_df = pd.DataFrame({'ID': train_data['ID'], 'efs': train_data['efs'], 'efs_time': train_data['efs_time'], 'race_group': train_data['race_group']})
        return -score(eval_df, submission_df, 'ID')
    else:
        # handle as DataFrame
        print("Handling `train_data` as DataFrame in ensemble_objective...")
        if 'ID' in train_data.columns:
            submission = pd.DataFrame({'ID': train_data['ID'], 'prediction': merged_preds})
            return -score(train_data, submission, 'ID')
        else:
            # fallback if no ID col
            submission = pd.DataFrame({'ID': train_data.index, 'prediction': merged_preds})
            return -score(train_data, submission, 'ID')


def optimize_merge_params(oof_preds_tree, oof_preds_cox, test_preds_tree, test_preds_cox, train_data, subm_path=CFG.subm_path, output_name="submission.csv", n_trials=1000):
    """Runs Optuna for the weighting scheme, merges test preds w/ best params, and writes out final submission"""
    import optuna
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: ensemble_objective(trial, oof_preds_tree, oof_preds_cox, train_data), n_trials=n_trials)

    best_params = study.best_params
    print(f"\nBest Params: {best_params}")
    print(f"Best Strat. C-Index: {-study.best_value}")

    # make final preds on test set
    y_hat_cls = [ test_preds_tree[0], test_preds_tree[1], test_preds_tree[2], test_preds_tree[3]]
    y_hat_reg_tree = [test_preds_tree[4], test_preds_tree[5], test_preds_tree[6]]
    y_hat_reg_km = [test_preds_tree[7], test_preds_tree[8], test_preds_tree[9]]
    y_hat_reg_na = [test_preds_tree[10], test_preds_tree[11], test_preds_tree[12]]
    y_hat_reg_rank = [test_preds_tree[13], test_preds_tree[14], test_preds_tree[15]]
    y_hat_reg_cox = np.array(test_preds_cox).squeeze()

    final_preds = model_merge(y_hat_reg_tree, y_hat_reg_km, y_hat_reg_na, y_hat_reg_rank, y_hat_reg_cox, y_hat_cls, best_params)

    # make submission from sample submission 
    subm = pd.read_csv(subm_path)
    subm["prediction"] = final_preds
    subm.to_csv(output_name, index=False)
    print(f"Saved final submission to {output_name}")
    return best_params
