import os

class CFG:
    # toggle btw Kaggle & local paths
    use_kaggle = False # True for Kaggle, False for local execution

    # paths:
    base_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # go up from src/
    data_directory = os.path.join(base_directory, 'data')

    if use_kaggle:
        train_path = '/kaggle/input/equity-post-HCT-survival-predictions/train.csv'
        test_path = '/kaggle/input/equity-post-HCT-survival-predictions/test.csv'
        subm_path = '/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv'
        dict_path = '/kaggle/input/equity-post-HCT-survival-predictions/data_dictionary.csv'
    else:
        train_path = os.path.join(data_directory, 'train.csv')
        test_path = os.path.join(data_directory, 'test.csv')
        subm_path = os.path.join(data_directory, 'sample_submission.csv')
        dict_path = os.path.join(data_directory, 'data_dictionary.csv')
    # general configs:
    color = '#117A65'
    batch_size = 32768 # large batch size for better CPU utilization w CoxResNet
    early_stop = 300
    penalizer = 0.01
    n_splits = 5 # CV splits

    # CoxResNet training parameters
    cox_resnet_params = {'learning_rate': 0.0001, 'weight_decay': 0.001, 'epochs': 100, 'hidden_size': 4096, 'dropout': 0.05}

    ###############################################################################################################################
    #  classification models (efs Prediction)
    ###############################################################################################################################
    # CatBoost for classification (efs) 
    ctb_class_params = {'loss_function':'Logloss',
                        'eval_metric':'AUC',
                        'task_type': 'CPU',
                        'random_seed': 42,
                        'depth':6, 
                        'learning_rate':0.02,
                        'n_estimators':2000,
                        'l2_leaf_reg':1,
                        'subsample':0.66,
                        'colsample_bylevel':0.8}

    # LightGBM for classification 
    lgb_class_params = {'objective': 'binary',
                        'min_child_samples': 75,
                        'num_iterations': 4000,
                        'learning_rate': 0.01,
                        'reg_lambda': 3.0,
                        'reg_alpha': 0.1,
                        'num_leaves': 64,
                        'metric': 'binary_logloss',
                        'max_depth': 2,
                        'device': 'gpu',  
                        'gpu_platform_id': 0,
                        'gpu_device_id': 0,
                        'max_bin': 128,
                        'verbose': -1,
                        'enable_categorical': False,  
                        'seed': 42}

    # XGBoost for classification 
    xgb_class_params = {'booster': 'gbtree',
                        'device': 'cuda',  
                        'tree_method': 'hist', 
                        'learning_rate': 0.055,
                        'gamma': 0.01,
                        'max_depth': 2,
                        'min_child_weight': 55,
                        'max_delta_step': 0.35,
                        'subsample': 1.0,
                        'colsample_bytree': 0.3,
                        'colsample_bylevel': 0.95,
                        'colsample_bynode': 0.35,
                        'lambda': 4.0,
                        'alpha': 0.0,
                        'max_bin': 255,
                        'objective': 'binary:logistic',
                        'eval_metric': 'logloss', 
                        'seed': 42}

    # HistGradientBoostingClassifier (sklearn)
    hist_class_params = {'loss': 'log_loss',
                         'learning_rate': 0.16,
                         'max_iter': 500, # was 800 before
                         'max_leaf_nodes': 64,
                         'max_depth': 3,
                         'min_samples_leaf': 48,
                         'l2_regularization': 3,
                         'max_bins': 127,
                         'interaction_cst': 'pairwise',
                         'warm_start': False,
                         'early_stopping': False,
                         'scoring': 'loss',
                         'random_state': 42}

    ###############################################################################################################################
    # regression models (efs_time prediction)
    ###############################################################################################################################
    
    # CatBoost for regression (efs_time) 
    cat_reg_params = {'loss_function': 'RMSE',
                      'eval_metric':'RMSE',
                      'iterations': 4000,
                      'depth': 8,
                      'learning_rate': 0.02,
                      'task_type': 'CPU',
                      'l2_leaf_reg': 10.0,
                      'random_seed': 42,
                      'allow_const_label': True, # constant labels to prevent degenerate solution
                      'bootstrap_type': 'Bayesian'} # CPU compatible

    # LightGBM for regression 
    lgb_reg_params = {'objective': 'regression',
                      'learning_rate': 0.02,
                      'extra_trees': True,
                      'num_leaves': 128,
                      'metric': 'mae',
                      'max_depth': 6,
                      'num_iterations': 10000,
                      'min_child_samples': 20,
                      'reg_alpha': 0.1,
                      'reg_lambda': 0.0,
                      'device': 'gpu',
                      'gpu_platform_id': 0,
                      'gpu_device_id': 0,
                      'seed': 42}

    # XGBoost for regression logloss target
    xgb_reg_log_EFSTime_params = {'booster': 'gbtree',
                                  'device': 'cuda',
                                  'tree_method': 'hist',
                                  'learning_rate': 0.035,
                                  'gamma': 0.005,
                                  'max_depth': 7,
                                  'min_child_weight': 5,
                                  'max_delta_step': 0.75,
                                  'subsample': 0.95,
                                  'colsample_bytree': 0.5,
                                  'colsample_bylevel': 0.8,
                                  'colsample_bynode': 0.95,
                                  'lambda': 5.0,
                                  'alpha': 0.0,
                                  'max_bin': 255,
                                  'objective': 'reg:squarederror',
                                  'eval_metric': 'rmse', 
                                  'seed': 42}

    # XGBoost for regression KM prob target
    xgb_reg_KM_params = {'booster': 'gbtree',
                         'device': 'cpu',
                         'nthread': 16,
                         'learning_rate': 0.035,
                         'gamma': 0.01,
                         'max_depth': 8,
                         'min_child_weight': 25,
                         'max_delta_step': 0.6,
                         'subsample': 0.85,
                         'colsample_bytree': 1,
                         'colsample_bylevel': 0.5,
                         'colsample_bynode': 0.95,
                         'lambda': 0.005,
                         'alpha': 0.,
                         'tree_method': 'hist',
                         'grow_policy': 'depthwise',
                         'max_bin': 384,
                         'objective': 'reg:squarederror',
                         'seed': 42}
    
    # HistGradientBoostingRegressor log EFS time target (sklearn)
    hist_reg_log_EFSTime_params = {'loss': 'squared_error',
                                   'learning_rate': 0.085,
                                   'max_iter': 480,
                                   'max_leaf_nodes': 64,
                                   'max_depth': 8,
                                   'min_samples_leaf': 12,
                                   'l2_regularization': 10,
                                   'max_bins': 255,
                                   'warm_start': False,
                                   'early_stopping': False,
                                   'scoring': 'loss',
                                   'random_state': 42}
    
    # HistGradientBoostingRegressor KM survival prob target (sklearn)
    hist_reg_KMSurv_params = {'loss': 'squared_error',
                              'learning_rate': 0.165,
                              'max_iter': 500,
                              'max_leaf_nodes': 48,
                              'max_depth': 7,
                              'min_samples_leaf': 12,
                              'l2_regularization': 0.05,
                              'max_bins': 255,
                              'warm_start': False,
                              'early_stopping': False,
                              'scoring': 'loss',
                              'random_state': 42}
    
    # HistGradientBoostingRegressor NA cumulative hazard target (sklearn)
    hist_reg_NACumHaz_params = {'loss': 'squared_error',
                                'learning_rate': 0.09,
                                'max_iter': 470,
                                'max_leaf_nodes': 64,
                                'max_depth': 8,
                                'min_samples_leaf': 24,
                                'l2_regularization': 5,
                                'max_bins': 255,
                                'warm_start': False,
                                'early_stopping': False,
                                'scoring': 'loss',
                                'random_state': 42}