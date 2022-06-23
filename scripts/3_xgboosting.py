import os
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from xgboost import plot_importance
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, roc_curve, auc
from scipy import interp
from matplotlib import pyplot as plt
from xgboost import plot_importance

path_to_data = '/workspace/data/'
path_to_figures = '/workspace/figures/'


def k_fold_cross_validation(K, model, x_df, y_df):
    eval_size = int(np.round(1./K))
    skf = StratifiedKFold(n_splits=K)

    fig = plt.figure(figsize=(7,7))
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    lw = 2
    i = 0
    roc_aucs = []
    for train_indices, test_indices in skf.split(x_df, y_df['label']):
        X_train, y_train = x_df.iloc[train_indices], y_df['label'].iloc[train_indices]
        X_valid, y_valid = x_df.iloc[test_indices], y_df['label'].iloc[test_indices]
        class_weight_scale = 1.*y_train.value_counts()[0]/y_train.value_counts()[1]
        print('class weight scale : {}'.format(class_weight_scale))
        model.set_params(**{'scale_pos_weight' : class_weight_scale})
        model.fit(X_train,y_train)
        pred_prob = model.predict_proba(X_valid)
        fpr, tpr, thresholds = roc_curve(y_valid, pred_prob[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        roc_aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=2, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

        i += 1
        
    plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k',
            label='Luck')

    mean_tpr /= K
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--',
            label='Mean ROC (area = %0.2f)' % mean_auc, lw=lw)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Initial estimator ROC curve')
    plt.legend(loc="lower right")

    fig.savefig(path_to_figures + 'ROC_AUC.png')
    return roc_aucs


def my_plot_importance(booster, figsize, **kwargs): 
    fig, ax = plt.subplots(1,1,figsize=(figsize))
    plot_importance(booster=booster, ax=ax, **kwargs)
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label,] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(10)
    plt.tight_layout()
    fig.savefig(path_to_figures + 'feature_importance.png')

    
def hyperparameter_optimization(xgb, x_df, y_df):
    X_train = x_df
    y_train = y_df['label']
    param_test0 = {
    'n_estimators':range(50,250,10)
    }
    print('performing hyperparamter optimization step 0')
    gsearch0 = GridSearchCV(estimator = xgb, param_grid = param_test0, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
    gsearch0.fit(X_train, y_train)
    print(gsearch0.best_params_, gsearch0.best_score_)

    param_test1 = {
    'max_depth':range(1,10),
    'min_child_weight':range(1,10)
    }
    print('performing hyperparamter optimization step 1')
    gsearch1 = GridSearchCV(estimator = gsearch0.best_estimator_,
    param_grid = param_test1, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
    gsearch1.fit(X_train, y_train)
    print(gsearch1.best_params_, gsearch1.best_score_)

    max_d = gsearch1.best_params_['max_depth']
    min_c = gsearch1.best_params_['min_child_weight']
    
    param_test2 = {
    'gamma':[i/10. for i in range(0,5)]
    }
    print('performing hyperparamter optimization step 2')
    gsearch2 = GridSearchCV(estimator = gsearch1.best_estimator_, 
    param_grid = param_test2, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
    gsearch2.fit(X_train,y_train)
    print(gsearch2.best_params_, gsearch2.best_score_)

    param_test3 = {
        'subsample':[i/10.0 for i in range(1,10)],
        'colsample_bytree':[i/10.0 for i in range(1,10)]
    }
    print('performing hyperparamter optimization step 3')
    gsearch3 = GridSearchCV(estimator = gsearch2.best_estimator_, 
    param_grid = param_test3, scoring='roc_auc', n_jobs=4,iid=False, cv=5)
    gsearch3.fit(X_train,y_train)
    print(gsearch3.best_params_, gsearch3.best_score_)

    param_test4 = {
        'reg_alpha':[0, 1e-5, 1e-3, 0.1, 10]
    }
    print('performing hyperparamter optimization step 4')
    gsearch4 = GridSearchCV(estimator = gsearch3.best_estimator_, 
    param_grid = param_test4, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
    gsearch4.fit(X_train, y_train)
    print(gsearch4.best_params_, gsearch4.best_score_)

    alpha = gsearch4.best_params_['reg_alpha']
    if alpha != 0:
        param_test4b = {
            'reg_alpha':[0.1*alpha, 0.25*alpha, 0.5*alpha, alpha, 2.5*alpha, 5*alpha, 10*alpha]
        }
        print('performing hyperparamter optimization step 4b')
        gsearch4b = GridSearchCV(estimator = gsearch4.best_estimator_, 
        param_grid = param_test4b, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
        gsearch4b.fit(X_train, y_train)
        print(gsearch4b.best_params_, gsearch4.best_score_)
        print('\nParameter optimization finished!')
        xgb_opt = gsearch4b.best_estimator_
        xgb_opt
    else:
        xgb_opt = gsearch4.best_estimator_
        xgb_opt

    best_scores = [gsearch0.best_score_, gsearch1.best_score_, gsearch2.best_score_, gsearch3.best_score_, gsearch4.best_score_]
    return xgb_opt, best_scores

def main():
    # Read preprocessed dataframes produced in script 2_preprocessing.py
    x_df = pd.read_csv(path_to_data + 'x_df.csv', index_col=0)
    y_df = pd.read_csv(path_to_data + 'y_df.csv')

    # Define the class weight scale (a hyperparameter) as the ration of negative labels to positive labels.
    # This instructs the classifier to address the class imbalance.
    class_weight_scale = 1.*y_df.label.value_counts()[0]/y_df.label.value_counts()[1]
    print(class_weight_scale)

    # Setting minimal required initial hyperparameters
    param={
        'objective': 'binary:logistic',
        'nthread': 4,
        'scale_pos_weight': class_weight_scale,
        'seed': 1   
    }
    xgb = XGBClassifier()
    xgb.set_params(**param)

    # Train initial classifier and analyze performace using K-fold cross-validation 
    K = 5
    roc_aucs_xgb = k_fold_cross_validation(K, xgb, x_df, y_df)
    
    # Option to perform hyperparameter optimization. Otherwise loads pre-defined xgb_opt params
    optimize = True

    if optimize:
        xgb_opt, best_scores = hyperparameter_optimization(xgb, x_df, y_df)
    else: 
        # Pre-optimized settings
        param={
        'objective':'binary:logistic',
        'nthread':4,
        'scale_pos_weight': class_weight_scale,
        'seed' : 1  ,
        'base_score': 0.5,
        'colsample_by_level': 1,
        'colsample_bytree': 0.7,
        'gamma': 0.1,
        'learning_rate': 0.1,
        'max_delta_step': 0,
        'max_depth': 3,
        'min_child_weight': 5,
        'missing': 0,
        'n_estimators': 70,
        'reg_alpha': 25.0,
        'reg_lambda': 1,
        'silent': True,
        'subsample':0.6
        }
        xgb_opt = XGBClassifier()
        xgb_opt.set_params(**param)
        
    print(xgb_opt)

    # K-fold cross-validation
    roc_aucs_xgb_opt = k_fold_cross_validation(K, xgb_opt, x_df, y_df)
    
    aucs = [np.mean(roc_aucs_xgb), np.mean(roc_aucs_xgb_opt)]
    if optimize: #append instead
        aucs = [np.mean(roc_aucs_xgb)] + best_scores + [np.mean(roc_aucs_xgb_opt)]
        #aucs = [np.mean(roc_aucs_xgb),
        #        gsearch0.best_score_,
        #        gsearch1.best_score_,
        #        gsearch2.best_score_,
        #        gsearch3.best_score_,
        #        gsearch4.best_score_,
        #        np.mean(roc_aucs_xgb_opt)]
        
    fig = plt.figure(figsize=(4,4))
    plt.scatter(np.arange(1,len(aucs)+1), aucs)
    plt.plot(np.arange(1,len(aucs)+1), aucs)
    plt.xlim([0.5, len(aucs)+0.5])
    plt.ylim([0.99*aucs[0], 1.01*aucs[-1]])
    plt.xlabel('Hyperparamter optimization step')
    plt.ylabel('AUC')
    plt.title('Hyperparameter optimization')
    plt.grid()
    fig.savefig(path_to_figures + 'optimization.png')

    print('Baseline') 
    print(classification_report(y_true = y_df.label, y_pred = np.zeros(y_df.shape[0])))
    print('xgb:')
    print(classification_report(y_true = y_df.label, y_pred = xgb_opt.predict(x_df)))
    print('xgb opt:')
    print(classification_report(y_true = y_df.label, y_pred = xgb_opt.predict(x_df)))

    my_plot_importance(xgb_opt, (5,10))
    
    print('done')

if __name__ == "__main__":
    main()

'''
Parameter optimization finished!
XGBClassifier(base_score=0.5, booster='gbtree', callbacks=None,
              colsample_bylevel=1, colsample_bynode=1, colsample_bytree=0.4,
              early_stopping_rounds=None, enable_categorical=False,
              eval_metric=None, gamma=0.0, gpu_id=-1, grow_policy='depthwise',
              importance_type=None, interaction_constraints='',
              learning_rate=0.300000012, max_bin=256, max_cat_to_onehot=4,
              max_delta_step=0, max_depth=2, max_leaves=0, min_child_weight=5,
              missing=nan, monotone_constraints='()', n_estimators=50, n_jobs=4,
              nthread=4, num_parallel_tree=1, predictor='auto', random_state=1,
              reg_alpha=50, ...)
'''

