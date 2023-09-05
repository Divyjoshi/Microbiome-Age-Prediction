import warnings
warnings.filterwarnings("ignore")

import os
import pandas as pd
import numpy as np
import statsmodels.formula.api as sm
import statistics
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest
from lightgbm import LGBMClassifier, plot_importance, plot_tree

import utils
from utils import initModel
from utils import saveModelSummary
from utils import saveCSV
from utils import saveConfusionMatrix
from utils import getFeatures
from utils import getClassifiers

from etl import mergeDataset
from etl import loadFinalDataset
from etl import getModeDataset
from etl import getCrossValidationDataset
from etl import getStratifiedDataset

from itertools import product

from feature_engineering import featureEngineering




def getMLPrediction(use_config, cv_features, X_train, y_train, X_test, target):

    model = getClassifiers(use_config["model"])
    model.fit(X_train[cv_features], y_train)

    ypred = model.predict(X_test[cv_features])
    ypred = [int(pred) for pred in ypred]

    yprob = model.predict_proba(X_test[cv_features])
    yprob = [prob.max() for prob in yprob]

    return ypred, yprob


def getPerformance(prediction_df):
    accuracy = metrics.balanced_accuracy_score(prediction_df.ytest, prediction_df.ypred)

    auc = metrics.roc_auc_score(prediction_df.ytest, prediction_df.ypred)
    
    auc_new = metrics.roc_auc_score(prediction_df.ytest, prediction_df.prob_prediction)

    return accuracy, auc, auc_new


def evaluateModel(use_config, model_config, data):

    X = data.reset_index().copy()
    y = data.reset_index().copy()

    cross_validator = StratifiedKFold(n_splits=10)

    prediction_df = pd.DataFrame()
    features = []
    counter = 0
    for train_index, test_index in cross_validator.split(data, data[["study_name"]]):

        Xtrain = X.loc[X.index.isin(train_index)]
        Xtest = X.loc[X.index.isin(test_index)]
        ytest = y.loc[y.index.isin(test_index)]

        # ====================== FEATURE ENGINEERING ===========================

        Xtrain_standardized, cv_features = featureEngineering(use_config, Xtrain, use_config["features"])
        Xtest_standardized, _ = featureEngineering(use_config, Xtest, cv_features)
        cv_features = use_config["features"]

        # ====================== FEATURE ENGINEERING ===========================

        y_pred, yprob = getMLPrediction(
            model_config, 
            cv_features, 
            Xtrain_standardized, 
            Xtrain[[model_config["target"]]],
            Xtest_standardized, 
            model_config["target"]
        )

        loop_df = pd.DataFrame(y_pred, columns=["ypred"])
        loop_df.loc[:, "ytest"] = ytest[[model_config["target"]]].values
        loop_df.loc[:, "sample_id"] = ytest.sample_id.values
        loop_df.loc[:, "yprob"] = yprob

        prediction_df = pd.concat([prediction_df, loop_df])

        # print(f"Counter: {counter}")

        counter += 1

    return prediction_df


def comprehensizeModelPerformance(use_config, hd_config, ma_config):

    outdir = initModel(use_config)

    use_df, use_species, use_pathways = mergeDataset(use_config["level"])

    # ====================== HEALTHY VS DISEASED MODEL =========================

    hd_prediction = evaluateModel(use_config, hd_config, use_df)

    # Add Ground-Truth and Probability Prediction
    hd_prediction.loc[:, "prob_prediction"] = hd_prediction.yprob.apply(
        lambda x: 1 if x>=hd_config["prob_threshold"] else 0
    )
    hd_prediction.loc[:, "ground_truth"] = hd_prediction.prob_prediction.values
    hd_prediction.loc[hd_prediction.ytest==0, "ground_truth"] = 0

    hd_accuracy, hd_auc, auc_new = getPerformance(hd_prediction)

    print(f"Accuracy: {hd_accuracy}")
    print(f"AUC: {hd_auc}")
    print(f"AUC New: {auc_new}")

    hd_prediction_file = os.path.join(
      outdir, 
      f"prediction_{use_config['feature_name']}.csv")
    hd_prediction.to_csv(
            hd_prediction_file,
            index=False
        )
    
    # ====================== HEALTHY VS DISEASED MODEL =========================

    # ====================== MICROBIAL AGE MODEL ===============================

    ground_truth_samples = hd_prediction.loc[hd_prediction.ground_truth==1].sample_id.unique()
    ground_truth_df = use_df.loc[use_df.sample_id.isin(ground_truth_samples)]

    percentage_gt = len(ground_truth_samples)/use_df.loc[use_df.cohort==1].shape[0]*100

    print(f"PERCENTAGE OF GROUND-TRUTH SAMPLES: {round(percentage_gt, 0)}%")

    ma_prediction = evaluateModel(use_config, ma_config, ground_truth_df)
    ma_prediction.loc[:, "final_prediction"] = ma_prediction.apply(
        lambda x: x.ypred if x.yprob>=ma_config["prob_threshold"] else None, axis=1
    )
    ma_prediction.loc[:, "original_ae"] = abs(ma_prediction.ypred - ma_prediction.ytest)
    ma_prediction.loc[:, "final_ae"] = abs(ma_prediction.final_prediction - ma_prediction.ytest)

    original_mae = ma_prediction.original_ae.mean()
    new_mae = ma_prediction.final_ae.mean()

    print(f"MICROBIAL AGE MODEL PERFORMANCE (ORIGINAL): {original_mae}")
    print(f"MICROBIAL AGE MODEL PERFORMANCE (NEW): {new_mae}")

    # ma_prediction_file = os.path.join(
    #   outdir, 
    #   f"microbial_age_prediction_{use_config['feature_name']}.csv")
    # ma_prediction.to_csv(
    #         ma_prediction_file,
    #         index=False
    #     )

    # ====================== MICROBIAL AGE MODEL ===============================

    # # SaveModel Performance
    performance_df = pd.DataFrame() 
    performance_df.loc[0, "features"] = use_config["feature_name"]
    performance_df.loc[0, "median_threshold"] = use_config["median_threshold"]
    performance_df.loc[0, "hd_prob_threshold"] = hd_config["prob_threshold"]
    performance_df.loc[0, "ma_prob_threshold"] = ma_config["prob_threshold"]
    performance_df.loc[0, "hd_model"] = hd_config["model"]
    performance_df.loc[0, "ma_model"] = ma_config["model"]
    performance_df.loc[0, "accuracy"] = hd_accuracy
    performance_df.loc[0, "auc"] = hd_auc
    performance_df.loc[0, "new_auc"] = auc_new   
    performance_df.loc[0, "original_mae"] = original_mae
    performance_df.loc[0, "new_mae"] = new_mae     


    print(performance_df)
    # saveCSV(
    #     performance_df,
    #     "D:\Health Data Science\healthy_vs_diseased_true\healthy_vs_diseased\Replicate_validation/replicated_model_performance_all.csv"
    # )

    return 0