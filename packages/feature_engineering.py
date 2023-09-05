import os
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from etl import loadFinalDataset
import pandas as pd



def standardizeFeature(data, use_features):
    scale_features = [f for f in use_features if f not in ["age", "gender", "country", "study_name"]]
    if len(scale_features) == 0: return data

    scaler = StandardScaler()
    scaler.fit(data[scale_features].values)

    standardized_features = scaler.transform(data[scale_features].values)

    standardized_df = data.copy()
    standardized_df[scale_features] = standardized_features

    return standardized_df


def varianceThresholding(use_config, df):
    use_features = use_config["features"]
    variance_threshold = use_config["variance_threshold"]

    species = [f for f in use_features if f not in ["age", "gender", "country", "study_name"]]
    if len(species) == 0: return use_features
    
    selector = VarianceThreshold(variance_threshold)
    selector.fit_transform(df[species])

    filtered_taxa = df[species].columns[selector.get_support()].to_list()

    use_features = [f for f in use_features if f not in species]
    use_features = use_features + filtered_taxa

    return use_features


def medianFiltering(use_config, df):
    use_features = use_config["features"]
    # target_values = df[use_config["target"]].unique()

    new_features = []
    for feature in use_features:
        feature_median = df[feature].median()

        if feature_median > use_config["median_threshold"]:
            new_features.append(feature)
    
    return new_features
        

def featureEngineering(use_config, use_df, model_features):
    use_features = use_config["features"]
    use_target = use_config["target"]

    if use_config["feature_name"] == "key_mediators":
        new_features = model_features

    else:
        new_features = medianFiltering(use_config, use_df)

    standardized_df = use_df.copy()

    return standardized_df, new_features

# -------------------------------------------------------------------------------------
