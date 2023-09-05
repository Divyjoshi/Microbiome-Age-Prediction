import warnings


import warnings
warnings.filterwarnings("ignore")

import os
import pandas as pd
import pickle
import random

from lightgbm import LGBMClassifier, plot_importance, plot_tree
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from etl import mergeDataset
from etl import preprocessing_otu_dataset
from etl import getModeDataset

from feature_engineering import featureEngineering


def getBestModelConfig():

    use_df, use_species, use_pathways = mergeDataset(7)

    significant_species = list(pd.read_csv("csv/significant_taxa.csv").taxa.values)
    key_mediators = list(pd.read_csv("csv/key_mediators.csv").taxa.values)


    use_config = {}
    use_config["level"] = 7
    use_config["target"] = "cohort"
    use_config["feature_name"] = "species"
    use_config["features"] = use_species
    use_config["alpha"] =  0.07
    use_config["toptaxa_percentage"] =  0.25
    use_config["variance_threshold"] =  0
    use_config["median_threshold"] =  0

    return use_config



def buildDataset(taxa_file):

    taxa_df = pd.read_csv(taxa_file)
    taxa_df = taxa_df \
        .rename(columns={"Unnamed: 0": "Taxa"})

    taxa_df = preprocessing_otu_dataset(
        taxa_df, 7
    )

    all_taxa = taxa_df.drop(["sample_id"], axis=1).columns.to_list()
    
    return taxa_df


def predictOnMultipleSamples(use_config, test_df):

    use_df, use_species, use_pathways = mergeDataset(use_config["level"])

    print("Class size")
    print(use_df.cohort.value_counts())

    train_standardized, model_features = featureEngineering(
        use_config,
        use_df, use_config["features"])

    model = LGBMClassifier()

    model.fit(train_standardized[use_species], train_standardized[["cohort"]])

    features = pd.DataFrame(use_species, columns=["predictors"])
    # features.to_csv("healthy_vs_diseased_model/all_species.csv", index=False)
    # sample_df[use_species+["sample_id", "age"]].to_csv("healthy_vs_diseased_model/demo_explain.csv",index=False)
    sample_df = use_df.copy()
    print(sample_df.cohort.value_counts())
    healthy = sample_df.loc[sample_df.cohort==1.0].sample_id.values[:50]
    diseased = sample_df.loc[sample_df.cohort==0.0].sample_id.values[:50]
    sample_df = sample_df.loc[sample_df.sample_id.isin(list(healthy) + list(diseased))]
    sample_df.cohort = sample_df.cohort.map({
        1.0: "Healthy",
        0.0: "Diseased"
    })

    print(sample_df.cohort.value_counts())



    sample_df = sample_df.rename(columns={"cohort": "group", "sample_id": "sampleID"})
    sample_df = sample_df.set_index("sampleID")[use_species + ["age", "group"]].T
    sample_df.index.name = "sampleID"

    
    # sample_df.to_csv("healthy_vs_diseased_model/group_demo_dataset.csv", sep="\t")
    
    # filename = 'healthy_vs_diseased_model/hd_all_species.sav'
    # pickle.dump(model, open(filename, 'wb'))

    # y_pred = model.predict(use_df[model_features])

    # prediction = pd.DataFrame({'sample_id':use_df.sample_id.values,'Prediction':y_pred})
    # prediction.loc[:, "Actual"] = use_df.cohort.values

    # prediction.loc[prediction.Prediction==0, "Prediction_class"] = "Diseased"
    # prediction.loc[prediction.Prediction==1, "Prediction_class"] = "Healthy"

    # diseased_sample = prediction.loc[prediction.Actual==prediction.Prediction]
    # diseased_sample = diseased_sample.loc[diseased_sample.Actual==0].sample_id.unique()
    # diseased_df = use_df.loc[use_df.sample_id.isin(diseased_sample)]

    # key_mediators = list(pd.read_csv("csv/key_mediators.csv").taxa.values)

    # use_sample = random.randint(0, diseased_df.shape[0])

    # demo = diseased_df.loc[diseased_df.index==use_sample, model_features+["age", "sample_id"]] \
    #     .T \
    #     .reset_index()
    # demo.columns = ["otuID", "relative_abundance"]

    # demo.to_csv("healthy_vs_diseased_model/demo_diseased_single_prediction.csv", index=False, sep="\t")

    # diseased_df[model_features+["sample_id", "age"]].to_csv("healthy_vs_diseased_model/demo_diseased_samples.csv", index=False)

    

    # accuracy = metrics.accuracy_score(prediction.Actual, prediction.Prediction)
    # auc = metrics.roc_auc_score(prediction.Actual.values, prediction.Prediction.values)

    # print(f"Accuracy: {round(accuracy*100, 0)}%")
    # print(f"AUC: {auc}")
    # print(prediction.Prediction.value_counts())

    # cm = confusion_matrix(prediction.Actual, prediction.Prediction)

    # disp = ConfusionMatrixDisplay(
    #     confusion_matrix=cm)
    
    # disp.plot(cmap="Blues")
    # plt.show()

    # prediction.to_csv(f"../data/curatedMD_healthydiseased_validation/validation/prediction_{use_config['feature_name']}.csv", index=False)



if __name__ == "__main__":
    TEST_DATA_DIR = "../data/curatedMD_healthydiseased_validation"
    healthy_file = os.path.join("../data/curatedMD_healthy_test", "20221118_healthy_taxrelabund_test.csv")
    taxa_file = os.path.join(TEST_DATA_DIR, "20230315_taxrelabund_healthydiseased_test.csv")

    use_config = getBestModelConfig()

    test_df = buildDataset(taxa_file)
    healthy_df = buildDataset(healthy_file)

    print(healthy_df.shape)

    all_taxa = healthy_df.drop("sample_id", axis=1).columns.to_list()
    all_taxa = [taxa for taxa in all_taxa if taxa in test_df.columns]

    new_df = pd.concat([test_df, healthy_df])
    new_df = new_df.drop_duplicates(subset=all_taxa, keep=False).sample_id.unique()

    test_df.loc[test_df.sample_id.isin(new_df), "y_test"] = 0
    test_df.loc[~test_df.sample_id.isin(new_df), "y_test"] = 1

    print(test_df.y_test.value_counts())

    predictOnMultipleSamples(use_config, test_df)

    # file ="healthy_vs_diseased_model/demo_diseased_single_prediction.csv"
    # df = pd.read_csv(file, sep="\t")
    # print(df.head())
