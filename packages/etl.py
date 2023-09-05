import os
import pandas as pd
import numpy as np
# import skbio
# import pycountry
from string import digits
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler

import utils

def preprocessing_otu_dataset(dataset_taxa, level:int):

    def get_level(x, level):
        level_dict = {}
        level_dict = {
            1: x.split("|")[0],
            2: x.split("|")[1],
            3: x.split("|")[2],
            4: x.split("|")[3],
            5: x.split("|")[4],
            6: x.split("|")[5],
            7: x.split("|")[6]
        }

        default_level = 'Something went wrong'

        return level_dict.get(level, default_level)

    #removing all the samples with 0 values (removing columns with all zeroes value)
    dataset_taxa=dataset_taxa.loc[:, (dataset_taxa != 0).any(axis=0)]
    dataset_taxa['Taxa']=dataset_taxa['Taxa'].apply(lambda x: get_level(x, level=level))
    dataset_taxa=dataset_taxa.groupby(by="Taxa").sum().reset_index()
    
    #transposing dataset
    dataset_taxa=dataset_taxa.set_index("Taxa").T
    dataset_taxa=dataset_taxa.rename_axis("", axis=1)
    dataset_taxa=dataset_taxa.reset_index()
    dataset_taxa=dataset_taxa.rename(columns={'index':'sample_id'})
    dataset_taxa.set_index("sample_id", inplace=True)
    
    #removing all the species with all zeroes vaue (keep in mind that the dataset is transposed)
    dataset_taxa=dataset_taxa.loc[:, (dataset_taxa != 0).any(axis=0)]
    dataset_taxa=dataset_taxa.fillna(0)
    dataset_taxa = dataset_taxa.reset_index()

    return dataset_taxa


def filterAbsentTaxa(data, all_taxa):
    # Remove taxa not present in all samples (taxa with 0 relative abundance on all samples)
    absence_df = data \
        .set_index("sample_id")[all_taxa].T \
        .apply( lambda s : (s.value_counts().get(key=0,default=0)/data.shape[0])*100, axis=1) \
        .to_frame(name="percentage_of_absence") \
        .reset_index()
    
    absence_df.columns = ["Feature", "percentage_of_absence"]
    filtered_taxa = absence_df.loc[absence_df.percentage_of_absence!=100, "Feature"].values

    return filtered_taxa


def filterByPresence(data, all_taxa):
    # Remove taxa that are not present in at least 4 samples
    absence_df = data \
        .set_index("sample_id")[all_taxa].T \
        .apply( lambda s : (s.value_counts().get(key=0,default=0)/data.shape[0])*100, axis=1) \
        .to_frame(name="percentage_of_absence") \
        .reset_index()
    
    absence_df.columns = ["Feature", "percentage_of_absence"]
    filtered_taxa = absence_df.loc[absence_df.percentage_of_absence<=99.9, "Feature"].values

    return filtered_taxa


def categorizeAge(data):

    data.loc[data.age<=39, "age_group"] = "young"
    data.loc[(data.age>=40) & (data.age<=59), "age_group"] = "middle_age"
    data.loc[data.age>=60, "age_group"] = "elderly"

    return data


def addDiversityIndex(data, all_taxa):

    def compute_diversity(otus_abundance, diversity_func):
        result=[]

        for i in range(0, len(otus_abundance)):
            diversity_value = diversity_func(otus_abundance[i])
            result.append(diversity_value)
        
        return result

    data.loc[:, "shannon_diversity"] = compute_diversity(data[all_taxa].values, skbio.diversity.alpha.shannon)
    data.loc[:, "simpson_diversity"]= compute_diversity(data[all_taxa].values, skbio.diversity.alpha.simpson)
    data.loc[:, "chao1_diversity"]= compute_diversity(data[all_taxa].values, skbio.diversity.alpha.chao1)

    return data


def loadAbundanceMetadata(level, metadata_file, taxa_file):
    metadata = pd.read_csv(metadata_file)
    metadata.loc[metadata.non_westernized=="yes", "non_westernized"] = 1
    metadata.loc[metadata.non_westernized=="no", "non_westernized"] = 0
    metadata.non_westernized = metadata.non_westernized.astype("int")
    metadata.loc[:, "location"] = metadata.country.values

    taxa_df = pd.read_csv(taxa_file)
    taxa_df = taxa_df \
        .rename(columns={"Unnamed: 0": "Taxa"})

    taxa_df = preprocessing_otu_dataset(
        taxa_df, level
    )

    # Age range: 18-75 yrs
    use_samples = metadata.loc[
        (metadata.age>=18) &
        (metadata.age<=75),
        "sample_id"
    ].values
    taxa_df = taxa_df.loc[taxa_df.sample_id.isin(use_samples)]
    
    all_taxa = taxa_df.drop(["sample_id"], axis=1).columns.to_list()
    filtered_taxa = list(filterAbsentTaxa(taxa_df, all_taxa))
    filtered_taxa = list(filterByPresence(taxa_df, filtered_taxa))

    final_df = pd.merge(
        metadata.loc[metadata.sample_id.isin(use_samples)],
        taxa_df,
        left_on="sample_id",
        right_on="sample_id",
        how="outer"
    )

    return final_df, filtered_taxa


def loadPathwayData(pathway_file):
    pathway_df = pd.read_csv(pathway_file)
    pathway_df = pathway_df.rename(columns={"rowname": "Pathway"})

    pathway_df = pathway_df.set_index("Pathway").T \
        .reset_index() \
        .rename(columns={"index": "sample_id"}) \
        .drop(["UNMAPPED", "UNINTEGRATED"], axis=1)

    for pathway in pathway_df:
        newcol = pathway \
            .strip() \
            .lstrip(digits) \
            .lower() \
            .replace("-", "_") \
            .replace(" ", "") \
            .replace(";", "_") \
            .replace(".", "") \
            .replace("(", "_") \
            .replace(")", "_") \
            .replace(":", "_") \
            .replace("'", "_") \
            .replace("]", "_") \
            .replace("[", "_") \
            .replace("+", "") \
            .replace(",", "") \
            .replace("/", "") \
            .replace("&", "_") \
            .rstrip("_") \
            .rstrip("__")
            
        newcol = newcol.replace(" ", "") \
            .rstrip("_") \
            .rstrip("__")
        pathway_df = pathway_df.rename(columns={pathway: newcol})

    pathways = pathway_df.drop(["sample_id"], axis=1).columns.to_list()
    filtered_pathways = list(filterAbsentTaxa(pathway_df, pathways))
    
    return pathway_df, filtered_pathways   


def loadFinalDataset(level, metadata_file, abundance_file, pathways_file):
    # Abundance
    abundance_df, all_taxa = loadAbundanceMetadata(level, metadata_file, abundance_file)
    abundance_df = categorizeAge(abundance_df)

    pathway_df, pathways = loadPathwayData(pathways_file)

    final_df = pd.merge(abundance_df, pathway_df, on="sample_id", how="inner")
    final_df = categorizeAge(final_df)

    features_dict = {}
    features_dict["abundance"] = all_taxa
    features_dict["metadata_original"] = ["non_westernized", "location"]
    features_dict["pathway"] = pathways

    return final_df, features_dict


def mergeDataset(level):
    # Do not run this line
    # gt_file = os.path.join(utils.EDA_CSV_DIR, "dataset_with_gt_info.csv")
    # ground_truth_samples = pd.read_csv(gt_file)
    # ground_truth_samples = ground_truth_samples.loc[ground_truth_samples.ground_truth=="Yes"].sample_id.unique()
    # ground_truth_samples.shape

    metadata_file = os.path.join(utils.HEALTHY_DIR, "20221118_healthy_metadata_train.csv")
    abundance_file = os.path.join(utils.HEALTHY_DIR, "20221118_healthy_taxrelabund_train.csv")
    pathways_file = os.path.join(utils.HEALTHY_DIR, "20221118_healthy_pathrelabund_train.csv")
    healthy_df, healthy_features = loadFinalDataset(
        level,
        metadata_file,
        abundance_file,
        pathways_file
    )

    # healthy_df = healthy_df.loc[healthy_df.sample_id.isin(ground_truth_samples)]
    metadata_file = os.path.join(utils.DISEASED_DIR, "20230112_diseased_metadata_train.csv")
    abundance_file = os.path.join(utils.DISEASED_DIR, "20221207_diseased_taxrelabund_train.csv")
    pathways_file = os.path.join(utils.DISEASED_DIR, "20221207_diseased_pathrelabund_train.csv")
    diseased_df, diseased_features = loadFinalDataset(
        level,
        metadata_file,
        abundance_file,
        pathways_file
    )

    # Remove samples with antibiotic use
    healthy_df = healthy_df.loc[healthy_df.antibiotics_current_use.isin(["missing", "no"])]
    diseased_df = diseased_df.loc[diseased_df.antibiotics_current_use.isin(["missing", "no"])]

    use_species = set(healthy_features["abundance"]).intersection(set(diseased_features["abundance"]))
    use_species = list(set(use_species))

    use_pathways = set(healthy_features["pathway"]).intersection(set(diseased_features["pathway"]))
    use_pathways = list(set(use_pathways))

    final_df = healthy_df[["sample_id", "age", "study_name"]+use_species+use_pathways].copy()
    # final_df = healthy_df[["sample_id", "age", "age_group", "study_name", "country", "gender"]+use_pathways].copy()
    final_df.loc[:, "cohort"] = int(1)
    # final_df = pd.concat([final_df, diseased_df[["sample_id", "age", "age_group", "study_name", "disease_group", "country", "gender"]+use_pathways]])
    final_df = pd.concat([final_df, diseased_df[["sample_id", "age", "study_name","disease_group"]+use_species+use_pathways]])
    final_df.loc[final_df.cohort.isnull(), "cohort"] = int(0)

    print(final_df.cohort.value_counts())

    return final_df, use_species, use_pathways


def getModeDataset(data_mode, data):
    if data_mode == "train":
        file = "../data/healthy_vs_diseased/2_extras/train_indices_stratified_by_cohort_studyname.csv"
        # file = "../data/healthy_vs_diseased/2_extras/train_indices.csv"
    elif data_mode == "test":
        file = "../data/healthy_vs_diseased/2_extras/test_indices_stratified_by_cohort_studyname.csv"
        # file = "../data/healthy_vs_diseased/2_extras/test_indices.csv"
    
    indices = pd.read_csv(file).sample_id.values

    df = data.loc[data.sample_id.isin(indices)]

    return df


def getStratifiedDataset(data_mode, data, trial):

    train_file = f"../data/healthy_vs_diseased/replicate_validation/split.8/train_dataset_{trial}.csv"
    train_ids = pd.read_csv(train_file).sample_id.unique()

    if data_mode == "train":
        df = data.loc[data.sample_id.isin(train_ids)]
    
    elif data_mode == "test":
        df = data.loc[~data.sample_id.isin(train_ids)]
    
    return df


def getCrossValidationDataset(fold_number, data):
    file = "../data/healthy_vs_diseased/replicate_validation/cross_validation_sampling.csv"
    sampling_df = pd.read_csv(file)

    test_sampleids = sampling_df.loc[sampling_df.fold_number==fold_number].sample_id.unique()
    test_df = data.loc[data.sample_id.isin(test_sampleids)]
    train_df = data.loc[~data.sample_id.isin(test_sampleids)]

    return train_df, test_df


if __name__ == "__main__":
    df, species, use_pathways = mergeDataset(7)
    df.loc[:, "age_treatment"] = pd.cut(df.age, 2, labels=[0, 1])

    print(df[use_pathways])

    # pathways = df.drop(["sample_id", "age", "age_group", "study_name", "country", "gender", "cohort", "disease_group", "age_treatment"], axis=1).columns.to_list()

    # for p in pathways:
    #     newcol = p.replace("-", "_").replace(" ", "_").replace(";", "_").replace(".", "").replace("(", "_").replace(")", "_")

    #     df = df.rename(columns={p: newcol})

    # print(df.head())

    # df.to_csv(
    #     "../data/mediation_analysis/csv/pathways_healthy_diseased_dataset.csv",
    #     index=False
    # )
