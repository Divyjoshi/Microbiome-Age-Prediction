import warnings
warnings.filterwarnings("ignore")

import os
import sys
import numpy as np
import pandas as pd

from itertools import product

sys.path.insert(0, "packages")

from packages.etl import mergeDataset
from packages.model_utils import comprehensizeModelPerformance


def runExperiment(val):
    which_level = 7
    use_df, use_species, use_pathways = mergeDataset(which_level)

    # significant_species = list(pd.read_csv("csv/significant_taxa.csv").taxa.values)
    # key_mediators = list(pd.read_csv("csv/key_mediators.csv").taxa.values)

    use_config = {}
    use_config["trial"] =  10
    use_config["level"] = which_level
    use_config["target"] = "cohort"
    use_config["feature_name"] = "species"
    use_config["features"] = use_species
    use_config["median_threshold"] =  0

    hd_config = {}
    hd_config["model"] = "lgbm"
    hd_config["prob_threshold"] = 0.7
    hd_config["target"] = "cohort"

    ma_config = {}
    ma_config["model"] = "random_forest"
    ma_config["prob_threshold"] = 0.45
    ma_config["target"] = "age"

    comprehensizeModelPerformance(use_config, hd_config, ma_config)


def collect():

    hd_thresh = np.arange(0, 1, 0.1)
    ma_thresh = np.arange(0, 1, 0.1)

    for t in product(hd_thresh, ma_thresh):
        print(f"Running experiment for {t}")

        runExperiment(t)

        break
        

if __name__ == "__main__":
    runExperiment(0)
    #collect()
    #collectPerformance()