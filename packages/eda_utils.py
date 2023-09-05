import sys
import os
# import skbio
import pandas as pd
import plotly.express as px
import numpy as np
from scipy import stats
from scipy.spatial import distance
from statsmodels.stats.multitest import multipletests

import utils
from utils import add_template_and_save



def KS_test(df, categories, column_cat, taxa):
    groups_to_test=[]
    for cat in categories:
        groups_to_test.append(df[df[column_cat]==cat][taxa].to_list())
    
    check_group=set( [item for sublist in groups_to_test for item in sublist])

    if len(check_group)==1 or len(groups_to_test)==0:
    
        return "not performed"
    else:
        p_value=stats.kruskal(*groups_to_test)[1]

    return p_value


def compute_diversity(otus_abudance, diversity_func):
    result=[]

    for i in range(0, len(otus_abudance)):
        diversity_value = diversity_func(otus_abudance[i])
        result.append(diversity_value)
    
    return result


def getEffectSize(df, use_var):

    effect_df = pd.DataFrame()

    for i, var in enumerate(use_var):
        healthy_median = df.loc[df.cohort=="Healthy", var].median()
        diseased_median = df.loc[df.cohort=="Diseased", var].median()

        effect_size = healthy_median - diseased_median
        healthy_median = round(healthy_median, 2)
        diseased_median = round(diseased_median, 2)

        effect_df.loc[i, "taxa"] = var
        effect_df.loc[i, "healthy"] = healthy_median
        effect_df.loc[i, "diseased"] = diseased_median
        effect_df.loc[i, "effect_size"] = effect_size
        effect_df.loc[i, "out"] = f"{round(effect_size, 2)} ({healthy_median} - {diseased_median})"
    
    effect_df.to_csv("../data/eda/csv/effect_size.csv", index=False)


def kruskalWallisWithCorrection(df, use_variables, use_col):
    ks_df = pd.DataFrame()
    for i, var in enumerate(use_variables):

        try:
            test_obj = stats.kruskal(
                df.loc[df.cohort=="Healthy", var],
                df.loc[df.cohort=="Diseased", var]
            )
            ks_df.loc[i, "pvalue"] = test_obj.pvalue
            ks_df.loc[i, "taxa"] = var

        except:
            ks_df.loc[i, "pvalue"] = -99
            ks_df.loc[i, "taxa"] = var
            
    pvalue_list = ks_df.loc[ks_df.pvalue!=-99].pvalue.values
    var_list = ks_df.loc[ks_df.pvalue!=-99].taxa.values
    
    corrected_pvalue = multipletests(pvalue_list, method="bonferroni")[1]
    corrected_result = multipletests(pvalue_list, method="bonferroni")[0]
    corrected_df = pd.DataFrame(corrected_result, columns=["decision"])

    corrected_df.loc[:, "corrected_pvalue"] = corrected_pvalue
    corrected_df.loc[:, "pvalue"] = pvalue_list
    corrected_df.loc[:, "variable"] = var_list

    significant_variables = corrected_df.loc[corrected_df.decision==True, "variable"].values

    # getEffectSize(df, significant_variables)

    return significant_variables


def generateComparisonByDistribution(df=None, level=None, use_col=None, use_taxa=None, xcmap=None, xorder=None, age_group="all", path=None):

    if path is not None:
        save_dir = os.path.join(path, f"L{level}", age_group)
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
    
    for taxa in use_taxa:
        fig = px.box(
            df, x=use_col, y=taxa,
            labels={taxa: "Relative Abundance"},
            color=use_col,
            color_discrete_map=xcmap,
            category_orders=xorder,
            hover_data=["sample_id", "age", "study_name", "country"],
            points="all",
            template="plotly_white+presentation"
        )

        fig.update_layout(xaxis_title=None)
        fig.update_layout(legend={'title_text':''})
        fig.update_layout(showlegend=False)
        
        add_template_and_save(
            fig,
            title=f"Distribution of {taxa} by Cohort (KSW Test adj p-value < 0.05)",
            subtitle=None,
            # subtitle="Differentially Abundant Taxa Based on Kruskal-Wallis Test with Bonferroni Correction",
            # chart_file_name=os.path.join(save_dir, f"{taxa}.html")
            chart_file_name=None
        )

        # fig.show()


def get_distance_matrix(df, metric="braycurtis"):
    distance_matrix=distance.squareform(distance.pdist(df.values, metric=metric))
    #avoiding NaN values
    distance_matrix=pd.DataFrame(distance_matrix).fillna(0).values

    return distance_matrix


def pcoa_dim_reduction(distance_matrix, final_df, number_of_dimensions=2):
    pcoa_data_2d =skbio.stats.ordination.pcoa(distance_matrix, number_of_dimensions=number_of_dimensions)
    pcoa_df_2d = pcoa_data_2d.samples
    pcoa_df_2d.loc[:, "sample_id"] = final_df.sample_id.values

    reduced_df = pd.merge(final_df, pcoa_df_2d, on="sample_id", how="left")

    return reduced_df, pcoa_data_2d


def plotPCoA(reduced_df=None, reduced_arr=None, level=None, age_group="all", use_col=None, cmap=None, path=None):
    fig = px.scatter(
        reduced_df, x="PC1", y="PC2", color=use_col, 
        color_discrete_map=cmap, opacity=0.7,
        hover_data=["sample_id", "age", "study_name", "country", "disease_group"])
    
    # save_dir = os.path.join(path, f"L{level}", age_group, "high_level")

    # if not os.path.isdir(save_dir):
    #     os.mkdir(save_dir)
        
    fig.update_layout(
        xaxis_title=f"PC1({np.round(reduced_arr.proportion_explained[0]*100,2)}%)",
        yaxis_title=f"PC2({np.round(reduced_arr.proportion_explained[1]*100,2)}%)",
        font=dict(
            
            size=15,
            color="white"
        ),
        legend_itemsizing='constant'
    )
    fig.update_traces(marker=dict(size=8,
                                symbol = 'circle',
                                line=dict(width=1,
                                            color='DarkSlateGrey')))
    fig.update_layout(font=dict(color="#1c3763"),plot_bgcolor='rgba(0,0,0,0)')
    fig.update_layout(legend={'title_text':''})

    add_template_and_save(
        fig,
        title=f"PCoA-2D Significant Taxa (Negative Association)",
        # chart_file_name=os.path.join(save_dir, f"PCoA_{age_group}_hnegative.html")
        chart_file_name=None
    )

    fig.show()


def generate_comparison_chart_diversity(df, level, use_col, taxa, path, 
    name_of_diversity, name_of_category,
    cmap, x_order):

    # diversity_path = os.path.join(config.OUTPLOT_DIR, path)
    # if not os.path.isdir(diversity_path):
    #     os.mkdir(diversity_path)

    p_value = KS_test(df,list(df[use_col].unique()),use_col,taxa)
    print(p_value)
    if p_value != "not performed" and float(p_value)<0.05:

        fig = px.violin(
            df, x=use_col, y=taxa, color=use_col,
            color_discrete_map=cmap,
            category_orders=x_order,
            points="all", 
            template="plotly_white",
            hover_data=["sample_id", "age", "study_name", "country"]
        )
        fig.update_layout(legend={'title_text':''})

        add_template_and_save(
            fig,
            title=f"{name_of_diversity} Diversity Distribution (KSW Test p-value < 0.05)",
            chart_file_name=os.path.join(
                path,
                f"alpha_diversity_{name_of_diversity}_L{level}.html")
        )

        fig.show()