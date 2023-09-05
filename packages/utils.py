import plotly.express as px
import os
import numpy as np

import plotly.graph_objects as go
from operator import itemgetter
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


#EDA_CSV_DIR = "../data/eda/csv"
# DISEASED_DIR = "D:\Health Data Science\healthy_vs_diseased_true\healthy_vs_diseased\curatedMD_diseased_train"
# HEALTHY_DIR = "D:\Health Data Science\healthy_vs_diseased_true\healthy_vs_diseased\curatedMD_healthy_train"
DISEASED_DIR = "curatedMD_diseased_train"
HEALTHY_DIR = "curatedMD_healthy_train"

#UNIVARIATE_ANALYSIS_DIR = "../data/eda/plots/univariate_analysis"

# HEALTHY_VS_DISEASED_DIR = "../data/healthy_vs_diseased"
# DASHBOARD_DIR = "D:/Health Data Science/healthy_vs_diseased_true/healthy_vs_diseased/dashboard"
DASHBOARD_DIR = "dashboard"


cohort_cmap = {
    "Healthy": px.colors.qualitative.Bold[2],
    "Diseased": px.colors.qualitative.Bold[6]
}

# cohort_cmap = {
#     "1": px.colors.qualitative.Bold[2],
#     "0" : px.colors.qualitative.Bold[6]
# }

gt_cmap = {
    "no": px.colors.qualitative.D3[2],
    "yes": px.colors.qualitative.D3[3]
}


custom_template = {
    "layout": go.Layout(
        font={
            "family": "Nunito",
            "size": 12,
            "color": "#707070",
        },
        title={
            "font": {
                "family": "Lato",
                "size": 18,
                "color": "#1f1f1f",
            },
        },
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        colorway=px.colors.qualitative.G10,
    )
}

def initModel(use_config):
    outdir = os.path.join(
        DASHBOARD_DIR,
        use_config["feature_name"]
    )

    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    
    return outdir


def saveModelSummary(model, outdir):

    params = model.params
    conf = model.conf_int()
    conf['Odds Ratio'] = params
    conf = np.exp(conf)
    conf['Coefficient'] = params
    conf["pvalue"] = model.pvalues
    conf.columns = ['5%', '95%', 'Odds Ratio', "Coefficient", "pvalues"]
    conf.to_csv(os.path.join(outdir, "params.csv"))

    plt.rc('figure')
    plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "model_summary.png"))


def saveConfusionMatrix(ytest, ypred, outdir):
    labels = ["Healthy", "Diseased"]
    cm = confusion_matrix(ytest, ypred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm)
    
    disp.plot(cmap="Blues")
    plt.show()
    # disp.savefig(os.path.join(outdir, "confusion_matrix_testset.png"))


def saveCSV(df, outfile):
    if os.path.isfile(outfile):
        df.to_csv(outfile, mode="a", header=None, index=False)
    
    else:
        df.to_csv(outfile, index=False)


def format_title(title, subtitle=None, subtitle_font_size=14):
    # title = f'<b>{title}</b>'
    title = f'{title}'
    if not subtitle:
        return title
    subtitle = f'<span style="font-size: {subtitle_font_size}px;">{subtitle}</span>'
    return f'{title}<br>{subtitle}'


def add_template_and_save(figure, title=None, subtitle=None, chart_file_name=None, x=0.1, y=1.01):
    fig=figure
    fig.add_layout_image(
        dict(
            dict(source="https://i.ibb.co/zZbcnVq/image001.png", xref="paper", yref="paper", x=x, y=y, sizex=0.1, sizey=0.1, xanchor="right",
             yanchor="bottom")
        )
    )

    # update layout properties
    fig.update_layout(
        title=format_title(title, subtitle),
        title_x=0.5,
        font=dict(
            
            size=18,
            color="white"
        ),
        legend_itemsizing='constant',
        template="plotly_white"
    )

    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', font=dict(color="#1c3763"), paper_bgcolor="white")
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=px.colors.qualitative.Pastel1[8])
    fig.update_layout(
        xaxis=dict(zeroline=False, showgrid=False), 
        yaxis=dict(zeroline=False, showgrid=False))
    fig.update_yaxes(zerolinecolor = 'rgba(0,0,0,0)', showticklabels=True, showgrid=False)
    
    fig.write_html(f"{chart_file_name}")


def getRegressors(model_name):
    if model_name == "gradient_boosting":
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor()
    
    elif model_name == "elastic_net":
        from sklearn.linear_model import ElasticNet
        return ElasticNet()
    
    elif model_name == "stochastic_gradient_descent":
        from sklearn.linear_model import SGDRegressor
        return SGDRegressor()
    
    elif model_name == "SVM":
        from sklearn.svm import SVR
        return SVR()
    
    elif model_name == "bayesian_ridge":
        from sklearn.linear_model import BayesianRidge
        return BayesianRidge()
    
    elif model_name == "kernel_ridge":
        from sklearn.kernel_ridge import KernelRidge
        return KernelRidge()
    
    elif model_name == "xgboost":
        from xgboost.sklearn import XGBRegressor
        return XGBRegressor(
            max_depth=2,
            n_estimators=3
        )
    
    elif model_name == "lgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(
            num_leaves=3
        )
    
    elif model_name == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(max_depth=2, random_state=0)



# def getClassifiers(model_name, nfeatures, ntarget):
def getClassifiers(model_name):
    if model_name == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression()
    
    elif model_name == "SVM_classifier":
        from sklearn.svm import SVC
        return SVC()
    
    elif model_name == "gaussian_naive_bayes":
        from sklearn.naive_bayes import GaussianNB
        return GaussianNB()
    
    elif model_name == "multinomial_naive_bayes":
        from sklearn.naive_bayes import MultinomialNB
        return MultinomialNB()
    
    elif model_name == "knn":
        from sklearn.neighbors import KNeighborsClassifier
        return KNeighborsClassifier()
    
    elif model_name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier()
    
    elif model_name == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            # num_leaves=3
        )
    
    elif model_name == "xgboost":
        from xgboost.sklearn import XGBClassifier
        return XGBClassifier(
            max_depth=2,
            n_estimators=3
        )
    
    # elif model_name == "NN":
    #     from sklearn.neural_network import MLPClassifier
    #     return MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 58), random_state=1)

    # elif model_name == "DNN":
    #     model = Sequential()
    #     model.add(Dropout(0.5, input_shape=(nfeatures,)))
    #     model.add(Dense(512, activation='relu'))
    
    #     model.add(Dense(76, activation='softmax'))
    #     # Compile model
    #     adam_opt = Adam(learning_rate=0.001)
    #     model.compile(loss='categorical_crossentropy', optimizer=adam_opt, metrics=['accuracy'])
    #     # model.compile(loss='mean_squared_error', optimizer=adam_opt)

    #     return model


# def getDNN():

#     model = Sequential()
#     model.add(Dropout(0.5, input_shape=(1567,)))
#     model.add(Dense(512, activation='relu'))
    
#     model.add(Dense(74, activation='softmax'))

#     # Compile model
#     adam_opt = Adam(learning_rate=0.001)
#     model.compile(loss='categorical_crossentropy', optimizer=adam_opt, metrics=['accuracy'])
#     # model.compile(loss='mean_squared_error', optimizer=adam_opt)

#     return model



def getFeatures(features, features_dict):

    if features == "abundance":
        return features_dict["abundance"]
    
    elif features == "pathways":
        return features_dict["pathway"] 
    
    elif features == "metadata":
        return features_dict["metadata"]  
    
    elif features == "diversity":
        return features_dict["diversity"]

    elif features == "enterotypes":
        return features_dict["enterotypes"]  
    
    elif features == "kendall":
        return features_dict["kendall"]   
    
    elif features == "abundance+pathways":
        return features_dict["abundance"] + features_dict["pathway"]  
    
    elif features == "abundance+diversity":
        return features_dict["abundance"] + features_dict["diversity"] 
    
    elif features == "pathways+diversity":
        return features_dict["pathway"] + features_dict["diversity"] 
    
    elif features == "abundance+metadata":
        return features_dict["abundance"] + features_dict["metadata"]  
    
    elif features == "metadata+diversity":
        return features_dict["metadata"] + features_dict["diversity"] 
    
    elif features == "pathways+metadata":
        return features_dict["pathway"] + features_dict["metadata"] 
    
    # elif features == "pathways+enterotypes":
    #     return pathways + enterotypes 
    
    # elif features == "pathways+kendall":
    #     return pathways + kendall_taxa
    
    # elif features == "enterotypes+diversity":
    #     return enterotypes + diversity_index
    
    # elif features == "kendall+diversity":
    #     return kendall_taxa + diversity_index

    elif features == "abundance+pathways+metadata":
        return features_dict["abundance"] + features_dict["pathway"] + features_dict["metadata"] 
    
    elif features == "abundance+pathways+diversity":
        return features_dict["abundance"] + features_dict["pathway"] + features_dict["diversity"] 
    
    elif features == "abundance+metadata+diversity":
        return features_dict["abundance"] + features_dict["metadata"] + features_dict["diversity"] 
    
    elif features == "abundance+pathways+diversity+metadata":
        return features_dict["abundance"] + features_dict["pathway"] + features_dict["diversity"] + features_dict["metadata"] 
    
    # elif features == "enterotypes+pathways+diversity":
    #     return enterotypes + pathways + diversity_index
    
    # elif features == "kendall+pathways+diversity":
    #     return kendall_taxa + pathways + diversity_index
    
    # elif features == "enterotypes+kendall+pathways+diversity":
    #     return enterotypes + kendall_taxa + pathways + diversity_index
