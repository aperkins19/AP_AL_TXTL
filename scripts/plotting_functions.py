import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os

def stripplot_over_rounds(df_path, save_path, plot_name):

    Current_Total_Ground_Truth_Df = pd.read_csv(Grid_Path+"/Ground_Truths/MasterGroundTruth.csv")


    fig = plt.figure(figsize=(10,5))

    ax = sns.boxplot(x="Round #", y="Modelled Final Protein", data=Current_Total_Ground_Truth_Df, whis=np.inf, width=0.3)
    ax = sns.stripplot(x="Round #", y="Modelled Final Protein", data=Current_Total_Ground_Truth_Df, color=".3")

    #ax.set_ylim(0,300)

    fig.suptitle("Actual Protein Produced from MLP-Proposed Compositions")
    fig.tight_layout()


    ##### Save fig

    # make directory for sticking the output in
    if os.path.isdir(save_path) == False:
        os.mkdir(save_path, mode=0o777)

    #navigate to tidy_data_files
    os.chdir(save_path)

    plt.savefig(plot_name)



def barplot_MAE_over_rounds(mae_list, save_path, plot_name):

    mae_df = pd.DataFrame({"Round #": range(0,len(mae_list),1), "Average Mean Squared Error": mae_list})


    fig = plt.figure(figsize=(10,5))

    ax = sns.barplot(x="Round #", y="Average Mean Squared Error", data=mae_df)


    #ax.set_ylim(0,300)

    fig.suptitle("Average MAE over rounds")
    fig.tight_layout()


    ##### Save fig



    # make directory for sticking the output in
    if os.path.isdir(save_path) == False:
        os.mkdir(save_path, mode=0o777)


    #navigate to tidy_data_files
    os.chdir(save_path)

    plt.savefig(plot_name)