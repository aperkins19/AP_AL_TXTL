import xgboost

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

path = "datasets/grids/Ground_Truths/MasterGroundTruth.csv"

data = pd.read_csv(path)


data = data[['NTP', 'Polymerised Nucleotide', 'Exhausted Nucleotide', 'tRNA',
       'Amino Acids', 'Creatine_Phosphate', 'Pyrophosphate', 'Creatine',
       'TL Enzymes', 'Modelled Final Protein']]

features = ['NTP', 'Polymerised Nucleotide', 'Exhausted Nucleotide', 'tRNA',
'Amino Acids', 'Creatine_Phosphate', 'Pyrophosphate', 'Creatine',
'TL Enzymes']


data = data.iloc[:-100, : ]

print(data.head)

x_pred = data.loc[-100:, features]



X = data.loc[:, features]
y = data.loc[:, "Modelled Final Protein"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

model = xgboost.XGBRegressor(objective='reg:squarederror',
                            learning_rate=0.01,
                            seed=578)

model.fit(X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=False)

#fi = pd.DataFrame(
#                data=model.feature_importances_,
#                index=model.feature_names_in_,
#                columns=["importance"]
#                )


#fi_plot =fi.sort_values("importance").plot("barh", title="importance")

eval_preds = model.predict(X_test)

mae = mean_absolute_error(y_test, eval_preds)

print("MAE: " +str(mae))


print("")
print("Predictions:")

real_preds = model.predict(x_pred)

x_pred['xgb preds'] = real_preds
x_pred['Modelled Final Protein'] = data.loc[-100:, "Modelled Final Protein"]

print(x_pred[["xgb preds", "Modelled Final Protein"]])

print("test")

print(x_pred.info)

print("test")

def x_preds_plot(df, save_path):


    fontsize =20

    fig = plt.figure(figsize=(15,15))

    ax = sns.scatterplot(x="xgb preds", y="Modelled Final Protein", data=df)
    ax.plot([0, 500], [0, 500], linewidth=2, c ="r")
    #ax.set_xticks(ticks= [0, 4, 9], labels= [1,5, 10], fontsize = fontsize )
    #ax.set_yticks(ticks= [0, 250, 500], labels= [0, 250, 500], fontsize = fontsize )
    
    ax.set_xlabel("xgb preds", fontsize = fontsize)
    ax.set_ylabel("Modelled Final Protein", fontsize = fontsize)


    fig.suptitle("xgb preds vs Modelled Final Protein", fontsize = fontsize)
    fig.tight_layout()


    ##### Save fig

    # make directory for sticking the output in
    if os.path.isdir(save_path) == False:
        os.mkdir(save_path, mode=0o777)

    #navigate to tidy_data_files
    os.chdir(save_path)

    plt.savefig("xgb_preds_modelled_protein.png")

x_preds_plot(x_pred, "tmp/")