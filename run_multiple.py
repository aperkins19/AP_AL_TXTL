from scripts.grid_generators import *
from scripts.neural_networks import *
from models.MavelliPURE import *

from tensorflow.keras.losses import MeanSquaredLogarithmicError


import numpy as np
import pandas as pd
import os
from scipy.integrate import odeint

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler


# Defining global parameters

Grid_Path = "./datasets/grids/"
Plot_Path = "./Plots/"


NUMBER_OF_ROUNDS = 20


# the grid size for each composition set to be passed into the model to simulate real data
in_vitro_grid_size = 100
in_silico_random_grid_size = 1000

# Modelling Parameters

# Time for modelling to run for
# 1 hour
TMAX = 6*60*60

# number of increments # 1 second
NSTEPS = TMAX




# Experimental Parameters

# These are the species and their concentrations which were published in the Mavelli PURE model.
# These are used as the seed for updating the concentrations with each proposed composition.

keysVar = ['NTP','NXP','nt','Ppi','ATr','a','T','A','CP','C','CTL']
valuesVar = [1500,0,0,0,0,0,1.9,300,20000,0,2.2]

initial_concs_dict = dict(zip(keysVar, valuesVar))



# This dictionary defines the Species to be perturbed

TargetSpecies = {
                     "NTP" : {"Look_Up": "NTP", "initial_condition_vector_index" : 0, "max_conc_mM" : 3000},
                     "tRNA" : {"Look_Up" : "T", "initial_condition_vector_index" : 6, "max_conc_mM" : 5},
                     "Amino Acids" : {"Look_Up" : "A", "initial_condition_vector_index" : 7, "max_conc_mM" : 600},
                     "Creatine_Phosphate" : {"Look_Up" : "CP", "initial_condition_vector_index" : 8, "max_conc_mM" : 5000},
                     "TL Enzymes" : {"Look_Up" : "CTL", "initial_condition_vector_index" : 10, "max_conc_mM" : 6}
}


# this list defines the fractions of the max concentrations of each species which are permissible.
# e.g. 0.1 x 1500 mM =  150 mM
PermissiblePercentagesOfMaxConcs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
#PermissiblePercentagesOfMaxConcs = np.arange(start=0.1, stop=10, step=0.1)

# iterate over TargetSpecies and store that max concs in a list which is converted to a np.array
max_concs_array = []
for key in TargetSpecies:
    max_concs_array.append(TargetSpecies[key]["max_conc_mM"])
max_concs_array = np.array(max_concs_array)


# gets the number of species being perturbed
NumOfTargetSpecies = len(TargetSpecies)






# Round 1 - Generating the initial grid
# Returns an initial grid and the array_to_avoid - the array of already tested compositions
# Functions in the grid_generators.py script

initialgrid = generate_initial_grid(in_vitro_grid_size, max_concs_array, PermissiblePercentagesOfMaxConcs, NumOfTargetSpecies)


# Conduct modelling
# Look at MavelliPURE.py for the function
initialgrid_modelled_df, time = Conduct_Modelling(initialgrid, TargetSpecies, initial_concs_dict, TMAX, NSTEPS)

print_all_df(initialgrid_modelled_df.head())
#### Right this is the risky bit. Scaling the initialgrid_modelled_df ready for ML. Between 0-1.
# I'm going to treat the reagents and product columns differently.
# I'm going to divide each of the columns by their biggest number 

# this should stay the same for the regents column but will change for the product column as the max gets bigger each round.. hopefully


# now split
X_ = initialgrid_modelled_df[list(TargetSpecies.keys())].copy()
X_ = X_.to_numpy()

Y_ = initialgrid_modelled_df["Modelled Final Protein"].copy()
Y_ = Y_.to_numpy()


# now transform.
scaler = MinMaxScaler(feature_range=(0,1))
X_ = scaler.fit_transform(X_)
print("")
print("print Y_ transformed")
print(Y_.reshape(-1,1)[0:5,:])
print("")
Y_ = scaler.fit_transform(Y_.reshape(-1,1))
print("")
print("print Y_ transformed")
print(Y_.reshape(-1,1)[0:5,:])
print("")
# add Y_ back column wise
print("")
print("print X_ transformed")
print(X_[0:10,:])




# get the column names and build df
initialgrid_scaled_df = pd.DataFrame(X_Y_, columns=  list(TargetSpecies.keys())+["Modelled Final Protein"]  )

# add the round #
initialgrid_modelled_df['Round #'] = 0
initialgrid_scaled_df['Round #'] = 0

# save the scaled grid
initialgrid_scaled_df.to_csv(Grid_Path+"MasterGroundTruth_scaled.csv", index=None)

# save the initial grid
initialgrid_modelled_df.to_csv(Grid_Path+"initial_grid_mM.csv", index=None)

# Initialise the MasterGroundTruth with initial grid.
initialgrid_modelled_df.to_csv(Grid_Path+"MasterGroundTruth.csv", index=None)



mae_list = []

for round_num in range(1, NUMBER_OF_ROUNDS):


    # read in master data set
    Current_Total_Ground_Truth_Df = pd.read_csv(Grid_Path+"MasterGroundTruth_scaled.csv")

    ######## Neural Networking


    ###### Divide data

    # separate evaluation test set. Top 20% of records. Which are then excluded from the training data

    top_20_index = int(Current_Total_Ground_Truth_Df.shape[0] / 5)

    # use the index to slice the dataframe
    train_data = Current_Total_Ground_Truth_Df.iloc[top_20_index:,:].copy().reset_index(drop=True)
    test_data = Current_Total_Ground_Truth_Df.iloc[:top_20_index, :].copy().reset_index(drop=True)




    ###### training

    # Select the input data using the TargetSpecies.keys
    TargetSpeciesKeys = list(TargetSpecies.keys())
    x_train = train_data[TargetSpeciesKeys].values

    # produces np array of 1D
    y_train = train_data["Modelled Final Protein"].values
    #y_train  = np.expand_dims(y_train, axis=1)

    print("")
    print("y_train")
    print(y_train)

    print("")
    print("Shape of x_train")
    print(x_train.shape)
    print("")

    # defines the input and output nodes of the neural network based on the data shape
    input_nodes = len(TargetSpecies)
    num_output_nodes = 1

    # Build the model - neural_networks.py for function and definiition
    model = define_model(input_nodes, num_output_nodes)


    # Fit!
    model.fit(x_train, y_train, epochs=50, validation_split=0.2)



    ########## Simulate compositions

    # this is the array of compositions that have already been sampled
    array_to_avoid = x_train

    print("")
    print("Shape of array_to_avoid")
    print(array_to_avoid.shape)
    print("")

    simulate_input = generate_random_grid(array_to_avoid, max_concs_array, in_silico_random_grid_size, NumOfTargetSpecies, PermissiblePercentagesOfMaxConcs)


    # perform predictions and drop the extra dimension from the numpy object
    predictions = model.predict(simulate_input).reshape(-1)

    # construct Dataframe and annotate with predictions
    simulate_input_preds = pd.DataFrame(simulate_input, columns = TargetSpeciesKeys)
    simulate_input_preds["Predicted Final Protein"] = predictions


    # Sort predictions to get top predicted perfomers.
    simulate_input_preds = simulate_input_preds.sort_values(by ='Predicted Final Protein', ascending=False)
    simulate_input_preds.reset_index(drop=True, inplace=True)


    # get top performers and from the predictions - in vitro grid size.
    Top_performing_predictions = simulate_input_preds.iloc[:in_vitro_grid_size,:].copy()
    
    #save
    Top_performing_predictions.to_csv(Grid_Path+"grid_round_"+str(round_num)+".csv", index=None)

    Top_performing_predictions = Top_performing_predictions.drop("Predicted Final Protein", axis=1)


    ############### Perform Modelling

    ### prepare input data

    Top_performing_predictions_array = Top_performing_predictions.to_numpy()

    ####### Conduct modelling!
    Top_performing_preds_quantified, time = Conduct_Modelling(Top_performing_predictions_array, TargetSpecies, initial_concs_dict, TMAX, NSTEPS)


    # add the round #
    Top_performing_preds_quantified['Round #'] = round_num

    # save the initial grid
    Top_performing_preds_quantified.to_csv(Grid_Path+"Top_Performing_predictions_"+str(round_num)+".csv", index=None)

    Current_Total_Ground_Truth_Df = pd.concat([Current_Total_Ground_Truth_Df, Top_performing_preds_quantified], axis=0)

    print(Current_Total_Ground_Truth_Df.shape)
    # Initialise the MasterGroundTruth with initial grid.
    Current_Total_Ground_Truth_Df.to_csv(Grid_Path+"MasterGroundTruth.csv", index=None)

    # Add this round of modelled compostions to array_to_avoid
    array_to_avoid = np.vstack([array_to_avoid,Top_performing_predictions_array])


    ########## model evaluation

    # use the index to slice the dataframe
    test_data = Current_Total_Ground_Truth_Df.iloc[:top_20_index, :].copy()




    ###### training

    # Select the input data using the TargetSpecies.keys
    x_test = test_data[TargetSpeciesKeys].values

    # produces np array of 1D
    y_test = test_data["Modelled Final Protein"].values


    # evaluate and save metric
    results = model.evaluate(x_test, y_test, batch_size=128)
    average_mae = results[1]
    mae_list.append(average_mae)



#Final plotting

Current_Total_Ground_Truth_Df = pd.read_csv(Grid_Path+"MasterGroundTruth.csv")


fig = plt.figure(figsize=(10,5))

ax = sns.boxplot(x="Round #", y="Modelled Final Protein", data=Current_Total_Ground_Truth_Df, whis=np.inf, width=0.3)
ax = sns.stripplot(x="Round #", y="Modelled Final Protein", data=Current_Total_Ground_Truth_Df, color=".3")

#ax.set_ylim(0,300)

#fig.suptitle("RFUs of all experiments at "+ str(timepoint) + " mins")
fig.tight_layout()


##### Save fig


path = "/app/datasets/plots/"

# make directory for sticking the output in
if os.path.isdir(path) == False:
    os.mkdir(path, mode=0o777)
    
    
#navigate to tidy_data_files
os.chdir(path)

plt.savefig("experiment_rounds_box_plots.png")




###### mae list



#Final plotting

mae_df = pd.DataFrame({"Round #": range(0,19,1), "Average Mean Squared Error": mae_list})


fig = plt.figure(figsize=(10,5))

ax = sns.barplot(x="Round #", y="Average Mean Squared Error", data=mae_df)


#ax.set_ylim(0,300)

#fig.suptitle("RFUs of all experiments at "+ str(timepoint) + " mins")
fig.tight_layout()


##### Save fig


path = "/app/datasets/plots/"

# make directory for sticking the output in
if os.path.isdir(path) == False:
    os.mkdir(path, mode=0o777)
    
    
#navigate to tidy_data_files
os.chdir(path)

plt.savefig("Average_Mean_Squared_Error_over_rounds.png")




