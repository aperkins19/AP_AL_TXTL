from scripts.grid_generators import *
from scripts.neural_networks import *
from scripts.MLP_definitions import *
from scripts.plotting_functions import *
from scripts.data_scaler import *
from models.MavelliPURE import *

from tensorflow.keras.losses import MeanSquaredLogarithmicError


import numpy as np
import pandas as pd
import os
from scipy.integrate import odeint


import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import sklearn



# Defining global parameters

Grid_Path = "./datasets/grids/"
Plot_Path = "./Plots/"


NUMBER_OF_ROUNDS = 20


# the grid size for each composition set to be passed into the model to simulate real data
in_vitro_grid_size = 100
exploitation_exploration_ratio = 0.9
exploitation_number = int(in_vitro_grid_size * exploitation_exploration_ratio)
exploration_number = int(in_vitro_grid_size - exploitation_number)

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


TargetSpeciesKeys = list(TargetSpecies.keys())

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






# Round 0 - Generating the initial grid
# Returns an initial grid and the array_to_avoid - the array of already tested compositions
# Functions in the grid_generators.py script

initialgrid = generate_initial_grid(in_vitro_grid_size, max_concs_array, PermissiblePercentagesOfMaxConcs, NumOfTargetSpecies)

initialgrid_modelled_df = pd.DataFrame(initialgrid, columns=TargetSpeciesKeys)

# Conduct modelling
# Look at MavelliPURE.py for the function
endpoint_protein_concentrations = Conduct_Modelling(initialgrid, TargetSpecies, initial_concs_dict, TMAX, NSTEPS)
initialgrid_modelled_df["Modelled Final Protein"] = endpoint_protein_concentrations


#### Scale data for machine learning
##### function in data.scaler.property


# add the round #
initialgrid_modelled_df['Round #'] = 0

# save the initial grid
initialgrid_modelled_df.to_csv(Grid_Path+"initial_grid_mM.csv", index=None)

# Initialise the MasterGroundTruth with initial grid.
initialgrid_modelled_df.to_csv(Grid_Path+"/Ground_Truths/MasterGroundTruth.csv", index=None)



mae_list = []

for round_num in range(1, NUMBER_OF_ROUNDS):

    print("Round #: "+str(round_num))


    # read in master data set
    Current_Total_Ground_Truth_Df = pd.read_csv(Grid_Path+"/Ground_Truths/MasterGroundTruth.csv")

    # do some basic trimming to get just inputs and outputs
    Current_Total_Ground_Truth_Df_trimmed = Current_Total_Ground_Truth_Df[TargetSpeciesKeys+["Modelled Final Protein"]].copy()

    #### Scale data for machine learning
    ##### function in data.scaler.property

    # only scales the inputs between 0-1
    Current_Total_Ground_Truth_Df_scaled = Initial_Scale_Data_Min_Max(Current_Total_Ground_Truth_Df_trimmed, TargetSpecies)


    # Neural Networking

    ###### Divide data

    # separate evaluation test set. Top 20% of records. Which are then excluded from the training data

    top_20_index = int(Current_Total_Ground_Truth_Df_scaled.shape[0] / 5)

    # use the index to slice the dataframe
    train_data = Current_Total_Ground_Truth_Df_scaled.iloc[top_20_index:,:].copy().reset_index(drop=True)
    test_data = Current_Total_Ground_Truth_Df_scaled.iloc[:top_20_index, :].copy().reset_index(drop=True)


    ###### training

    # Select the input data using the TargetSpecies.keys
    x_train = train_data[TargetSpeciesKeys].values

    # produces np array of 1D
    y_train = train_data["Modelled Final Protein"].values


    # defines the input and output nodes of the neural network based on the data shape
    input_nodes = len(TargetSpecies)
    num_output_nodes = 1

    # Build the ensemble of MLPs - neural_networks.py for function and definiition
    MLP_ensemble = generate_MLP_ensemble(input_nodes, num_output_nodes, MLP_Settings_Dictionary)


    # iterate over and fit.
    for model in MLP_ensemble:

        # Fit!
        model.fit(x_train, y_train, epochs=50, validation_split=0.2)


    # Evaluate all of the models and choose the best one

    # generate test data
    x_test = test_data[TargetSpeciesKeys].values
    y_test = test_data["Modelled Final Protein"].values


    R2_Scores = []
    for MLP in MLP_ensemble:

        # calculate the y predictions for the current model using the x_test inputs
        # compare them to the actual y_tests using the R squared metric.
        # append said metric to the list
        R2_Scores.append(sklearn.metrics.r2_score(y_test, MLP.predict(x_test)))



    # get the index of the greatest R2 score and use it to index the best MLP in the ensemble.
    # set it to "best MLP"

    Best_MLP = MLP_ensemble[R2_Scores.index(max(R2_Scores))]


    ########## Simulate compositions

    # this is the array of compositions that have already been sampled
    array_to_avoid = x_train

    # returns array of inputs of in vitro concs
    simulate_input = generate_random_grid(array_to_avoid, max_concs_array, in_silico_random_grid_size, NumOfTargetSpecies, PermissiblePercentagesOfMaxConcs)

    # Scale down to 0-1
    simulate_input_scaled = Just_Input_Scale_Data_Min_Max(simulate_input)

    # construct Dataframe and initialise with the actual compositons
    # this will then be populated with the predictions for each model
    Random_compositions_predictions_and_reality_DF = pd.DataFrame(simulate_input, columns = TargetSpeciesKeys)

    ########################### Exploitation
    # just do the best model first to get exploitation compositions
    # perform predictions and drop the extra dimension from the numpy object
    Best_simulate_predictions_array = Best_MLP.predict(simulate_input_scaled).reshape(-1)
    # add 
    Random_compositions_predictions_and_reality_DF["Pred for Best Model"] = Best_simulate_predictions_array

    # Sort predictions to get top predicted perfomers.
    Best_Simulated_Predictions = Random_compositions_predictions_and_reality_DF.sort_values(by ="Pred for Best Model", ascending=False).copy()
    Best_Simulated_Predictions.reset_index(drop=True, inplace=True)

    # get top performers and from the predictions - exploitation_number is number of compositions in the next set devoted to exploitation
    Best_Simulated_Predictions = Best_Simulated_Predictions.iloc[:exploitation_number,:]




    ############################ now for the exploration

    # first I'm going to see what each model predicts for each train composition
    # those predictions will be saved in the df column wise under the model's name
    # the names will also be saved to make slicing easier later.


    #### could use train?



    model_name_list = []

    Exploration_test_data_model_preds_df = pd.DataFrame(simulate_input, columns=TargetSpeciesKeys)

    # iterate over and PREDICT!.
    for model in MLP_ensemble:

        # perform predictions and drop the extra dimension from the numpy object
        test_simulated_predictions_array = model.predict(simulate_input_scaled).reshape(-1)

        # get the model name and add to the list
        model_name = "Pred for Model #: " + model.name
        model_name_list.append(model_name)

        # add 
        Exploration_test_data_model_preds_df[model_name] = test_simulated_predictions_array

    # Now generate the apparent difference between the predictions. Use the StdDeviation to begin with.
    # add to the df under.....

    Exploration_test_data_model_preds_df['Mean'] = Exploration_test_data_model_preds_df[model_name_list].mean(axis=1)
    Exploration_test_data_model_preds_df['StdDev'] = Exploration_test_data_model_preds_df[model_name_list].std(axis=1)

    # Sort df by STD to get the most undecided compositions
    Exploration_test_data_model_preds_df = Exploration_test_data_model_preds_df.sort_values(by ="StdDev", ascending=False).copy()
    Exploration_test_data_model_preds_df.reset_index(drop=True, inplace=True)

    # get top performers and from the predictions - exploitation_number is number of compositions in the next set devoted to exploitation
    Most_undecideded_Compositions = Exploration_test_data_model_preds_df.iloc[:exploration_number,:]


    #### Now build the proposed_plate_df
    Most_undecideded_Compositions_just_comps = Most_undecideded_Compositions[TargetSpeciesKeys]
    Best_Simulated_Predictions_just_comps = Best_Simulated_Predictions[TargetSpeciesKeys]

    # add the exploitation and exploration samples together and reset the index
    proposed_plate_df = pd.concat([Best_Simulated_Predictions_just_comps, Most_undecideded_Compositions_just_comps], axis=0)
    proposed_plate_df.reset_index(inplace=True, drop=True)

    
    #save
    proposed_plate_df.to_csv(Grid_Path+"/Proposed_Grids/Proposed_Grid_for_round_"+str(round_num)+".csv", index=None)

    ############### Perform Modelling

    ### prepare input data. Produce NP Matrix
    proposed_plate_matrix = proposed_plate_df.to_numpy()

    ####### Conduct modelling!
    endpoint_protein_concentrations = Conduct_Modelling(proposed_plate_matrix, TargetSpecies, initial_concs_dict, TMAX, NSTEPS)
    proposed_plate_df["Modelled Final Protein"] = endpoint_protein_concentrations

    # add the round #
    proposed_plate_df['Round #'] = round_num


    # Run the predictions on the proposed plate to generate the STD plots.
    # Function in neural_networks.py
    evaluate_model(proposed_plate_df, MLP_ensemble, TargetSpeciesKeys, round_num)


    #append to Master Ground Truth
    Current_Total_Ground_Truth_Df = pd.concat([Current_Total_Ground_Truth_Df, proposed_plate_df], axis=0)


    # Initialise the MasterGroundTruth with initial grid.
    Current_Total_Ground_Truth_Df.to_csv(Grid_Path+"/Ground_Truths/MasterGroundTruth.csv", index=None)




#Final plotting

stripplot_over_rounds("./datasets/grids/Ground_Truths/MasterGroundTruth.csv", "/app/datasets/plots/", "experiment_rounds_box_plots.png")


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
