import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os



# Import the MLP dictionary
from scripts.MLP_definitions import *
from scripts.data_scaler import *

def define_model(input_nodes, num_output_nodes):

    model = keras.Sequential([
    keras.layers.Dense(32, input_shape = (input_nodes,), activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(num_output_nodes)
    ])

    
    model.compile(optimizer=Adam(learning_rate=0.0001),
                loss= "mse",
                metrics=['mae'])

    return model




def generate_MLP_ensemble(input_nodes, num_output_nodes, MLP_Settings_Dictionary):

    """"Iterates over the MLP settings dictionary, builds the MLP and appends it to the list."""
    """ The MLP_Settings_Dictionary is definied in MLP_definitions.py """

    MLP_ensemble = []

    # iterate over each MLP
    for MLP in MLP_Settings_Dictionary.keys():


        # initalise model
        model = keras.Sequential(name=str(MLP))

        # iterate over the layers
        for layer in MLP_Settings_Dictionary[MLP]['layers'].keys():


            # if it's the input layer...
            if "Input" in MLP_Settings_Dictionary[MLP]['layers'][layer]:

                # define the layer
                model.add(keras.layers.Dense(MLP_Settings_Dictionary[MLP]["layers"][layer]["Hidden Nodes"], input_shape = (input_nodes,), activation=MLP_Settings_Dictionary[MLP]["layers"][layer]["activation"]))
            

            # or if it's the output layer
            elif "Output" in MLP_Settings_Dictionary[MLP]['layers'][layer]:
                model.add(keras.layers.Dense(num_output_nodes))

            # any other layer
            else:

                # define the layer
                model.add(keras.layers.Dense(MLP_Settings_Dictionary[MLP]["layers"][layer]["Hidden Nodes"], activation=MLP_Settings_Dictionary[MLP]["layers"][layer]["activation"]))
                
                
                # if this layer has a dropout.......
                if "Dropout" in MLP_Settings_Dictionary[MLP]['layers'][layer]:
                    # get the threshold and use it as an argument
                    model.add(keras.layers.Dropout(MLP_Settings_Dictionary[MLP]['layers'][layer]["Dropout"]))

        # Compile the model
        model.compile(optimizer = Adam(learning_rate=MLP_Settings_Dictionary[MLP]['HyperParams']['learning_rate']),
                loss= MLP_Settings_Dictionary[MLP]['HyperParams']['loss_function'],
                metrics=[MLP_Settings_Dictionary[MLP]['HyperParams']['metrics']])

        # add the model to the list
        MLP_ensemble.append(model)  

    return MLP_ensemble


def evaluate_model(proposed_plate_df, MLP_ensemble, TargetSpeciesKeys, round_num):
    """ Takes the proposed_plate_df complete with actual modelled protein and conducts predictions with model ensemble."""
    """ uses Mean and STD of predictions to generate plots."""


    inputs = proposed_plate_df[TargetSpeciesKeys].values
    ######################################################################## need to check that this worked properly
    inputs_scaled = Just_Input_Scale_Data_Min_Max(inputs)

    #print("")
    #print("inputs")
    #print(inputs[:10])
    #print("")
    #print("inputs scaled")
    #print(inputs_scaled[:10])

    model_name_list = []

    # iterate over and PREDICT!.
    for model in MLP_ensemble:

        # perform predictions and drop the extra dimension from the numpy object
        predictions_array = model.predict(inputs_scaled).reshape(-1)

        # get the model name and add to the list
        model_name = "Pred for Model #: " + model.name
        model_name_list.append(model_name)

        # add 
        proposed_plate_df[model_name] = predictions_array

    # Now generate the apparent difference between the predictions. Use the StdDeviation to begin with.
    # add to the df under.....

    proposed_plate_df['Predicted Mean'] = proposed_plate_df[model_name_list].mean(axis=1)
    proposed_plate_df['StdDev'] = proposed_plate_df[model_name_list].std(axis=1)


    ###### plotting

    fig = plt.figure(figsize=(15,10))

    ax = plt.subplot(1, 1, 1)

    # grabs the real values for posterity.
    x = proposed_plate_df["Predicted Mean"]
    x_err = proposed_plate_df["StdDev"]
    y = proposed_plate_df["Modelled Final Protein"].copy()

    # plot
    plt.errorbar(x, y, xerr = x_err, fmt="o")
    plt.plot(np.linspace(0,600,600), np.linspace(0,600,600),'-r')
    plt.xlabel("Predicted Mean")
    plt.ylabel("Modelled Final Protein")

    fig.suptitle("MLP Performance @ Round "+str(round_num))
    fig.tight_layout()


    ##### Save fig

    comparisonplotPath =  "./datasets/round_comparison_plots/"



    # make directory for sticking the output in
    if os.path.isdir(comparisonplotPath) == False:
        os.mkdir(comparisonplotPath, mode=0o777)


    #navigate to tidy_data_files
    os.chdir(comparisonplotPath)

    plt.savefig(str(round_num)+" ComparisonPlot"+".png")


    #navigate home for neatness
    os.chdir('/app')





