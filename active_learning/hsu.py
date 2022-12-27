from scripts.grid_generators import *
from scripts.plotting_functions import *
from scripts.data_scaler import *

import numpy as np
import pandas as pd

import subprocess


# Defining global parameters

Grid_Path = "./datasets/grids/"
Plot_Path = "./Plots/"

Verbose_Toggle = False

# the grid size for each composition set to be passed into the model to simulate real data
in_vitro_grid_size = 100

in_silico_random_grid_size = 2000

# Modelling Parameters

# Time for modelling to run for
# 1 hour
TMAX = 6*60*60

# number of increments # 1 second
NSTEPS = TMAX


# Experimental Parameters

# These are the species and their concentrations which were published in the Mavelli PURE model.
# These are used as the seed for updating the concentrations with each proposed composition.

# This dictionary defines the Species to be perturbed

TargetSpecies = {
                     "Ribosomes"  : {"Look_Up": "RIBOSOME",
                                     "initial_condition_vector_index" : 76,
                                     "max_conc_mM" : 0.5},
                     "RNAP"       : {"Look_Up": "RNAP",
                                     "initial_condition_vector_index" : 73,
                                     "max_conc_mM" : 0.3},
                     "Acetyl_CoA" : {"Look_Up": "M_accoa_c",
                                     "initial_condition_vector_index" : 11,
                                     "max_conc_mM" : 1}
}

TargetSpeciesKeys = list(TargetSpecies.keys())  


# this list defines the fractions of the max concentrations of each species which are permissible.
# e.g. 0.1 x 1500 mM =  150 mM
PermissiblePercentagesOfMaxConcs = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

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
initialgrid_modelled_df.to_csv(Grid_Path+"initial_grid_mM_pre_model.csv", index=None)

# import horvath seed_initial_condition.dat
seed_initial_condition = np.genfromtxt('models/horvath/model/params/seed_initial_condition.dat',
                     skip_header=0,
                     skip_footer=1,
                     names=True,
                     dtype=None,
                     delimiter=' ')

# due to the weird voids, this pulls out the floats and builds 1d numpy array
seed_initial_condition_list = []
for i in seed_initial_condition:
    seed_initial_condition_list.append(i[0])
seed_initial_condition = np.array(seed_initial_condition_list)


### l
# iterate over the compositions
for composition in initialgrid:

    print(composition)
    
    # Make a copy of the original seed_initial_condition array
    updated_concentrations = seed_initial_condition.copy()
    
    # iterate over each species in the array and update it
    for new_conc, key in zip(composition, TargetSpecies):
        
        # get the array index from the dictionary
        index = TargetSpecies[key]["initial_condition_vector_index"]
        # UPDATE
        updated_concentrations[index] = new_conc

    # save updated concentrations to initial_condition.dat
    np.savetxt("initial_condition.dat",updated_concentrations,delimiter=",")
    
    print("running julia model..")
    process = subprocess.Popen(["julia", "models/horvath/run_model.jl"])
    process.wait()
    print("modelling done")

    #print(updated_concentrations)
    break





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