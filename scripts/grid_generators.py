
import numpy as np
import pandas as pd



def print_all_df(df):
    # Permanently changes the pandas settings
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', -1)
    
    # All dataframes hereafter reflect these changes.
    print(df)
    
    print('**RESET_OPTIONS**')
    
    # Resets the options
    pd.reset_option('all')
    



def present_in_array(this_sample_conc, array_to_avoid):
# Checks whether composition sample is already present in array
# present,i = present_in_array(this_sample_conc,x_data)
    present = False

    #defining this variable as global allows it to persist outside the function
    ind = 0

    # iterate over the array presenting each sample slice
    # sample slice = array_to_avoid[i,:]
    for idx, array in enumerate(array_to_avoid):

        # if said sample slice is equal to this proposed sample concentration composition then:
        # sent present equal to true, save the index and return both

        # if the arrays are the same
        if (array == this_sample_conc).all():

            present = True 
            ind = idx

    return present, ind



def generate_random_grid(array_to_avoid, max_concs_array, random_grid_size, NumOfTargetSpecies, PermissiblePercentagesOfMaxConcs):

    # a counter for tracking how many samples have been accepted
    accepted_counter = 0

    # initialise list for holding the accepted samples
    accepted_sample_list = []
        
    # the process repeats until the total number is full
    while accepted_counter < random_grid_size:
        
        ### 1) Make random sample (an array)
        # Generate an array of size: 1 row and as many columns as the number of species.
        # populate it randomly selected conc %s
        this_sample = np.random.choice(PermissiblePercentagesOfMaxConcs, size=(1, NumOfTargetSpecies)).reshape(-1)
        
        
        
        
        
        # convert to real concs
        this_sample = np.multiply(this_sample, max_concs_array)###################################################################

        # ask if this sample is present in the array to ignore
        present, index = present_in_array(this_sample, array_to_avoid)

        # if not already present
        if present == False:

            # Add sample to new ALarray as well as array_to_avoid
            accepted_counter +=1

            #  first add it.
            accepted_sample_list.append(this_sample)
            
            # add also to the array to avoid...
            array_to_avoid = np.vstack((array_to_avoid, this_sample))

        else:
            pass

    # build np array from list
    ALarray = np.array(accepted_sample_list)

    return ALarray





def generate_initial_grid(grid_size, max_concs_array, PermissiblePercentagesOfMaxConcs, NumOfTargetSpecies):
    """ Master function for generating the initial grid """

    max_concentration_fraction = max(PermissiblePercentagesOfMaxConcs)
    min_concentration_fraction = min(PermissiblePercentagesOfMaxConcs)

    ###  1) Create an array containing all maximum compositions except one which is the minimum.

    # create a 1D array full of 1.0 which is the length of how many species are being varied.
    # Scale the 1.0 by the maximum to generate an array of maximum fractions.

    allmax = np.ones(NumOfTargetSpecies) * max_concentration_fraction 

    # store the max array as the first element in the final array.
    allmaxonelow = allmax.copy()

    # create the diagonal array
    # use the index of the length of the array to walk across the array, setting that element to the minimum.
    # And then append
    for idx in range(0, NumOfTargetSpecies):

        this_sample = allmax.copy()

        this_sample[idx] = min_concentration_fraction

        # stack the array
        allmaxonelow = np.vstack((allmaxonelow,this_sample))

    ### 2) Now do the inverse - all min but one which is max
    # all minimum
    allmin = np.ones(NumOfTargetSpecies) * min_concentration_fraction
    # init 2d array
    allminonehigh =  allmin.copy()

    for idx in range(0, NumOfTargetSpecies):
        this_sample = allmin.copy()
        this_sample[idx] = max_concentration_fraction

        # stack
        allminonehigh = np.vstack((allminonehigh, this_sample))

    ### 3) Now combine the two high & low arrays

    # we will pass this into to the random generator which will use it to ensure that compositions are not duplicated
    high_and_low = np.vstack((allmaxonelow, allminonehigh))




    # multiple columnwise with the max_concs_away
    high_and_low = np.multiply(high_and_low, max_concs_array) #####################################################





    # initialise array to avoid with high_and_low
    array_to_avoid = high_and_low.copy()

    ### 4) Now generate the randomally sampled array

    # the minusing and adding is to account for the onehot / one cold matrix.
    # the maths is correct.
    random_grid_size = grid_size - high_and_low.shape[0]

    # generate random grid
    randomgrid = generate_random_grid(array_to_avoid, max_concs_array, random_grid_size, NumOfTargetSpecies, PermissiblePercentagesOfMaxConcs);

    # now concatenate all the grids to produce the initial grid
    initialgrid = np.vstack((high_and_low, randomgrid))

    # finally multiply 

    return initialgrid