import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def Initial_Scale_Data_Min_Max(freshly_modelled_df, round_num, TargetSpecies):
    """This function takes in a freshly modelled df, separates the inputs from the outputs. scales both column wise to between 0-1.
     Reassembles the df and adds a round # column. Returns df"""

    #print_all_df(initialgrid_modelled_df.head())
    #### Right this is the risky bit. Scaling the initialgrid_modelled_df ready for ML. Between 0-1.
    # I'm going to treat the reagents and product columns differently.
    # I'm going to divide each of the columns by their biggest number

    # this should stay the same for the regents column but will change for the product column as the max gets bigger each round.. hopefully


    # now split
    X_ = freshly_modelled_df[list(TargetSpecies.keys())].copy()
    X_ = X_.to_numpy()

    Y_ = freshly_modelled_df["Modelled Final Protein"].copy()
    Y_ = Y_.to_numpy().reshape(-1,1)

    # now transform.
    scaler = MinMaxScaler(feature_range=(0,1))

    # only transform the inputs
    X_ = scaler.fit_transform(X_)
    #Y_ = scaler.fit_transform(Y_.reshape(-1,1))

    # add Y_ back column wise
    X_Y_ = np.hstack([X_, Y_])

    # get the column names and build df
    newly_scaled_df = pd.DataFrame(X_Y_, columns=  list(TargetSpecies.keys())+["Modelled Final Protein"]  )

    # add the round #
    newly_scaled_df['Round #'] = round_num

    return newly_scaled_df


def Just_Input_Scale_Data_Min_Max(freshly_generated_input_array):

    # now transform.
    scaler = MinMaxScaler(feature_range=(0,1))

    # only transform the inputs
    freshly_scaled_input_array = scaler.fit_transform(freshly_generated_input_array)

    return freshly_scaled_input_array
