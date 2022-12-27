import numpy as np
import pandas as pd
from scipy.integrate import odeint

# Define helper Hill function
def MM(c,K):
    return(c/(K+c))

def model(y, t, params):
    NTP = y[0]
    NXP = y[1]
    nt = y[2]
    Ppi = y[3]
    ATr = y[4]
    a = y[5]
    T = y[6]
    A = y[7]
    CP = y[8]  
    C = y[9]
    CTL = y[10]


    d = params[0]
    CTX = params[1]
    CRS = params[2]
    CEN = params[3]    
    kTX = params[4]
    kTL = params[5]
    kRS = params[6]
    kEN = params[7]
    kntdeg = params[8]
    kTLdeg = params[9]
    Ktxd = params[10]
    KtxN = params[11]
    Ktln = params[12]
    Ktla = params[13]
    KtlNt = params[14]
    KrsA = params[15]
    Krst = params[16]
    KrsN = params[17]
    Kenc = params[18]
    KenN = params[19]
    nT = params[20]
    nA = params[21]

    VTX = kTX*CTX*MM(d,Ktxd)*MM(NTP,KtxN)
    VTL = kTL*CTL*MM(nt,Ktln)*MM(nT/nA*ATr,Ktla)*MM(NTP,KtlNt)
    VRS = kRS*CRS*MM(A,KrsA)*MM(nT/nA*T,Krst)*MM(NTP,KrsN)
    VEN = kEN*CEN*MM(CP,Kenc)*MM(NXP,KenN)

    derivs = [
        -VTX - 2*VTL -VRS + VEN,
        2*VTL + VRS + VEN,
        VTX - kntdeg*nt,
        VTX,
        -VTL + VRS,
        VTL ,
        VTL - VRS,
        -VRS,
        -VEN,
        VEN,
        -kTLdeg*CTL
    ]
    return(derivs)



def solvePURE(TMAX, NSTEPS, initial_concs):

    # Parameters from Mavelli paper


    keysPar = ['d','CTX','CRS','CEN','kTX','kTL','kRS','kEN','kntdeg',
               'kTLdeg','Ktxd','KtxN','Ktln','Ktla','KtlNt','KrsA',
               'Krst','KrsN','Kenc','KenN','nT','nA']

    valuesPar = [1.7e-3,0.1,0.16,0.08,1.67,0.085,6.2,100,7.92e-5,
                1.86e-4,5e-3,80,226,10,10,23,
                0.7,200,200,40,46,20]
    


    # setting up for running the loop
    y0 = np.array(initial_concs)
    
    params = np.array(valuesPar)

    time = np.linspace(0,TMAX,NSTEPS)

    sol = odeint(model, y0, time, args=(params,)) # Scipy solver

    return sol, time




def Conduct_Modelling(proposed_grid_array, TargetSpecies, initial_concs_dict, TMAX, NSTEPS):
    """  Iterates over the rows in the matrix and uses the compositions to update the inital concs. Then solves the model and returns a list of endpoint protein concentrations"""

    # initialise an empty list to populate with protein concs and use to make the column
    endpoint_protein_concentrations = []

    # iterate over the compositions
    for composition in proposed_grid_array:
        
        # Make a copy of the original array
        original_concentrations = list(initial_concs_dict.values())
        updated_concentrations = original_concentrations.copy()
        
        # iterate over each species in the array and update it
        for new_conc, key in zip(composition, TargetSpecies):
            
            # get the array index from the dictionary
            index = TargetSpecies[key]["initial_condition_vector_index"]
            # UPDATE
            updated_concentrations[index] = new_conc
            
        # conduct the modelling!
        sol, time = solvePURE(TMAX, NSTEPS, updated_concentrations)
        
        # last time point and just get the polymerised protein at index 5
        endpoint_protein_concentrations.append(sol[-1,:][5])
    


    return endpoint_protein_concentrations


