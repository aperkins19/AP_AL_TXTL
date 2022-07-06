import numpy as np
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
