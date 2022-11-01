
import models.NiessArnold.PUREV1 as TL 



from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tellurium as te
import models.sbmlIO as sio

def Run_NiessPURE():
    # 30-residue peptide
    r = te.loada('./NiessArnold/PUREV1') # from Antimony file
    odes = sio.getODEsFromModel(r)
    speciesIds, speciesValues, parameterIds, parameterValues, derivatives = sio.parseODEs(r,odes)
    sio.writePython(speciesIds,speciesValues,parameterIds,parameterValues,derivatives,'./NiessArnold/', 'PUREV1.py')
    print(odes)     




    # assume dilution factor 
    dil = 0.05 # same as Niess

    # All concentrations in uM
    Cgtp = 2000; # arbitrary = PURE
    Cgdp = 0; 
    Cgmp = 0;
    Catp = 2000; # arbitrary = PURE
    Camp = 0; 
    Cutp = 1000; # arbitrary = PURE
    Cctp = 1000; # arbitrary = PURE
    Cump = 0;
    Ccmp = 0;
    Cefg = 31.9*dil;
    Cefggdp = 0;
    Cefggtp = 1e-9; # small value div by zero error
    Ceftu = 236.69*dil;
    Ceftugdp = 0;
    Ceftugtp = 1e-9; # small value div by zero error
    Ct3 = 1e-9; # small value div by zero error
    Cif = 10*dil; #// approximate
    Cifgtp = 0;
    Cr30SIF = 0;
    Cr70SIC = 0;


    Caa = 300*20; # arbitrary = PURE
    Caat = 0; 
    Ct = 7.78; # arbitrary, average of all species from Dong et al. 1996 J Mol Biol
    CfmetT = 100.0; # arbitrary
    Cpi = 0;
    Cppi = 0;

    Cribo70s = 0; # uM
    Cribo30s = 39.82*dil;
    Cribo50s = 39.82*dil;
    Cmrna = 1e-9; # small value div by zero error
    Cp = 0;

    C1 = 0; 
    C2 = 0;
    C3 = 0;
    C4 = 0;
    C5 = 0;
    C6 = 0;
    C7 = 0;
    C8 = 0;
    C9 = 0;
    C10 = 0;
    C11 = 0;
    C12 = 0;
    C13 = 0;
    C14 = 0;
    C15 = 0;
    C16 = 0;
    C17 = 0;
    C18 = 0;
    C19 = 0;
    C20 = 0;
    C21 = 0;
    C22 = 0;
    C23 = 0;
    C24 = 0;
    C25 = 0;
    C26 = 0;
    C27 = 0;
    C28 = 0;
    C29 = 0;
    C30 = 0;
    C31 = 0;
    C32 = 1e-9; # small value div by zero error

    y0 = np.array([Cgtp,Cgdp,Cgmp,Catp,Camp,Cribo70s,Cmrna,Cp,Cefg,Cefggdp,Cefggtp,Ceftu,
                    Ceftugdp,Ceftugtp,Ct3,Cif,Cifgtp,Cribo30s,Cribo50s,Cr30SIF,
                    Cr70SIC,CfmetT,Caa,Caat,Ct,Cpi,Cppi,Cutp,Cctp,Cump,Ccmp,C1,C2,C3,C4,C5,
                    C6,C7,C8,C9,C10,C11,C12,C13,C14,C15,
                    C16,C17,C18,C19,C20,C21,C22,C23,C24,C25,C26,C27,C28,C29,C30,C31,C32])
    params = np.array(TL.valuesPar) # Take parameters from model
    params[24] = 0.3 # aa/s for PURE
    params[49] = 0.00001 # s^-1 for PURE mRNA degradation
    params[42] = 0.01 # uM/s^-1 for PURE T7 transcription Vmax
    params[44] = 1*1e-3 # nM template DNA

    TMAX = 6*60*60 # s
    NSTEPS = 1000000
    time = np.linspace(0,TMAX,NSTEPS)
    sol = odeint(TL.model, y0, time, args=(params,)) # Scipy solver

    sel = [7]
    for i in sel:
        plt.plot(time,sol[:,i], label=TL.keysVar[i]);
    plt.xlabel('t (s)'); plt.ylabel('concs (uM)'); plt.legend() 
    #plt.savefig('./NiessArnold/plots/TLelo10.png',transparent=True)
    plt.show()