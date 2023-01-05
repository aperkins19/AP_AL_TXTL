# Solve model using ensemble parameters
# The reference for the model is Horvath et al. 2020
# paper: https://doi.org/10.1016/j.mec.2019.e00113
# github: https://github.com/varnerlab/Kinetic-CFPS-Model-Publication-Code

include("./varner_TXTL/scripts/DataFile.jl")
include("./varner_TXTL/scripts/MassBalances.jl")
 

function solvemodel(compositions,maxcomps,speciesidx)

    # Global settings: paths and plot
    PATH_OUT = "./results/"
    PATH_NET = pwd()*"/scripts/varner_TXTL/network/Network.dat"
    PATH_PARAMS = pwd()*"/scripts/varner_TXTL/params"
    FN = "CAT_timeseries.pdf" # Output filename


    # 1. Load model stoichiometry, initial
    # conditions, and parameters.

    TSTART = 0.0
    TSTOP = 3.0
    Ts = 0.1

    # Create the data_dictionary and initialise it with network stoichiometric matrix
    # and all params to 0 (rates), 1 (initial conds, saturation, order),
    # and 0.1 (gain).

    # Then populate with parameters from file.

    PARAMDICT = []
    folders = readdir(PATH_PARAMS*"/Ensemble/") # We have 100 parameter sets

    for folder in folders

        data_dictionary = DataFile(TSTART,TSTOP,Ts,PATH_NET)

        # Import initial conditions
        initial_condition_vector = vec(broadcast(abs, float(open(readdlm,PATH_PARAMS*"/initial_condition.dat"))));
        data_dictionary["INITIAL_CONDITION_ARRAY"] = initial_condition_vector;
        # Import rate constants
        rate_constant_vector = vec(broadcast(abs, float(open(readdlm,PATH_PARAMS*"/Ensemble/"*folder*"/rate_constant.dat"))));
        data_dictionary["RATE_CONSTANT_ARRAY"] = rate_constant_vector;
        # Import saturation constants
        saturation_constant_array = broadcast(abs, float(open(readdlm,PATH_PARAMS*"/Ensemble/"*folder*"/saturation_constant.dat")));
        data_dictionary["SATURATION_CONSTANT_ARRAY"] = saturation_constant_array;
        # Import control parameters
        control_constant_array = broadcast(abs, float(open(readdlm,PATH_PARAMS*"/Ensemble/"*folder*"/control_constant.dat")));
        data_dictionary["CONTROL_PARAMETER_ARRAY"] = control_constant_array;

        push!(PARAMDICT, data_dictionary)
    end

    # Ready to solve
    # 2. Solve ODEs

    this_composition = maxcomps.*compositions

    SEL = [101] # selected parameter sets 1-100 and best
    for idx in SEL
        @info "Solving $(idx)"
        # Set simulation timespan and initial conditions
        tspan = (TSTART,TSTOP);
        initial_condition_vector = PARAMDICT[idx]["INITIAL_CONDITION_ARRAY"];

        # Enter compositions into initial condition vector
        for idx2 in 1:length(speciesidx)
            initial_condition_vector[speciesidx[idx2]] = this_composition[idx2]
        end

#=         # Nucleotides
        for idx3 in [106,109,112,115]
            initial_condition_vector[idx3] = this_composition[1]
        end

        # Amino acids excluding glutamate
        for idx4 in vcat(collect(118:135),137)
            initial_condition_vector[idx4] = this_composition[2]
        end
 =#
        # Use sundials to solve fluxes.
        prob = ODEProblem(MassBalances,initial_condition_vector,tspan,PARAMDICT[idx]);
        sol = solve(prob,CVODE_BDF(),abstol=1e-12,reltol=1e-12);
        df = DataFrame(sol);  # df is 352 rows x 2508 columns

        # make arrays and plot
        t=sol.t
        global x
        x=Array(df[98,:])
    end

    return x[end]
end