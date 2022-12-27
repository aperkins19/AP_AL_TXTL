# Solve model using ensemble parameters
# The reference for the model is Horvath et al. 2020
# paper: https://doi.org/10.1016/j.mec.2019.e00113
# github: https://github.com/varnerlab/Kinetic-CFPS-Model-Publication-Code
# modified by N Laohakunakorn
# University of Edinburgh 2020


# import the guts of the model 

include("./model/scripts/DataFile.jl");
include("./model/scripts/MassBalances.jl");


using Sundials
using Plots
using DataFrames
using CSV
using ProgressBars
using DifferentialEquations

# Global settings: paths and plot
PATH_OUT = "./models/horvath/results/"
PATH_NET = "./models/horvath/model/network/Network.dat"
PARAM_PATH = "./models/horvath/model/params"
FN = "CAT_timeseries.pdf" # Output filename

# Important: select which parameter sets to solve with
#PARAMS_TO_SOLVE = collect(1:100) # all parameter sets
#PARAMS_TO_SOLVE = [101] # best fit parameter set

Plots.pyplot()
fntsm = Plots.font("sans-serif", pointsize=round(10.0))
fntlg = Plots.font("sans-serif", pointsize=round(12.0))
default(titlefont=fntlg, guidefont=fntlg, tickfont=fntlg, legendfont=fntsm)

# 1. Load model stoichiometry, initial
# conditions, and parameters.

TSTART = 0.0
TSTOP = 3.0
Ts = 0.1

# Create the data_dictionary and initialise it with network stoichiometric matrix
# and all params to 0 (rates), 1 (initial conds, saturation, order),
# and 0.1 (gain).

# Then populate with parameters from file.


data_dictionary = DataFile(TSTART,TSTOP,Ts,PATH_NET)

# Import initial conditions
initial_condition_vector = vec(broadcast(abs, float(open(readdlm,PARAM_PATH*"/initial_condition.dat"))));
data_dictionary["INITIAL_CONDITION_ARRAY"] = initial_condition_vector;
# Import rate constants
rate_constant_vector = vec(broadcast(abs, float(open(readdlm,PARAM_PATH*"/rate_constant.dat"))));
data_dictionary["RATE_CONSTANT_ARRAY"] = rate_constant_vector;
# Import saturation constants
saturation_constant_array = broadcast(abs, float(open(readdlm,PARAM_PATH*"/saturation_constant.dat")));
data_dictionary["SATURATION_CONSTANT_ARRAY"] = saturation_constant_array;
# Import control parameters
control_constant_array = broadcast(abs, float(open(readdlm,PARAM_PATH*"/control_constant.dat")));
data_dictionary["CONTROL_PARAMETER_ARRAY"] = control_constant_array;


# Ready to solve
# 2. Solve ODEs, plot and save results
# selected parameter sets defined at top of script

@info "Solving"
# Set simulation timespan and initial conditions
tspan = (TSTART,TSTOP);
initial_condition_vector = data_dictionary["INITIAL_CONDITION_ARRAY"];

# Use sundials to solve fluxes.
prob = ODEProblem(MassBalances,initial_condition_vector,tspan,data_dictionary);

@info "pre solve"

# uses Sundials
sol = solve(prob,CVODE_BDF(),abstol=1e-12,reltol=1e-12); # very small tolerances required

@info "post solve"
df = DataFrame(sol);  # df is 352 rows x 2508 columns
CSV.write(string(PATH_OUT,"solution.csv"), df) # uncomment if you want to save


