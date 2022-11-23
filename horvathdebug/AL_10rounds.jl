# AL_full_loop.jl
# Active learning to determine optimal compositions of a cell-free reaction
# Data can come from plate reader/microfluidics experiments
# or be generated in-silico for demonstration purposes

# The in silico model is a genome-scale metabolic model from Horvath et al. 2020
# paper: https://doi.org/10.1016/j.mec.2019.e00113
# github: https://github.com/varnerlab/Kinetic-CFPS-Model-Publication-Code

# The active learning framework is adapted from Borkowski et al. 2020 
# paper: https://doi.org/10.1101/751669.
# github: https://github.com/brsynth/active_learning_cell_free 

# by N. Laohakunakorn (nadanai.laohakunakorn@ed.ac.uk)
# April 2021

using Flux
using Flux.Optimise: update!
using Flux: Chain, Dense, Dropout, @epochs
using CUDA
using Base: @kwdef
using Statistics, LsqFit, DataFrames, CSV, ProgressBars, Serialization
using Plots 
using Sundials
using Random; rng = MersenneTwister(1234);

include("./scripts/AL_helpers.jl");
include("./scripts/AL_solver.jl");

# Directories
OUTPUT = "./models/"
INPUT = "./data/"

# Hyperparameters
@kwdef mutable struct Args
    η::Float64 = 0.001       # learning rate
    batchsize::Int = 99    # batch size
    epochs::Int = 500     # number of epochs
    use_cuda::Bool = true   # use gpu (if cuda available)
end
args = Args()

# GPU 
if CUDA.functional() && args.use_cuda
    @info "Training on CUDA GPU"
    CUDA.allowscalar(false)
    device = gpu
else
    @info "Training on CPU"
    device = cpu
end

# Compositions to vary
NCOMPS = 8
ntp_c_max = 1.5
aa_nglu_c_max = 3.0 
nad_c_max = 2.94
thf_c_max = 1.24
coa_c_max = 1.24
pga_c_max = 4.26
glu_c_max = 340
trna_c_max = 3.2
ALLOWEDCONCS = [0.1, 0.3, 0.5, 1]
maxcomps = [ntp_c_max,aa_nglu_c_max,nad_c_max,thf_c_max,coa_c_max,pga_c_max,glu_c_max,trna_c_max]
speciesidx = [106,118,145,38,10,7,136,99] # relevent indices, defined in DataFiles.jl

# Generate and save initial grid
@info "Generating initial grid"
gridsize = 99
initialgrid = generate_initial_grid(gridsize,NCOMPS,ALLOWEDCONCS)
CSV.write(INPUT*"initial_grid.csv", DataFrame(initialgrid)); 

# Filenames
COMPDATA = ["initial_grid","round1_proposed","round2_proposed",
        "round3_proposed","round4_proposed","round5_proposed",
        "round6_proposed","round7_proposed","round8_proposed","round9_proposed"]
FLUORDATA = ["round1_fluor","round2_fluor","round3_fluor",
        "round4_fluor","round5_fluor","round6_fluor","round7_fluor",
        "round8_fluor","round9_fluor","round10_fluor"]

# Main loop
for round = 1:10

    ROUND = "round"*string(round)
    # Read composition data
    df = CSV.read(INPUT*COMPDATA[round]*".csv", DataFrame, type=Float32);
    x_data = Matrix(df); 
    # size(x_data) # should be (8,99)
    heatmap(x_data, c=:bluegreenyellow,yaxis="composition",axis="ID")
    savefig(OUTPUT*COMPDATA[round]*".pdf")

    # 1. Generate training data for this round
    @info "1. Generating training data"
    # Solve Varner model for initial composition array
    cat_c = []
    for j in ProgressBar(1:size(x_data,2))
        compositions = x_data[:,j]
        cat = solvemodel(compositions,maxcomps,speciesidx)
        push!(cat_c,cat)
    end
    CSV.write(INPUT*FLUORDATA[round]*".csv", DataFrame(reshape(cat_c,(size(x_data,2),1)))); 
    df = CSV.read(INPUT*FLUORDATA[round]*".csv", DataFrame, type=Float32);
    y_data = reshape(Array(df),gridsize); #/0.018560623281234926 # normalise to original maximum
    # size(y_data) # should be (99,)
    heatmap(reshape(y_data,1,length(y_data)), c=:bluegreenyellow,yaxis="fluorescence",axis="ID")
    savefig(OUTPUT*FLUORDATA[round]*".pdf")

    # Use all training data obtained so far

    if round>1
        tmpx=[]
        tmpy=[]
        for idxs = 1:round
            df = CSV.read(INPUT*COMPDATA[idxs]*".csv", DataFrame, type=Float32);
            x_data = Matrix(df); 
            push!(tmpx,x_data)
            df = CSV.read(INPUT*FLUORDATA[idxs]*".csv", DataFrame, type=Float32);
            y_data = reshape(Array(df),gridsize)
            push!(tmpy,y_data)
        end
        ax = tmpx[1]
        ay = tmpy[1]
        for j = 2:round
            ax=hcat(ax,tmpx[j])
            ay=vcat(ay,tmpy[j])
        end
        x_data = ax
        y_data = ay
    else
        x_data = x_data
        y_data = y_data
    end

    # Now train an ensemble MLP model on this data
    # 2. Train all models and generate ensemble:
    @info "2. Training ensemble MLP"
    ensemble_size = 25
    number_of_models = 10
    ensemble_MLP = ensemble_generate(ensemble_size,number_of_models,x_data,y_data,args,device,NCOMPS)

    # Save ensemble
    for idx in 1:ensemble_size
        open(io -> Serialization.serialize(io, cpu(ensemble_MLP[idx])), 
        OUTPUT*ROUND*"/model_"*string(idx)*".jls", "w")
    end

    # Load ensemble
    iter=1
    ensemble_MLP = []
    for idx in 1:ensemble_size
        ms = open(io -> Serialization.deserialize(io), OUTPUT*ROUND*"/model_"*string(idx)*".jls")
        push!(ensemble_MLP,ms)
    end
    # Predict using ensemble 
    plotpreds(ensemble_MLP,x_data,y_data,0)
    savefig(OUTPUT*ROUND*".pdf")


    # 3. Propose new compositions (active learning)
    @info "3. Proposing new compositions"
    # First generate random array (which does not repeat previously-tested compositions)
    grid_size = 20000 
    ALarray = generate_random_grid(x_data,grid_size,NCOMPS,ALLOWEDCONCS); # takes long time for large grids

    # Make predictions on these new compositions
    means,stdevs = ensemblepreds(ensemble_MLP,ALarray);

    # Now choose best compositions according to some metric
    exploitation = 1
    exploration = 1.41
    array_to_max = deepcopy(exploitation*means + exploration*stdevs); # deepcopy so original array unchanged

    # Maximum exploitation
    number_of_proposals = 99
    conditions_list_exploit = []
    for count = 1:number_of_proposals
        indmax = argmax(means)[1]
        push!(conditions_list_exploit,indmax)
        means[indmax] = -1
    end

    # Maximum exploration
    conditions_list_explore = []
    for count = 1:number_of_proposals
        indmax = argmax(stdevs)[1]
        push!(conditions_list_explore,indmax)
        stdevs[indmax] = -1
    end

    # Balanced exploration/exploitation
    conditions_list_balance = []
    for count = 1:number_of_proposals
        indmax = argmax(array_to_max)[1]
        push!(conditions_list_balance,indmax)
        array_to_max[indmax] = -1
    end

    # show best exploitation
 #   ALarray_exploit = ALarray[:,conditions_list_exploit];
 #   a,b = ensemblepreds(ensemble_MLP,ALarray_exploit)
 #   plot(a,yerror=b)

    # show best exploration
    #ALarray_explore = ALarray[:,conditions_list_explore]
    #c,d = ensemblepreds(ensemble_MLP,ALarray_explore)
    #plot(c,yerror=d)

    # show best balance 
    ALarray_balance = ALarray[:,conditions_list_balance];
    e,f = ensemblepreds(ensemble_MLP,ALarray_balance);
    plot(e,yerror=f)
    savefig(OUTPUT*ROUND*"_preds.pdf")
    #heatmap(ALarray_balance, c=:bluegreenyellow,yaxis="composition",axis="ID")
    #savefig(OUTPUT*ROUND*"_proposed.pdf")

    # Save best balance as proposed compositions
    CSV.write(INPUT*ROUND*"_proposed.csv", DataFrame(ALarray_balance)); 


end
