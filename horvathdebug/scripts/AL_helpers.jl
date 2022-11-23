# AL_helpers.jl
# Helper functions for active learning script
using Printf
# 1. Functions for random grid generation 

function present_in_array(sample,array)
    # Checks whether composition sample is already present in array
    # present,i = present_in_array(this_sample_conc,x_data)
        present = false
        global ind = 0
        for i=1:size(array)[2]
            if array[:,1] == sample
                present = true 
                ind = i
                break
            end
        end
        return present,ind
    end
    
function generate_random_grid(array_to_avoid,sample_size,NCOMPS,ALLOWEDCONCS)
# Generates random grid while avoiding compositions already defined
# ID and number of compositions hard-coded
    answerSize = 0
    ALarray = []
    while answerSize < sample_size
        # Make random sample
        this_sample = rand(1:length(ALLOWEDCONCS),NCOMPS)
        this_sample_conc = ALLOWEDCONCS[this_sample]

        # Add sample to new ALarray as well as array_to_avoid
        # if not already present
        if ! present_in_array(this_sample_conc,array_to_avoid)[1]
            answerSize+=1           
            reshapedsample = reshape(this_sample_conc,(NCOMPS,1))
            if ALarray == []
                ALarray = reshapedsample
            else
                ALarray = hcat(ALarray,reshapedsample)
            end
            array_to_avoid = hcat(array_to_avoid,reshapedsample)
        else
            @warn("skipping")
        end
    end
    return ALarray
end

# Generate initial plate
function generate_initial_grid(gridsize,NCOMPS,ALLOWEDCONCS)

    minconc = minimum(ALLOWEDCONCS)
    maxconc = maximum(ALLOWEDCONCS)

    # All max but one
    allmax = ones(1,NCOMPS)*maxconc # maximum
    allmaxonelow = allmax
    for idx = 1:NCOMPS
        this_sample = copy(allmax)
        this_sample[idx] = minconc
        allmaxonelow = vcat(allmaxonelow,this_sample)
    end

    # All min but one
    allmin = ones(1,NCOMPS)*minconc # minimum
    allminonehigh = allmin
    for idx = 1:NCOMPS
        this_sample = copy(allmin)
        this_sample[idx] = maxconc
        allminonehigh = vcat(allminonehigh,this_sample)
    end


    grid_size = gridsize-2*(NCOMPS+1)
    randomgrid = generate_random_grid(vcat(allmaxonelow,allminonehigh)',grid_size,NCOMPS,ALLOWEDCONCS);
    initialgrid = hcat(allminonehigh',allmaxonelow',randomgrid)
    return initialgrid
end

# 2. Functions for ML 

function build_model(; inputsize=NCOMPS, nclasses=1)
    return Chain(
            Dense(inputsize, 10, relu),
            Dense(10, 100, relu),
            Dropout(0.5), # originally, 0.3
            Dense(100, 100, relu),
            Dropout(0.5), # originally, 0.3
            Dense(100, 20, relu),
            Dense(20, nclasses,identity))
end

# Evaluation function
function evaluate_one_model(model,x_data)
    if device == gpu
        ypred = model(gpu(x_data)) |> cpu
    elseif device == cpu
        ypred = model(x_data)
    end
    return ypred
end    

# Loss function after complete epoch
function lossfn(data_loader, model, device)
    ls = 0.0f0
    num = 0
    for (x, y) in data_loader
        x, y = device(x), device(y)
        ls += (sum(model(x)'.-y).^2) # transpose to get data in same orientation
        num +=  size(x, 2)
    end
    return ls / num 
end

# R2 evaluation function
function R2(A,B)
    return 1-sum((A .- B).^2)/sum((A .- mean(A)).^2)
end

# Custom train loop
function train!(loss,ps,data,opt,model,device)
    # Trains one epoch
    for (x,y) in data # one epoch of data
        x,y = device(x),device(y) # transfer data to device
        gs = gradient(ps) do # compute gradient of loss
            loss(x,y)
        end
        update!(opt,ps,gs) # call optimiser
    end
    return lossfn(data,model,device)
end

function train_one_model!(x_data,y_data,args,device,NCOMPS)

    train_data = Flux.Data.DataLoader((x_data,y_data),batchsize=args.batchsize,shuffle=true) # Data is now mini-batched ready for training
    model = build_model(inputsize=NCOMPS) |> device
    loss(x,y) = (mean((model(x)'.-y).^2)) # loss function, with correctly transposed data
    ps = Flux.params(model)
    opt = ADAM(args.η)

    # Then train for multiple epochs
 #   @info "Beginning training loop"

    for epoch_idx in 1:args.epochs
        ls = train!(loss,ps,train_data,opt,model,device)
        if epoch_idx%1000 == 0
            @show ls
        end
    end
    ypred=evaluate_one_model(model,x_data)
    score = R2(y_data,ypred[1,:])
 #   @show score
    return model,score,ypred
end

# Now train multiple models
function select_best_model(models_number,x_data,y_data,args,device,NCOMPS)
    trained_model_list = []
    scores = []
    for i = 1:models_number
        model,score,ypred=train_one_model!(x_data,y_data,args,device,NCOMPS)
        push!(trained_model_list,model)
        push!(scores,score)
    end

    best_idx = argmax(scores)
    best_score = scores[best_idx]
    best_model = trained_model_list[best_idx]
    @info(" Best score $(best_score)")
    return best_model
end

# Generate ensemble_MLP 
function ensemble_generate(ensemble_size,number_of_models,x_data,y_data,args,device,NCOMPS)
    ensemble_MLP = []
    for i = ProgressBar(1:ensemble_size)
        bestmodel = select_best_model(number_of_models,x_data,y_data,args,device,NCOMPS)
        push!(ensemble_MLP,bestmodel)
    end
    return ensemble_MLP
end

# Plot ensemble model predictions
function plotpreds(ensemble_MLP,x_data,y_data,y_std_data=0)
    global predictions = zeros(size(x_data,2),length(ensemble_MLP))
    for idx in 1:length(ensemble_MLP)
        ypred=evaluate_one_model(ensemble_MLP[idx],x_data)
        predictions[:,idx] = ypred[1,:]
    end
    means = mean(predictions,dims=2)
    stdevs = std(predictions,dims=2)

    # convert to uM
    means = means*1000
    stdevs = stdevs*1000
    y_data = y_data*1000
    y_std_data = y_std_data*1000

    p = plot(grid=:true,legend=:false,xaxis="Protein at 3h (uM) predicted",
    yaxis="Protein at 3h (uM) actual",
    xtickfontsize=16,ytickfontsize=16,xguidefontsize=16,yguidefontsize=16,ylims=(-1,45),
    size=(600,400))
    if y_std_data==0
        plot!(means,y_data,xerror=stdevs,seriestype=:scatter,legend=:false,xtickfontsize=12,ytickfontsize=12,xguidefontsize=12,yguidefontsize=12)
    else
        plot!(means,y_data,yerror = y_std_data, xerror=stdevs,seriestype=:scatter,legend=:false,xtickfontsize=12,ytickfontsize=12,xguidefontsize=12,yguidefontsize=12)
    end
    # linear y=x fit
    @. modellsq(x, p) = x*p[1]
    p0 = [0.5]
    fit = curve_fit(modellsq, means[:,1], y_data,p0)
    linspace = LinRange(0,maximum(y_data),5)
    linout = linspace*coef(fit)[1]
    plot!(linspace,linout,linewidth=2)
#    plot!(legend=:topleft)
    score = R2(y_data,means);
    formatted = @sprintf("R^2=%.3f",score)
    plot!(title=formatted)
end

# Calculate ensemble model predictions
function ensemblepreds(ensemble_MLP,x_data)
    global predictions = zeros(size(x_data,2),length(ensemble_MLP))
    for idx in 1:length(ensemble_MLP)
        ypred=evaluate_one_model(ensemble_MLP[idx],x_data)
        predictions[:,idx] = ypred[1,:]
    end
    print(predictions)
    means = mean(predictions,dims=2)
    stdevs = std(predictions,dims=2)
    return means,stdevs
end