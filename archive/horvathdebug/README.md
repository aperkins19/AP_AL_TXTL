# nllab-ActiveLearn
Active learning to determine optimal compositions of a cell-free reaction

This repository demonstrates using active learning to optimise the output of a complex biochemical system such as a cell-free protein synthesis reaction. As the in silico proxy for the reaction, we use a complex genome-scale metabolic model from the Varner lab (original paper by [Horvath et al. 2020](https://doi.org/10.1016/j.mec.2019.e00113)). The active learning algorithm is adapted from [Borokowski et al. 2020](https://doi.org/10.1101/751669). The original metabolic model code is [here](https://github.com/varnerlab/Kinetic-CFPS-Model-Publication-Code), and the original AL code is [here](https://github.com/brsynth/active_learning_cell_free).

## Usage

The script `AL_10rounds.jl` carries out 10 rounds of active learning to maximise the output protein production of the Varner model. It does this by varying the composition of 8 species which correspond to energy solution components, training an ensemble of 25 multi-layer perceptrons, and using these to propose new unique compositions which maximise the weighted sum of exploitation and exploration, as given by the formula `max(means + 1.4*stdevs)`. This process is thus an in silico implementation of the optimisation carried out in [Borokowski et al. 2020](https://doi.org/10.1101/751669).

The Jupyter notebook steps through one round of learning interactively, to demonstrate the various components of the script.

Please note that the most computationally intensive part is the simulation of the model, and thus the generation of training data takes the most time.

## Running the AL

### Nadanai's Julia Docker Container

##### Windows
`docker run -p 8887:8888 -it -v "%CD%":/home/jovyan nadanai263/nllab-julia:005`

##### Apple / Linux
`docker run -p 8887:8888 -it -v "$PWD":/home/jovyan nadanai263/nllab-julia:005`

This will spin up his docker container on port 8887, mounted to this directory and open the julia REPL.

### Activating the julia environment (Installing the packages)

Use the following two commands to call up the package manager and activating the environment: `]` `activate .` (don't forget the dot). Exit the package manager by pressing `Backspace`,

### Navigate to the scripts

As you can see in the `docker run` command, the local container was mounted into the /home/jovyan/ directory.  
If you call `pwd()` followed by `readdir()`, you will see that you are in the root of the linux container.  
Navigate into the correct directory with `cd("/home/jovyan/")`.  
Confirm with `readdir()`, you should see all your files.

### Run the .jl scripts

`include("AL_10rounds.jl")`


## Analysis

### Run Nadanai's Julia/Jupyter Docker Container


`docker run -it -v "%CD%":/src -p 8886:8888 --name al_analysis_cntr nadanai263/nllab-jupyter:006`

### Usage

The way it works is by:
a. starting a Docker Container
b. Mounting your current directory ("%CD%") to a directory in the container ("/src") so that files can be shared and moved in and out.
c. starting a jupyter server.

* If it has started correctly, you'll get a url token. Copy the token provided into your brower URL

It should look like this:

`http://127.0.0.1:8888/?token=3c96d2a50decb4302c3e96b87ba7444d286e335d07c478fe`

It should open up a Jupyter File explorer in the directory in your browser.