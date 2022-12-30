# nllab-ActiveLearn
Active learning to determine optimal compositions of a cell-free reaction

This repository demonstrates using active learning to optimise the output of a complex biochemical system such as a cell-free protein synthesis reaction. As the in silico proxy for the reaction, we use a complex genome-scale metabolic model from the Varner lab (original paper by [Horvath et al. 2020](https://doi.org/10.1016/j.mec.2019.e00113)). The active learning algorithm is adapted from [Borokowski et al. 2020](https://doi.org/10.1101/751669). The original metabolic model code is [here](https://github.com/varnerlab/Kinetic-CFPS-Model-Publication-Code), and the original AL code is [here](https://github.com/brsynth/active_learning_cell_free).

### Usage

The script `AL_10rounds.jl` carries out 10 rounds of active learning to maximise the output protein production of the Varner model. It does this by varying the composition of 8 species which correspond to energy solution components, training an ensemble of 25 multi-layer perceptrons, and using these to propose new unique compositions which maximise the weighted sum of exploitation and exploration, as given by the formula `max(means + 1.4*stdevs)`. This process is thus an in silico implementation of the optimisation carried out in [Borokowski et al. 2020](https://doi.org/10.1101/751669).

The Jupyter notebook steps through one round of learning interactively, to demonstrate the various components of the script.

Please note that the most computationally intensive part is the simulation of the model, and thus the generation of training data takes the most time.

With a working local installation of Julia, start up the REPL by running `julia` on the command line. Use the following two commands to call up the package manager and activating the environment: `]` `activate .` (don't forget the dot). Exit the package manager by pressing Backspace, and then run the scripts as required.

Alternatively, run the scripts on our lab Julia Docker container according to the instructions [here](https://github.com/Laohakunakorn-Group/nllab-Dockerfiles).