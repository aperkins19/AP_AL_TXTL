# Docker for Python and Jupyter with GPU-leverage

# Prerequisites

To be honest it took me ages to set this up so not entirely sure. However the Nvidia driver, the Nvidia toolbox were installed. I also installed a load of stuff and changed some settings using a Ubuntu instance. I followed this tutorial but also did a load of other stuff in the dark - sorry!

https://www.youtube.com/watch?v=PdxXlZJiuxA

Dockerfile adapted from Tensorflow

# Usage


`git clone https://github.com/aperkins19/Docker_GPU_python_datascience_jupyter_bespoke_packages.git`

## Define Python Packages in requirements.txt

## Build Image


`docker build -t python_gpu .`


## Run Container


`docker run -p 8883:8888 --gpus all  -v "%CD%":/src --name python_gpu python_gpu`

