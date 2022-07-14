# Intro

 This is an active learning implementation for optimising CFPS based on Borkowski 2020.



# Docker for Python and Jupyter with GPU-leverage

# Prerequisites

To be honest it took me ages to set this up so not entirely sure. However the Nvidia driver, the Nvidia toolbox were installed. I also installed a load of stuff and changed some settings using a Ubuntu instance. I followed this tutorial but also did a load of other stuff in the dark - sorry!

https://www.youtube.com/watch?v=PdxXlZJiuxA

Dockerfile adapted from Tensorflow

# Usage

```bash
git clone https://https://github.com/aperkins19/AP_AL_TXTL.git
```

## Define Python Packages in requirements.txt

## Build Image

```bash
docker build -t al_txtl_python_gpu .
```


## Run Container

#### Windows:
#### GPU
```bash
docker run -p 8883:8888 --gpus all  -v "%CD%":/app --name al_txtl_python_gpu al_txtl_python_gpu
```
#### No GPU
```bash
docker run -p 8883:8888 -v "%CD%":/app --name al_txtl_python_gpu al_txtl_python_gpu
```
#### Linux:
#### No GPU

```bash
docker run -p 8883:8888 -v $(pwd):/app --name al_txtl_python_gpu al_txtl_python_gpu
```