# ParallelGeneticsInspiredNeuralNetworkTuning


## Introduction:

The first analysis for optimizing neural network is shown [here](https://github.com/bhatnags/GeneticsInspiredNeuralNetworkTuning
). Keeping the search space and coding methodology same, the tuning of neural networks using genetic algorithm has been parallelized using Mpi4py. For that two models are considered: 
* multiple deme course grained genetic algorithm model
* hybrid multiple deme course grained genetic algorithm (multiple deme model built on top of multiple demem model) - Island model
Thanks to the research paper [here](https://www.researchgate.net/publication/2362670_A_Survey_of_Parallel_Genetic_Algorithms)

The same is run using 7 processors

## Requirements:

1. Server: Scientific Linux release 7.5 (Nitrogen)
2. conda 4.5.10
3. Python 2.7.15 :: Anaconda, Inc.
4. numpy 1.15.0
5. Keras 2.2.2
6. Tensorflow 1.5
7. gcc: gcc (GCC) 4.8.5 20150623 (Red Hat 4.8.5-28)
8. ld: ldd (GNU libc) 2.17
9. Mpi4py 2.0.0

## Usage:
    mpiexec -n 7 python dnnt.py
    mpiexec -n 7 python innts.py
This saves the output in a log file.
