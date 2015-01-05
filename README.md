
Restricted Bolzmann Machines in Julia
=====================================

[![Build Status](https://travis-ci.org/dfdx/Boltzmann.jl.svg)](https://travis-ci.org/dfdx/Boltzmann.jl)

This package provides implementation of restricted Boltzmann machines and deep belief networks in Julia. Its API is designed to resemble SciPy and at the same time conform with latest Julia statistical and machine learning packages. 

Installation 


This package provides implementation of 2 most commonly used types of Restricted Bolzmann Machines, namely: 

- **BernoulliRBM**: RBM with binary visible and hidden units
- **GRBM**: RBM with Gaussian visible and binary hidden units

Usage: 

    X = ...  # data matrix, observations as columns, variables as rows
    model = GRBM(n_visibles, n_hiddens)
    fit(model, X, n_iter=10, n_gibbs=3, lr=0.1)
    comps = components(model)    # matrix of learned components (on columns)

`components()` returns learned weights as a columnar matrix. This, however, is a transpose of real weight matrix. To avoid overhead one can pass `transpose=false`:

    comps = components(model, transpose=false)  # matrix of learned components (on rows)

Both - dense and sparse matrices are now supported. 

See `examples` directory for more use cases. 