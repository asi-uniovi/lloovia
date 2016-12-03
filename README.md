lloovia
==============================

LLOOVIA: Load Level based OptimizatiOn of VIrtual machine Allocation

This repository contains the supplementary material for the paper: "Optimal allocation of virtual machines in multi-cloud environments with reserved and on-demand pricing"

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Unused ?
    │   ├── interim        <- Pickle files with intermediate results
    │   ├── processed      <- Processed wikipedia traces, prices of providers, etc
    │   └── raw            <- Synthetic traces, wikipedia raw traces, VM types and prices as reported by cloud providers
    │
    ├── docs               <- Developer's documentation of Lloovia modules
    │
    ├── models             <- Unused?
    │
    ├── notebooks          <- Jupyter notebooks with presentation of experiments and results
    │
    ├── references         <- Unused ?
    │
    ├── reports            <- HTML or PDF versions of Notebooks
    │   └── figures        <- Generated graphics and figures to be used in paper
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code of Llovia and auxiliar python scripts
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── process_data   <- Scripts to turn raw data into the input required
    │   │   └── build_processed.py
    │   │
    │   ├── solve          <- Lloovia implementation
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

