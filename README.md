lloovia
==============================

> LLOOVIA: Load Level based OptimizatiOn of VIrtual machine Allocation

This repository contains the supplementary material for the paper: "Optimal allocation of virtual machines in multi-cloud environments with reserved and on-demand pricing".

This material comprises the datasets used to generate the tables and figures in the paper, as well as all `lloovia` library and all necessary scripts to ensure the repeatability of the experiments. In addition, some scripts to download/update the data from cloud providers, Wikipedia traces, etc. are also provided, but not used by default to ensure the same results than the ones shown in the paper.

To repeat the experiments you will need Python 3, several scientific packages (`numpy`, `pandas`, etc.) the LP solver `COIN-cbc` and the python library `PuLP` used to model the LP problem. To automate the large number of scripts to run and the correct order to do so `Snakemake` is used. In short we will provide more detailed installation and running instructions.

But you can see the results without having to re-run the experiments, because the most relevant information is presented in form of Jupyter Notebooks, which can be read directly from GitHub's web interface (see `notebooks` folder).

