[![conda](https://img.shields.io/badge/install%20with-conda-brightgreen.svg?style=flat-square)](https://www.continuum.io/downloads)  [ ![python3](https://img.shields.io/badge/python-3-blue.svg?style=flat-square)](https://www.python.org/download/releases/3.0/)  [ ![PuLP](https://img.shields.io/badge/PuLP-≥1.6.0-blue.svg?style=flat-square)](https://pythonhosted.org/PuLP/)

# Installation

These instructions are aimed at Linux platforms. However, since anaconda and coin-or are available also for Windows, it should be no problem to install the required software also in a Windows platform.

## Python packages and environments

To be able to reproduce the experiments in this repository, you'll need Python 3 and several scientific and graphics packages. The easisest way to get them all is to install [conda][1] or [miniconda][2] and then run the following command:

```
conda create -f lloovia_conda_env.yml
```

All required packages in the appropiate version will be downloaded and installed in a local folder. Nothing will be installed in your global system path, so you can safely run it even if you already have a working python installation. As a result, a "conda enviroment" called `lloovia` will be created.  You can check that it is visible with `conda list`. To use the environment you have to "activate" it with the command:

```
source activate lloovia
```

While the environment is active, you are using a local Python 3 installation with all required packages. You can exit the environment with the command:

```
source deactivate
```

and you'll return to your default python installation.

## Non-python software

Llovia uses [PuLP][] as Linear Programming modelling language and interface to several solvers. It is a python library which is installed as part of the conda environment above. PuLP allows the creation of files describing the problem (using `.lp`  or `.mps` formats) using Python, and provides a consistent interface with different solvers, but it is not itself a solver.

Although PuLP includes a binary executable with `cbc` solver, which is used by default if no other solver is specified, in order to gain more flexibility in the number of options which can be passed to the solver a working installation of [COIN-OR cbc][cbc] solver is needed.

In debian based distributions of Linux it is easy to get:

```
sudo apt-get install coinor-cbc
```

The version of `cbc` used by the authors is `2.8.7`


# Running the experiments

All experiments are automated using [snakemake][3] (which is also installed as part of the `lloovia` environment). This is a tool similar to standard `make` but more appropiate to manage complex workflows which involve multiple files with similar names which use the same rule to be generated, and integrates well with python code.

Several `Snakemake` files are provided, and each one contains rules to perform specific experiments. The most important tasks are:

## Obtaining the required datasets for the experiments

The experiments require data from different sources: 

* Synthetic traces for the initial example (section 4) and synthetic experiments (section 5.2)
* Real traces from Wikipedia (for section 5.3)
* Providers data (prices, characteristics) from Amazon (for section 5.2, 5.3.2 and 5.3.3) and Azure (section 5.3.4)
* Benchmarks results from Wikibench (section 5.2, 5.3.2 and 5.3.3) and Oldisim (section 5.3.4)

All this data can be rebuild from it source, but it will take some time to download all Wikipedia traces. In addition, the prices, limits and VMs available per region could be different than the ones used in the paper, making the experiments not reproducible. This is why we provided the processed version of the data in folder `data/paper`, so reproducibility is ensured and no time is required to rebuild this dataset.

To use the original dataset in the paper:

```
snakemake
```

To rebuild the dataset, edit `Snakefile` and set `USE_PAPER_DATASET=False`, then run `snakemake` as above. This will not overwrite the contents of `data/paper`, so at any time you can put `USE_PAPER_DATASET=True` and run `snakemake` again to recover the original dataset.

## Running experiments

### Running simple example

```
snakemake -s Snakefile.example
```

This will create several `pickle` files with the results of the example given in Section 4 of the paper.

However you may prefer to read the notebook `04-Example`, which does not use `snakemake`, but explains the steps required to arrive to the solution, and provide plots of the workload, the histograms, and the solution. The notebook can be read directly at [GitHub][] (because GitHub has support to interpret and render this kind of files), or you can launch your own Jupyter Notebook server to be able to edit the notebook and interactively explore the solution and methods.

### Synthetic experiments

> **Important note:** Running the experiments is required only to ensure the reproducibility of the results, but it is not required to explore or plot the solutions, since the `pickle` files with the final results are provided in the git repository. To explore the solutions only the provided pickles are needed. The notebook `05-Experiments.ipynb` can be used as an example of how to load and visualize the solutions. 
>
> Skip to [Visualizing results](#visualizing-results) for details.

```
snakemake -s Snakefile.experiments
```

Be warned, there are more than 1000 experiments to be performed, and the total time required to complete them all is around 48h. The result of this command will be a great number of `pickle` files, each one with the solution of a single experiment (for a given workload case, base level  and `max_bins` value).

### Wikipedia experiments

```
snakemake -s Snakefile.wikipedia
```

Although the number of experiments in this case is not as large as in previous case, the time required to complete this command can be of several hours, because of the "nobins" case, which requires an extremely large amount of time. In fact, it is aborted after 1h if no solution is found (which was the case in our machine).

### Join all results

In order to make easier to analyze the results of all experiments, compare them, etc. all the `pickle` files generated by previous commands are joined in a single one. In this operation the particular allocations (number and type of VM at each timeslot) for each experiment is discarded, and only some "metadata" of each experiment is considered, such as the global cost of the solution for the whole year, the time required to create the problem and to solve it, and some parameters of the problem.

```
snakemake -s Snakefile.analyze_experiments
```

As a result, five files are created, which are used by notebook `05-Experiments.ipynb`

* `data/processed/all_experiments_results.pickle` contains a single dataframe with the results for the synthetic experiments. All data required to recreate the figures and tables of section 5.2 in the paper is in this file. 
* `data/processed/all_wikipedia_results.pickle` contains a single dataframe with the results for the wikipedia experiments, except for the multi-cloud experiment. All data required to recreate figures and tables of sections 5.3.2 and 5.3.3 of the paper is in this file.
* `data/processed/case_wikipedia_2014_multi_*_bins_40.pickle` These are three files (with `* ` = `Amazon`, `Azure` or `Both`, containing the full solution (including allocation for each timeslot) for the multicloud experiment.

# Visualizing results

The result of running all the experiments will be a set of `pickle` files as explained above. Those files contain pandas dataframes which can be interactively explored from a notebook. Notebooks are interactive python interpreters, and they can also be used to explore LLOOVIA's API, create and solve different allocation problems, etc.

Two notebooks are provided as example:

* [04-Example](notebooks/04-Example.ipynb) shows how to (interactively) create the problem described in Section 4 of the paper, and how to use Lloovia to solve it, as well as how to extract relevant information from the solution and create some plots.
* [05-Experiments](notebooks/05-Experiments.ipynb) shows how to load the `pickle` files containing the results of all experiments and, from this data, creates all plots and tables shown in section 5 of the paper.

Following the links above, a non-interactive version of these notebooks can be read in GitHub. This has the advantage of not requiring any installation. However, to be able to modify and run some of the cells to explore the solutions and API, a Jupyter Notebook server has to be launched.

From Unix, this can be done with:

```
cd notebooks
jupyter notebook --no-browser
```

And open a Web browser at the URL shown by the command above. To modify server's ip and port you can use options `--ip` and `--port`. If you are running it in a Linux with GUI, you can omit `--no-browser` and a browser will be automatically open connected to the jupyter server.

[1]: https://www.continuum.io/downloads
[2]: http://conda.pydata.org/miniconda.html
[3]: https://bitbucket.org/snakemake/snakemake/wiki/Home
[4]: https://github.com/asi-uniovi/lloovia/tree/master/notebooks
[PuLP]: https://pythonhosted.org/PuLP/
[cbc]: https://projects.coin-or.org/Cbc

