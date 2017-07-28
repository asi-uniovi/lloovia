lloovia
==============================

> LLOOVIA: Load Level based OptimizatiOn of VIrtual machine Allocation

This repository contains the supplementary material for the paper: "[Optimal allocation of virtual machines in multi-cloud environments with reserved and on-demand pricing][1]" (Future Generation Computer Systems, Volume 71, June 2017, Pages 129-144, ISSN 0167-739X)

This material comprises the datasets used to generate the tables and figures in the paper, as well as all `lloovia` library and all necessary scripts to ensure the repeatability of the experiments. In addition, some scripts to download/update the data from cloud providers, Wikipedia traces, etc. are also provided, but not used by default to ensure the same results as the ones shown in the paper.

To repeat the experiments you will need Python 3, several scientific packages (`numpy`, `pandas`, etc.), the LP solver `COIN-cbc` and the python library `PuLP` used to model the LP problem. To automate the large number of scripts to run and the correct order to do so `Snakemake` is used. See [INSTALL.md][].

However, you can see the results without having to re-run the experiments, because the most relevant information is presented in form of Jupyter Notebooks, which can be read directly from GitHub's web interface (see `notebooks` folder).

Here you can see a simple example of how to use module `lloovia`:

    import lloovia

    # Create workloads
    ltwp = [5, 2, 9, 9, 11, 5, 11, 50] # long term workload prediction
    stwp = [1, 22, 5, 6, 10, 20, 50, 20] # short term workload prediction

    # Create limiting sets
    ls_us_east = lloovia.LimitingSet("US East (N. California)", max_vms=20, max_cores=0)
    ls_us_west_m4 = lloovia.LimitingSet("us_west_m4", max_vms=5, max_cores=0)

    # Create instances
    ic1 = lloovia.InstanceClass(name="m3.large", cloud=ls_us_east, max_vms=20,
                                reserved=True, price=2.3, performance=5)

    ic2 = lloovia.InstanceClass(name="m4.medium", cloud=ls_us_west_m4, max_vms=5,
                                reserved=False, price=4.5, performance=6)

    ic3 = lloovia.InstanceClass(name="m4.large", cloud=ls_us_west_m4, max_vms=5,
                                reserved=False, price=8.2, performance=7)
    instances = [ic1, ic2, ic3]

    # Create problem for phase I with LTWP
    problem_phase_i = lloovia.Problem(instances=instances, workload=ltwp)

    # Solve and print cost
    phase_i = lloovia.PhaseI(problem_phase_i)
    phase_i.solve()
    print("Cost for the LTWP:", phase_i.solution.get_cost())
    print("Allocation:", phase_i.solution.get_allocation())

    # Crete problem for phase II with STWP
    problem_phase_ii = lloovia.Problem(instances=instances, workload=stwp)

    # Solve and print cost
    phase_ii = lloovia.PhaseII(problem_phase_ii, phase_i.solution)
    phase_ii.solve_period()
    print("Cost for the STWP:", phase_ii.solution.get_cost())
    print("Allocation:", phase_ii.solution.get_allocation())

[INSTALL.md]: INSTALL.md
[1]: http://dx.doi.org/10.1016/j.future.2017.02.004
