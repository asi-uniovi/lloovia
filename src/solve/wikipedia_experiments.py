import pandas as pd
import lloovia
import pickle
import pulp


###############################################################################
# Helper functions for performing the wikipedia experiments
###############################################################################
def load_workload(filename, year):
    '''Returns the pages requested per hour for a wikipedia project in a year.
    Args:
        filename: filename of the csv file
        year: string from 2010 to 2015

    Returns:
        A list of integers. Each value is the number of request in a hour
        of the year
    '''
    year = str(year)
    load = pd.read_csv(filename, sep=";",
                       names=["time", "requests"], header=None)
    load.time = pd.to_datetime(load.time)
    load = load.set_index(load.time)
    return load[year].requests.values   # [:8760]  Truncate leap years?


def load_infrastructure(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def perform_experiment_phase_I(infrastructure_file,
                               workload_file,
                               year,
                               max_bins,
                               output_file,
                               frac_gap=0.01,
                               max_seconds=10*60):
    problem = lloovia.Problem(load_infrastructure(infrastructure_file),
                              load_workload(workload_file, year))
    phaseI = lloovia.PhaseI(problem)
    solver = pulp.COIN(threads=2, fracGap=frac_gap,
                       maxSeconds=max_seconds, msg=1)
    phaseI.solve(max_bins=max_bins, solver=solver)
    phaseI.solution.save(output_file)


def perform_experiment_phase_II(infrastructure_file,
                                workload_file,
                                year,
                                phase_I_solution,
                                output_file,
                                frac_gap=None,
                                max_seconds=10*60):

    solution_I = lloovia.SolutionI.load(phase_I_solution)
    if solution_I.solving_stats.status == "aborted":
        # There is no solution of Phase I, so Phase II cannot
        # be performed
        print("Case '%s' has no solution for Phase I. Skipping" %
              (phase_I_solution))
        # Save empty dataframe as solution
        pd.DataFrame().to_pickle(output_file)
        return

    problem = lloovia.Problem(load_infrastructure(infrastructure_file),
                              load_workload(workload_file, year))
    solver = pulp.COIN(threads=2, fracGap=frac_gap,
                       maxSeconds=max_seconds, msg=1)
    phaseII = lloovia.PhaseII(problem, solution_I, solver=solver)

    phaseII.solve_period()
    phaseII.solution.save(output_file)


def perform_experiment_phase_II_od_only(infrastructure_file,
                                        workload_file,
                                        year,
                                        output_file,
                                        frac_gap=None,
                                        max_seconds=10*60):

    infrastructure = load_infrastructure(infrastructure_file)
    only_ondemand = [vm for vm in infrastructure if not vm.reserved]
    only_reserved = [vm for vm in infrastructure if vm.reserved]
    problem = lloovia.Problem(only_ondemand,
                              load_workload(workload_file, year))
    # Create a fake solution from phase I in which all reserved
    # instances are deactivated
    trivial_stats = lloovia.SolvingStatsI(
                                max_bins=None,
                                workload=lloovia.LlooviaHistogram({0: 0}),
                                frac_gap=None,
                                max_seconds=None,
                                creation_time=0.0,
                                solving_time=0.0,
                                status="optimal",
                                lower_bound=None,
                                optimal_cost=0.0
                            )
    # Solution: Zero reserved VMs of each type
    sol = dict((vm, 0) for vm in only_reserved)
    trivial_allocation = pd.DataFrame([sol])[only_reserved]
    trivial_solution = lloovia.Solution(problem, trivial_stats,
                                        allocation=trivial_allocation)
    solver = pulp.COIN(threads=2, fracGap=frac_gap,
                       maxSeconds=max_seconds, msg=1)
    phaseII = lloovia.PhaseII(problem, trivial_solution, solver=solver)

    phaseII.solve_period()
    phaseII.solution.save(output_file)


def get_infrastructure_for_providers(providers,
                                     azure_infrastructure_file,
                                     amazon_infrastructure_file):
    infrastructure = []
    if providers == "Azure" or providers == "Both":
        infrastructure.extend(
                load_infrastructure(azure_infrastructure_file))
    if providers == "Amazon" or providers == "Both":
        infrastructure.extend(
                load_infrastructure(amazon_infrastructure_file))
    return infrastructure


def perform_experiment_multiregion_phase_I(
                                   azure_infrastructure_file,
                                   amazon_infrastructure_file,
                                   workload_file,
                                   year,
                                   max_bins,
                                   providers_to_use,
                                   output_file,
                                   frac_gap=None,
                                   max_seconds=60*60):

    infrastructure = get_infrastructure_for_providers(
                            providers_to_use,
                            azure_infrastructure_file,
                            amazon_infrastructure_file
                            )
    problem = lloovia.Problem(infrastructure,
                              load_workload(workload_file, year))
    phaseI = lloovia.PhaseI(problem)
    solver = pulp.COIN(threads=2, fracGap=frac_gap,
                       maxSeconds=max_seconds, msg=1)
    phaseI.solve(max_bins=max_bins, solver=solver)
    phaseI.solution.save(output_file)


def perform_experiment_multiregion_phase_II(
                            azure_infrastructure_file,
                            amazon_infrastructure_file,
                            workload_file,
                            year,
                            providers_to_use,
                            phase_I_solution,
                            output_file,
                            frac_gap=None,
                            max_seconds=10*60):

    infrastructure = get_infrastructure_for_providers(
                            providers_to_use,
                            azure_infrastructure_file,
                            amazon_infrastructure_file
                            )
    solution_I = lloovia.SolutionI.load(phase_I_solution)
    if solution_I.solving_stats.status == "aborted":
        print("Case '%s' has no solution for Phase I. Skipping" %
              (phase_I_solution))
        # Save empty dataframe as solution
        pd.DataFrame().to_pickle(output_file)
        return
    problem = lloovia.Problem(infrastructure,
                              load_workload(workload_file, year))
    solver = pulp.COIN(threads=2, fracGap=frac_gap,
                       maxSeconds=max_seconds, msg=1)
    phaseII = lloovia.PhaseII(problem, solution_I, solver=solver)
    phaseII.solve_period()
    phaseII.solution.save(output_file)
