import pandas as pd
import lloovia
import pickle
import pulp
import re
import os


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

## TODO from this line on ...


###############################################################################
# Helper functions for processing the pickles generated by the experiments
###############################################################################
experiment_with_bins_pattern = re.compile(
                    r"(?P<folder>.*?)case_(?P<case_name>[a-z]+)"
                    r"_level_(?P<level>\d+)"
                    r"_bins_(?P<bins>\d+)\.pickle")
experiment_without_bins_pattern = re.compile(
                    r"(?P<folder>.*?)case_(?P<case_name>[a-z]+)"
                    r"_level_(?P<level>\d+)_nobins.pickle")


def extract_info_from_filename(filename):
    m1 = re.match(experiment_with_bins_pattern, filename)
    m2 = re.match(experiment_without_bins_pattern, filename)
    if m1:
        return (m1.group("folder"),
                m1.group("case_name"),
                int(m1.group("level")),
                int(m1.group("bins")))
    elif m2:
        return (m2.group("folder"),
                m2.group("case_name"),
                int(m2.group("level")),
                None)
    else:
        raise Exception("Filename %s doesnt match expected pattern" % filename)


def preprocess_filenames(file_list):
    """Checks that the filenames conform to the expected pattern, and
    that all of them belong to the same experimental scenario. Returns
    the name and level of the experimental scenario and the list of
    max_bins cases"""
    # Preprocess filenames to ensure they are consistent, and extract
    # info to later visit them in proper order
    folders = set()
    cases = set()
    levels = set()
    bins = list()
    for f in file_list:
        folder, case, level, max_bins = extract_info_from_filename(f)
        folders.add(folder)
        cases.add(case)
        levels.add(level)
        bins.append(max_bins)

    if len(cases) > 1 or len(levels) > 1 or len(folders) > 1:
        raise Exception("The list of files to process contains mixed "
                        "experimental scenarios")

    folder = folders.pop()
    case = cases.pop()
    level = levels.pop()
    return (folder, case, level, bins)


def get_info_from_experiment(folder, case, level, max_bins):
    """Reads the solution of a single experiment from a pickle file,
    extract the relevant data and returns it as a dictionary"""

    if max_bins is not None:
        filename = "{}case_{}_level_{}_bins_{}.pickle".format(
                        folder, case, level, max_bins)
    else:
        filename = "{}case_{}_level_{}_nobins.pickle".format(
                        folder, case, level)

    s = lloovia.Solution.load(filename)

    # Extract info, depending on whether it is a phaseI or phaseII solution
    if isinstance(s, lloovia.SolutionII):
        # Phase II solutions doesn't have lower_bounds nor a single status,
        # fracgap or max_second.
        # They have a list of status, fracGap and max_seconds per timeslot,
        # which are summarized in a single global_status
        status = s.solving_stats.global_status
        cost = s.get_cost()
        frac_gap = s.solving_stats.default_frac_gap
        max_seconds = s.solving_stats.default_max_seconds
        creation_time = s.solving_stats.global_creation_time
        solving_time = s.solving_stats.global_solving_time
    elif isinstance(s, lloovia.Solution):
        # Ensure that the max_bins stated in the filename matches with
        # the one contained in it (only for SolutionI case, since)
        if s.solving_stats.max_bins != max_bins:
            raise Exception("The file %s contains data for max_bins=%d" %
                            filename, s.max_bins)
        status = s.solving_stats.status
        # Store either the optimal solution, or the best known lower bound
        if status == "aborted":
            cost = s.solving_stats.lower_bound
        else:
            cost = s.solving_stats.optimal_cost
        frac_gap = s.solving_stats.frac_gap
        max_seconds = s.solving_stats.max_seconds
        creation_time = s.solving_stats.creation_time
        solving_time = s.solving_stats.solving_time
    else:
        # Unknown kind of solution. It is a marker of no-solution. Skip it
        return dict(max_bins=max_bins)

    if max_bins is None:
        max_bins = 24*365   # (No-bins case uses the full year length)
    return dict(
            max_bins=max_bins,
            cost=cost,
            seconds_to_create=creation_time,
            seconds_to_solve=solving_time,
            frac_gap=frac_gap,
            max_seconds=max_seconds,
            status=status
           )


def summarize_experiment(file_list, output_file=None):
    """
    The input is a list of pickle filenames. They are expected to have
    the pattern "case_{casename}_level_{number}_bins_{number}.pickle"
    or "case_{casename}_level_{number}_nobins.pickle"

    Returns a single dataframe in which each row is an experiment with
    a different value for max_bins. The columns are the global results
    of the experiment required to plot the graphs (max_bins, cost,
    seconds_to_solve, seconds_to_create, fracGap, maxSeconds, and status).
    """

    folder, case, level, bins = preprocess_filenames(file_list)

    data = []
    for max_bins in bins:
        data.append(get_info_from_experiment(folder, case, level, max_bins))

    df = pd.DataFrame(data)
    # Reorder columns
    df = df[["max_bins", "cost", "seconds_to_solve", "seconds_to_create",
             "frac_gap", "max_seconds", "status"]]
    df.max_bins.fillna(24*365, inplace=True)
    df.max_bins = df.max_bins.astype(int)
    df = df.sort_values(by="max_bins")
    if output_file:
        df.to_pickle(output_file)
    return


def collect_all_results(file_list, phases_dict, output_file=None):
    """Input: list of pickle filenames with abridged results for a set
    of experiments.
    Output: Single multi-level indexed dataframe with the results of
    all experiments, as requierd by the notebooks which plot the
    graphics comparisons"""

    # Regexp for the abridged results pickles
    re_experiment_summary = re.compile(
                        r"(?P<folder>.*?)case_(?P<case_name>[a-z]+)"
                        r"_level_(?P<level>\d+)_summary.pickle")

    def parse_filename(filename):
        "Extracts experiment metadata from the name of the file"
        m1 = re.match(re_experiment_summary, filename)
        if m1:
            return (m1.group("folder"),
                    m1.group("case_name"),
                    int(m1.group("level")))
        else:
            raise Exception("Filename %s "
                            "doesnt match expected pattern" % filename)

    def infer_phase_from_path(path, phases_dict):
        foldername = os.path.basename(os.path.normpath(path))
        for f, phase in phases_dict.items():
            if f == foldername:
                return phase
        return "Unknown"

    data = []
    for filename in file_list:
        folder, case, level = parse_filename(filename)
        phase = infer_phase_from_path(folder, phases_dict)
        # Each pickle is read into a dataframe, and new columns
        # are added to store the scenario name, load level and phase
        df = pd.read_pickle(filename)
        if df.empty:     # Skip empty dataframes
            continue
        df = (df
              .assign(Case=case.title())
              .assign(level=level).rename(columns={"level": "Base level"})
              .assign(Phase=phase))
        data.append(df)
    # All those dataframes are concatenated into a single one, and the
    # metadata is used as multi-level index
    df = (pd.concat(data)
          .set_index(["Case", "Base level", "Phase", "max_bins"])
          .sort_index())
    if output_file:
        df.to_pickle(output_file)
    return df
