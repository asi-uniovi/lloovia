# coding: utf-8
# import matplotlib.pyplot as plt
# from pulp import *
import pulp
import pandas as pd
import numpy as np
from pulp import (LpContinuous, LpInteger, LpVariable, lpSum,
                  LpProblem, LpMinimize, LpMaximize, PulpSolverError,
                  COIN, COIN_CMD, log, subprocess)
from collections import (namedtuple, OrderedDict, Iterable)
from itertools import product as cartesian_product
from inspect import ismethod
import os
import pickle
import time

# Set following ones in notebook, if required
# %matplotlib inline
# plt.style.use("bmh")   # Set this one in notebook
# from IPython.display import display


# LimitingSet

class LimitingSet:
    def __init__(self, name, max_vms=20, max_cores=0):
        self.name = name
        self.max_vms = max_vms if max_vms is not None else 0
        self.max_cores = max_cores if max_cores is not None else 0

    def __repr__(self):
        return self.name


# InstanceClass

class InstanceClass:
    """This class represent a virtual machine class in some cloud provider
    and its attributes (performance, price, etc). It includes the attribute
    max_vms because some providers (eg. Amazon) pose a limit in the
    maximum number of active instances per class and region"""
    def __init__(self, name, cloud, price, performance, max_vms=0,
                 _type="", reserved=False, provides=None, comment=""):
        """'name' is any string, usually very short, e.g: 'S', 'L'
           which is meaningful for the analyst.

        'cloud' is a LimitingSet

            The 'name' and 'cloud' string repr are concatenated to create the
            string representation of this InstanceClass, which is used when
            printing objects of this type.

        'price' is the price/hour of this kind of instance (if the instance
             has any kind of discount, this price has to include it,
             proportional to the hour)

        'performance' is the performance given for this instance class
             in any unit which is meaningful for the analyst, who should
             use the same units when specifying the expected load

        'max_vms' is the limit on the number of active virtual machines of this
             type in the limit set for this instance class (eg: Amazon imposes
             such a limit in a region). If the provider does not impose this
             kind of limit, it should be None.

        '_type' is the identifier the cloud provider uses for this kind
             of instance, eg, Amazon's 't2.small'

        'reserved' is a boolean indicating if this instance is reserved.
             This kind of instances are cheaper, but they are paid even
             if they are not instantiated, so the formula computing the
             cost has to deal with them differently.

        'provides' it can contain special features of this kind of instance,
             such as the OS it can run, the amount of RAM, etc. which can be
             used to restrict the mapping of only certain types of machines.
             Currently, only the number of cores is used

        'comment' is a free form string.
        """
        self.name = name
        self.cloud = cloud
        self.price = price
        self.max_vms = max_vms if max_vms is not None else 0
        self._type = _type
        self.performance = performance
        self.reserved = reserved
        self.provides = provides
        self.comment = comment
        if self.provides and "cpus" in self.provides:
            self.cores = provides["cpus"]
        else:
            self.cores = 1

    def __repr__(self):
        """PuLP uses repr(obj) to create names for the linear problem
        variables, so we provide a method which builds a sensible
        representation. It consists in the concatenation of the name
        and cloud strings, and an extra '_R' if it is reserved"""
        return "{}_{}{}".format(self.name, self.cloud,
                                "_R" if self.reserved else "")

    def _repr_html_(self):
        """HTML Table representation, used in IPython Notebook
        via IPython.display"""
        html = "<table><tr><th colspan='2'>" +\
               "<center><b>Instance {0}</b></center></th></tr>".format(self)
        for attr in sorted(self.__dict__.keys()):
            html += "<tr><td><b>{0}</b></td><td>{1}</td></tr>"\
                    .format(attr, self.__dict__[attr])
        return html + "</table>"

    def order_key(self):
        """Returns a "key" used to order a list of instance classes"""
        # The key is built to group first all instance classes in the same
        # cloud (by alphabetical order of cloud), then all reserved classes
        # before all non-reserved, and finally by alphabetical order of the
        # type name but dealing specially with names 'S', 'L' and 'XL', to be
        # sorted in crescent order of size
        rsv = "0" if self.instances_res else ""
        order_sizes = {"S": "0A", "L": "0B", "XL": "0C"}
        if self.name in order_sizes.keys():
            name = order_sizes[self.name]
        else:
            name = self.name
        cloud = self.cloud
        return "{}{}{}".format(cloud, rsv, name)

# Problem


class Problem:
    """This class represents a cloud allocation problem that must be solved.
    It has the list of instances to use and the load, expressed as the load
    for each slot. It provides convenience methods to save and load the problem
    """
    def __init__(self, instances, workload):
        """Parameters:
        - instances: list of InstanceClass
        - load: list of numbers with the load of each slot
        """
        self.instances = instances
        self.workload = workload

    def __repr__(self):
        return "Problem with {} instance classes and {} timeslots".format(
                len(self.instances), len(self.workload)
                )

    def get_only_ondemand(self):
        return [vm for vm in self.instances if not vm.reserved]

    def save(self, filename):
        """Parameters:
        - filename: string with the name of the file
        """
        with open(filename, "wb") as f:
            pickle.dump(file=f, obj=self)

    def load(filename):
        """Parameters:
        - filename: string with the name of the file

        Returns:
        - The loaded problem
        """
        with open(filename, "rb") as f:
            return pickle.load(file=f)

# Lloovia


# Llovia uses plain dictionaries to store histograms. We subclasse it
# to provide a custom _repr_ more compact, better suitable for
# interactive inspection
class LlooviaHistogram(dict):
    def __repr__(self):
        return "LlooviaHistogram(%d elements)" % len(self)


# To store the individual status of each timeslot we use a list
# but subclass it to provide a custom _repr_ more compact, better
# suitable for interactive inspection
class StatusList(list):
    def __repr__(self):
        return "StatusList(%s)" % set(self)


class Lloovia:
    """This class contains methods to create a linear programming problem
    (using PuLP), to add restrictions and extra variables to it,
    to solve it (using PuLP supported solvers), and to retrieve
    the solution in a format amenable to further analysis and display.

    This class implements the problem in which the number of
    machines of a set of InstanceClasses has to be obtained for a set of values
    for the expected load, whose histogram is known.

    The only restrictions in this problem are:

    1. For each workload level, the total performance of the deployed machines
       has to be greater than (or equal to) that workload level.
    2. If the provider imposes a limit on the maximum number of virtual
       machines active per limiting set (or per instance class and limiting
       set), this constrain is taken into account.

    The problem instantiates variables are:

    - For reserved instances: `Y_(_ic)`, where `Y` is a fixed prefix, and`ic`
    is the string representation of each reserved instance class considered.
    The value of the variable is the number of machines executed of instance
    class `ic`

    - For on-demand instances: `X_(_ic,_l)`, where `X` is a fixed prefix,
    `ic` is the string representation of each on-demand instance class
    considered and `l` is each of the load levels considered. The value of the
    variable is the number of machines executed of instance class `ic` when the
    workload is `l`

    All possible combinations of the tuple (it) for reserved instances
    and (it,l) for on-demand instances are precomputed in method
    `create_problem` and stored in `self.mapping_res` y `self.mapping_dem`
    respectively.
    """
    def __init__(self, instances, workload, max_bins=None,
                 title="Optimize cost", relaxed=False):
        """Initalizes the optimization problem. The parameters are:
        'instances': list of InstanceClass to consider in the deployment.
        'workload': list of number indicating the workload that has to be
            supported in each time slot
        'max_bins': maximum number of bins to consder when computing
              the histogram of the workload. If None, each load level would
              be a bin.
        'title': optional title for the linear programming problem.
        'relaxed': boolean; if True, the problem uses continuous variables
              instead of integer ones
        """
        self.instances = instances
        self.workload = workload
        self.max_bins = max_bins
        self.title = title
        self.relaxed = relaxed

        # Compute the histogram
        self.load_hist = get_load_hist_from_load(workload, max_bins=max_bins)

        # Separate the instances in two types: reserved and on-demand
        self.instances_res = []
        self.instances_dem = []
        for i in self.instances:
            if i.reserved:
                self.instances_res.append(i)
            else:
                self.instances_dem.append(i)

        # Compute the set of LimitingSets (clouds), extracted
        # from the instances
        self.clouds = set()
        for i in instances:
            self.clouds.add(i.cloud)

    def create_variables(self):
        """Creates the set of variables Y* and X* of the basic problem.
        Override it if you need to create extra variables (first use
        super().create_variables() to call the base clase method)"""
        if self.relaxed:
            kind = LpContinuous
        else:
            kind = LpInteger
        self.vms_res = LpVariable.dicts('Y', self.mapping_res, 0, None, kind)
        self.vms_dem = LpVariable.dicts('X', self.mapping_dem, 0, None, kind)

    def cost_function(self):
        """Adds to the problem the function to optimize, which is the
        cost of the deployment. In the basic problem it is the sum of
        all Y_ic multiplied by the length of the period and by the
        price/hour of each reserved instance class plus all X_ic_l
        multiplied by the price/hour of each on-demand instance class.

        Override to change the way the cost is computed"""

        # The lenght of the period is the number of timeslots in the workload
        if self.workload is not None:
            period_length = len(self.workload)

        # IF the load was not provided, we can use the information from
        # the histogram instead
        else:
            period_length = sum(self.load_hist.values())
        self.prob += lpSum(
                          [self.vms_res[_ic] * _ic.price * period_length
                           for _ic in self.mapping_res] +
                          [self.vms_dem[_ic, _l] * _ic.price *
                           self.load_hist[_l]
                           for (_ic, _l) in self.mapping_dem
                           ]), "Objective: minimize cost"

    def create_problem(self):
        """This method creates the PuLP problem, and calls other
        methods to add variables and restrictions to it.

        It initializes the following attributes, which can be
        useful in derived classes:

          self.mapping_res: list of all ic for reserved instances, useful to
            create iteration loops.
          self.mapping_dem: list of all combinations (ic,l) for on-demand
            instances, useful to create iteration loops.
          self.prob: instance of the PuLP problem
        """

        # Create the linear programming problem
        self.prob = LpProblem(self.title, LpMinimize)

        # Create the list of workload levels where the scheduling
        # has to be found
        self.loads = self.load_hist.keys()

        # The variables in the problem for on-demand instances are all the
        # possible tuples (instance, load). In the solution, each tuple has a
        # numeric value that indicates the number of virtual machines
        # instanciated of that instance class for that load. Reserved instances
        # are the same, but without the last index (load)
        self.mapping_res = self.instances_res
        self.mapping_dem = list(cartesian_product(self.instances_dem,
                                                  self.loads))

        # Once we have the variables represented as tuples, we use
        # the tuples to create the linear programming variables for pulp
        self.create_variables()

        # Create the goal function
        self.cost_function()

        # Add all restrictions indicated with functions *_restriction
        # in this class
        self.add_all_restrictions()

        return self.prob

    def add_all_restrictions(self):
        """This functions uses introspection to discover all implemented
        methods whose name ends with `_restriction`, and runs them all"""
        for name in dir(self):
            attribute = getattr(self, name)
            if ismethod(attribute) and name.endswith("_restriction"):
                attribute()

    def performance_restriction(self):
        """Adds to the PuLP problem the restriction of performance, which
        consists on forcing for each workload level, the performance of the
        deployment to be greater than or equal to that workload level.
        """
        for ld in self.load_hist.keys():
            self.prob += lpSum(
                [self.vms_res[_ic] * _ic.performance
                    for _ic in self.mapping_res] +
                [self.vms_dem[_ic, ld] * _ic.performance
                    for (_ic, _ld) in self.mapping_dem if _ld == ld]) >= ld,\
                    "Minimum performance when workload is %d" % ld

    def limit_instances_per_class_restriction(self):
        """If the instance has a max_vms attribute, this is a limit for that
        variable."""
        for ic in self.instances:
            if ic.max_vms > 0:
                # There is a limit per instance class

                if ic.reserved:
                    self.prob += lpSum(self.vms_res[ic]) <= ic.max_vms, \
                            "Max instances for reserved "\
                            "instance class %r" % (ic)
                else:
                    for l in self.loads:
                        self.prob += lpSum(self.vms_dem[ic, l]) \
                                     <= ic.max_vms, \
                                     "Max instances for on-demand instance "\
                                     "class %r when workload is %d" % (ic, l)

    def limit_instances_per_limiting_set_restriction(self):
        """If the limiting set provides a max_vms > 0, then the sum of all
        instances in that limiting set should be limited to that maximum"""
        for l in self.loads:
            for cloud in self.clouds:
                if cloud.max_vms > 0:
                    self.prob += lpSum(
                        [self.vms_res[ic] for ic in self.instances_res
                            if ic.cloud == cloud] +
                        [self.vms_dem[ic, l] for ic in self.instances_dem
                            if ic.cloud == cloud]) <= cloud.max_vms,\
                                  "Max instances for limiting set %r "\
                                  "when workload is %d" % (cloud, l)

    def limit_cores_per_limiting_set_restriction(self):
        """If the limiting set provides a max_cores > 0, then the sum of all
        instance cores in that region should be limited to that maximum"""
        for l in self.loads:
            for cloud in self.clouds:
                if cloud.max_cores > 0:
                    self.prob += lpSum(
                        [self.vms_res[ic]*ic.cores
                            for ic in self.instances_res
                            if ic.cloud == cloud] +
                        [self.vms_dem[ic, l]*ic.cores
                            for ic in self.instances_dem
                            if ic.cloud == cloud]) <= cloud.max_cores,\
                                "Max cores for limiting set %r "\
                                "when workload is %d" % (cloud, l)

    def solve(self, *args, **kwargs):
        """Calls PuLP solver. Accepts the same arguments as a pulp solver"""
        return self.prob.solve(*args, **kwargs)

    def cost(self):
        """Gets the cost of the problem obtained after solving it"""
        if self.prob.status != pulp.LpStatusOptimal:  # Not solved
            raise Exception("Cannot get the cost of an unsolved problem")
        return pulp.value(self.prob.objective)

    def get_soldf(self, only_used=False):
        """Returns the solution as a DataFrame. Rows are workload levels and
        columns are instance clases. If only_used is True, instance classes
        never used are not included in the DataFrame.
       """

        if self.prob.status != pulp.LpStatusOptimal:  # Not solved
            raise Exception("Cannot get the solution for an unsolved problem")

        # Two DataFrames are created, one for reserved instances and another
        # one with on-demand instances, and then they are joined because,
        # this way, reserved instances are first and plots are easier to
        # understand and compare

        sol = {}
        for i in self.vms_res:
            instance = i
            sol[instance] = {}
            for load in self.load_hist:
                sol[instance][load] = self.vms_res[i].varValue

        df_res = pd.DataFrame(sol)

        sol = {}
        for i in self.vms_dem:
            instance = i[0]
            load = i[1]
            try:
                sol[instance][load] = self.vms_dem[i].varValue
            except:
                sol[instance] = {}
                sol[instance][load] = self.vms_dem[i].varValue

        df_dem = pd.DataFrame(sol)

        soldf = pd.concat([df_res, df_dem], axis=1)

        if only_used:
            return soldf[soldf.columns[(soldf != 0).any()]]
        else:
            return soldf


# Phase I
SolvingStatsI = namedtuple("SolvingStatsI",
                           ["max_bins", "workload",
                            "frac_gap", "max_seconds",
                            "creation_time", "solving_time",
                            "status", "lower_bound", "optimal_cost"
                            ]
                           )


class PhaseI:
    def __init__(self, problem, title="Optimize cost Phase I"):
        self.problem = problem
        self.title = title
        self.solution = None
        self._lloovia = None

    def solve(self, max_bins=None, solver=None, relaxed=False):

        allocation = None
        lower_bound = None
        optimal_cost = None
        status = "unknown"

        # Instantiate problem
        self._lloovia = Lloovia(self.problem.instances,
                                self.problem.workload,
                                max_bins=max_bins,
                                title=self.title,
                                relaxed=relaxed)

        # Write the LP problem and measure the time required
        start = time.perf_counter()
        p = self._lloovia.create_problem()
        creation_time = time.perf_counter() - start

        # Prepare solver
        if solver is None:
            solver = COIN(msg=1)
        frac_gap = solver.fracGap
        max_seconds = solver.maxSeconds

        # Solve the problem and measure the time required
        start = time.perf_counter()
        try:
            p.solve(solver, use_mps=False)
        except PulpSolverError as exception:
            end = time.perf_counter()
            solving_time = end - start
            status = "unknown_error"
            print("Exception PulpSolverError. Time to failure: {} seconds\n"
                  .format(solving_time), exception)

        else:
            # No exceptions
            end = time.perf_counter()
            solving_time = end - start
            if p.status == pulp.LpStatusInfeasible:
                status = 'infeasible'
            elif p.status == pulp.LpStatusNotSolved:
                status = 'aborted'
                lower_bound = p.bestBound
            elif p.status == pulp.LpStatusOptimal:
                status = "optimal"
                allocation = self._lloovia.get_soldf()
                lower_bound = p.bestBound
                optimal_cost = self._lloovia.cost()
            else:
                status = str(p.status)

        solving_stats = SolvingStatsI(max_bins=max_bins,
                                      workload=self._lloovia.load_hist,
                                      frac_gap=frac_gap,
                                      max_seconds=max_seconds,
                                      creation_time=creation_time,
                                      solving_time=solving_time,
                                      status=status,
                                      lower_bound=lower_bound,
                                      optimal_cost=optimal_cost
                                      )

        self.solution = Solution(self.problem, solving_stats,
                                 allocation)


class Solution:
    def __init__(self, problem, solving_stats, allocation):
        """Stores all relevant data related to the solution of an allocation
        problem, including the problem data."""
        self.problem = problem
        self.solving_stats = solving_stats
        self.allocation = allocation
        self._cost = None  # Computed from the allocation
        self._allocation_with_data = None
        self._full_dataframe = None

    def __repr__(self):
        return "{} solution with cost {}".format(
                self.solving_stats.status,
                self.solving_stats.optimal_cost
                )

    def get_allocation(self, only_used=True):
        if only_used:
            return self.allocation[
                    self.allocation.columns[(self.allocation != 0).any()]
                    ]
        else:
            return self.allocation

    def get_cost(self, kind="total"):
        if self._cost is None:
            self._cost = self.compute_cost()
        if kind == "total":
            return self._cost.sum()
        elif kind == "reserved":
            return self._cost[:, True].sum()
        elif kind == "ondemand":
            return self._cost[:, False].sum()
        elif kind in self._cost:
            return self._cost[kind].sum()
        elif kind in self._cost.index.levels[2]:
            return self._cost[:, :, kind].sum()
        elif kind in self._cost.index.levels[3]:
            return self._cost[:, :, :, kind].sum()
        else:
            return self._cost

    def get_allocation_with_data(self):
        """Returns a multi-index dataframe which categorizes the instance classes
        per provider, region, kind of princing plan. In addition the dataframe
        contains the price and performance of each instance class, besides the
        number of each one required for each load level"""

        if self._allocation_with_data is not None:
            return self._allocation_with_data

        a = self.get_allocation(only_used=True)
        a.columns.name = "VM"
        solution = (a.T.reset_index()
                    .assign(Name=lambda x: x.VM.map(lambda x: x.name))
                    .assign(LS=lambda x: x.VM.map(lambda x: x.cloud.name))
                    .assign(provider=lambda x: x.LS.map(lambda x: "azure"
                                                        if "us-east-2" in x
                                                        else "amazon"))
                    .assign(rsv=lambda x: x.VM.map(lambda x: x.reserved))
                    .assign(perf=lambda x: x.VM.map(lambda x: x.performance))
                    .assign(price=lambda x: x.VM.map(lambda x: x.price))
                    .set_index(["provider", "rsv", "Name", "LS"])
                    .sort_index()
                    )
        self._allocation_with_data = solution
        return self._allocation_with_data

    def get_cost_and_perf_dataframe(self):
        """Returns a dataframe which contains the cost and performance
        of each load-level, taking into account the number of times each
        load-levels appears in the histogram"""
        if self._full_dataframe is not None:
            return self._full_dataframe
        solution = self.get_allocation_with_data()
        vm_numbers = solution.iloc[:, 1:-2].sort_index()
        costs = (vm_numbers.T * solution.price)
        perfs = (vm_numbers.T * solution.perf)

        # If the provided workload is a Histogram, multiply
        # by the number of times each loadlevel appears
        if isinstance(self.solving_stats.workload, Iterable):
            H = pd.Series(self.solving_stats.workload)
            costs = costs.multiply(H, axis=0)
            perfs = perfs.multiply(H, axis=0)
        self._full_dataframe = pd.concat([costs, perfs], axis=1,
                                         keys=("Cost", "Performance"))
        return self._full_dataframe

    def compute_cost(self):
        all_data = self.get_cost_and_perf_dataframe()
        return all_data.loc[:, "Cost"].sum()

    def compute_performance(self):
        all_data = self.get_cost_and_perf_dataframe()
        return all_data.loc[:, "Performance"].sum()

    def compute_reserved_performance(self):
        """Computes the performance given at each timeslot
        for all reserved instances."""
        # Get detailed data about the allocation and VM characteristics
        aux = self.get_allocation_with_data()
        # Extract data about reserved instances
        reserved = aux.loc[pd.IndexSlice[:, True],]
        # Take the allocation of the first load-level (the reserved allocation
        # is the same for any load-level) and compute the performance it gives
        perf = (reserved.iloc[:, 1] * reserved.perf).sum()
        return perf

    def save(self, filename):
        """Parameters:
        - filename: string with the name of the file
        """
        with open(filename, "wb") as f:
            pickle.dump(file=f, obj=self)

    def load(filename):
        """Parameters:
        - filename: string with the name of the file

        Returns:
        - The loaded solution
        """
        with open(filename, "rb") as f:
            return pickle.load(file=f)


class SolutionI(Solution):
    """Subclass of general solution for the particular case of
    Phase I solution"""
    pass    # The general case is valid, no overload required


# Phase II
SolvingStatsTimeslot = namedtuple("SolvingStatsTimeslot",
                                  ["workload", "ondemand_workload",
                                   "frac_gap", "max_seconds",
                                   "creation_time", "solving_time",
                                   "status", "lower_bound", "optimal_cost"
                                   ]
                                  )

SolvingStatsII = namedtuple("SolvingStatsII",
                            ["workload",
                             "default_frac_gap", "default_max_seconds",
                             "global_creation_time", "global_solving_time",
                             "global_status", "global_cost",
                             "individual_status"
                             ]
                            )


class PhaseII:
    def __init__(self, problem, phase_I_solution,
                 title="Optimize cost Phase II", solver=None):
        self.problem = problem
        self.title = title
        if solver is None:
            self.solver = COIN(msg=1)
        else:
            self.solver = solver
        self.phase_I_solution = phase_I_solution
        self.ondemand_instances = problem.get_only_ondemand()
        self.reserved_performance = (phase_I_solution.
                                     compute_reserved_performance())

        # Hash with the computed solutions for each workload level
        # initially empty
        self._solutions = OrderedDict()

        # Internal handle to the inner lloovia solver
        self._lloovia = None

        # Prepare a "trivial" solution for the cases in which
        # the workload does not require on-demand instances
        trivial_stats = SolvingStatsTimeslot(
                                workload=0,
                                ondemand_workload=0,
                                frac_gap=None,
                                max_seconds=None,
                                creation_time=0.0,
                                solving_time=0.0,
                                status="trivial",
                                lower_bound=None,
                                optimal_cost=0.0
                                )
        # Solution: Zero ondemand VMs of each type
        sol = dict((vm, 0) for vm in self.ondemand_instances)
        trivial_allocation = pd.DataFrame([sol])[self.ondemand_instances]
        self.__trivial_sol = Solution(self.problem, trivial_stats,
                                      allocation=trivial_allocation)

        # Prepare a solution which maximizes the performance for the
        # cases in which the workload is greater than the maximum performance
        # that the system can give. In this case, the original problem
        # would be infeasible, and instead of giving a "none" solution
        # we give one which maximizes the performance.
        (self.max_performance_allocation,
         self.max_performance,
         self.max_performance_cost) = self.compute_max_performance()

    def compute_max_performance(self):
        # Obtain the maximum load that can be handled with the on-demand VMs
        # and the dataframe with the number of VMs for that solution
        lloovia_max_perf = LlooviaMaxPerformance(self.ondemand_instances)
        lloovia_max_perf.create_problem()
        lloovia_max_perf.solve(self.solver)
        max_perf_allocation = lloovia_max_perf.get_soldf()
        max_perf = lloovia_max_perf.max_perf()
        max_perf_cost = lloovia_max_perf.cost()
        return (max_perf_allocation[self.ondemand_instances],
                max_perf, max_perf_cost)

    def solve_timeslot(self, workload, solver=None, relaxed=False):

        if workload in self._solutions:
            # This workload was already solved. Nothing to be done
            return

        ondemand_workload = workload - self.reserved_performance

        # Zero or negative workload implies all demand is served with
        # reserved instances, so no on-demand ones are required
        if ondemand_workload <= 0:
            self._solutions[workload] = self.__trivial_sol
            return

        # Workload greater than the maximum achievable is an infeasible
        # case. We use the precomputed solution which maximizes performance
        if ondemand_workload >= self.max_performance:
            stats = SolvingStatsTimeslot(
                                workload=workload,
                                ondemand_workload=ondemand_workload,
                                frac_gap=self.solver.fracGap,
                                max_seconds=self.solver.maxSeconds,
                                creation_time=0.0,
                                solving_time=0.0,
                                status="overfull",
                                lower_bound=None,
                                optimal_cost=self.max_performance_cost
                                )
            self._solutions[workload] = (Solution(self.problem, stats,
                                         self.max_performance_allocation))
            return

        # Otherwise, we have to solve this timeslot
        allocation = None
        lower_bound = None
        optimal_cost = None
        status = "unknown"

        if solver is None:   # default to class solver
            solver = self.solver

        # Instantiate problem
        self._lloovia = Lloovia(self.ondemand_instances,
                                [ondemand_workload],
                                title=self.title,
                                relaxed=relaxed)

        # Write the LP problem and measure the time required
        start = time.perf_counter()
        p = self._lloovia.create_problem()
        creation_time = time.perf_counter() - start
        frac_gap = solver.fracGap
        max_seconds = solver.maxSeconds

        # Solve the problem and measure the time required
        start = time.perf_counter()
        try:
            p.solve(solver, use_mps=False)
        except PulpSolverError as exception:
            end = time.perf_counter()
            solving_time = end - start
            status = "unknown_error"
            print("Exception PulpSolverError. Time to failure: {} seconds\n"
                  .format(solving_time), exception)

        else:
            # No exceptions
            end = time.perf_counter()
            solving_time = end - start
            if p.status == pulp.LpStatusInfeasible:
                status = 'infeasible'
            elif p.status == pulp.LpStatusNotSolved:
                status = 'aborted'
                lower_bound = p.bestBound
            elif p.status == pulp.LpStatusOptimal:
                status = "optimal"
                allocation = self._lloovia.get_soldf()[self.ondemand_instances]
                lower_bound = p.bestBound
                optimal_cost = self._lloovia.cost()
            else:
                status = str(p.status)

        solving_stats = SolvingStatsTimeslot(
                                workload=workload,
                                ondemand_workload=ondemand_workload,
                                frac_gap=frac_gap,
                                max_seconds=max_seconds,
                                creation_time=creation_time,
                                solving_time=solving_time,
                                status=status,
                                lower_bound=lower_bound,
                                optimal_cost=optimal_cost
                                )

        self._solutions[workload] = Solution(self.problem, solving_stats,
                                             allocation)

    def solve_period(self):
        """Iterates over each timeslot, solving it and storing the solution
        in self._solutions. Finally all solutions are aggregated into a
        single global solution."""
        for load in self.problem.workload:
            self.solve_timeslot(load)
        self.aggregate_solutions()

    def aggregate_solutions(self):
        """Build a SolutionII object from the data in the _solutions
        attribute. It has to convert the dictionary of Solutions for
        each load-level into a single solution which will contain
        a list of stats (per timeslot) plus a DataFame with allocations
        per timeslot"""

        # Extract global stats from timeslots stats
        individual_status = StatusList(x.solving_stats.status
                                       for x in self._solutions.values())
        global_status = ("overfull"
                         if any(x == "overfull" for x in individual_status)
                         else "optimal")
        global_creation_time = sum(x.solving_stats.creation_time
                                   for x in self._solutions.values())
        global_solving_time = sum(x.solving_stats.solving_time
                                  for x in self._solutions.values())
        global_cost = sum(self._solutions[l].solving_stats.optimal_cost
                          for l in self.problem.workload)
        default_frac_gap = self.solver.fracGap
        default_max_seconds = self.solver.maxSeconds
        workload = np.array(self.problem.workload)

        global_stats = (SolvingStatsII(
                                  workload=workload,
                                  default_frac_gap=default_frac_gap,
                                  default_max_seconds=default_max_seconds,
                                  global_creation_time=global_creation_time,
                                  global_solving_time=global_solving_time,
                                  global_status=global_status,
                                  global_cost=global_cost,
                                  individual_status=individual_status
                                  )
                        )
        # Compose single allocation dataframe from timeslots allocations

        # Extract the allocation of reserved instances from phase I
        # Since this allocation is identical for any load level, we
        # arbitrarily take the first one (iloc[0])
        alloc_I = self.phase_I_solution.allocation
        rsv_instances_I = list(x for x in alloc_I.columns if x.reserved)
        reserved_alloc = alloc_I[rsv_instances_I].iloc[0]
        # Extend this allocation for all timeslots
        full_period_rsv_alloc = np.repeat([reserved_alloc.values],
                                          len(self.problem.workload),
                                          axis=0)
        # Extract the allocation of ondemand instances for each timeslot
        # Iterating for each timeslot, we lookup in the dictionary of
        # solutions the one for the workloadlevel in that timeslot
        full_period_dem_alloc = (np.array(
                     [self._solutions[l].allocation.values[0]
                      for l in self.problem.workload])
                     )

        # The extracted data is in form of numpy array. We join
        # both in a single dataframe with appropiate column names
        # (the instance classes)
        allocation = pd.DataFrame(np.append(full_period_rsv_alloc,
                                            full_period_dem_alloc, axis=1),
                                  columns=rsv_instances_I +
                                  self.ondemand_instances
                                  )

        self.solution = SolutionII(self.problem, global_stats, allocation)


class SolutionII(Solution):
    def get_cost_and_perf_dataframe(self):
        """This function is overriden because Phase II solution does
        not use any histogram, nor a list of allocations per load-level,
        but a list of allocations per timeslot"""
        if self._full_dataframe is not None:
            return self._full_dataframe
        solution = self.get_allocation_with_data()
        vm_numbers = solution.iloc[:, 1:-2].sort_index()
        costs = vm_numbers.T * solution.price
        perfs = vm_numbers.T * solution.perf
        self._full_dataframe = pd.concat([costs, perfs], axis=1,
                                         keys=("Cost", "Performance"))
        return self._full_dataframe

    def __repr__(self):
        return "{} solution with global ondemand cost {}".format(
                self.solving_stats.global_status,
                self.solving_stats.global_cost
                )


def get_load_hist_from_load(load, max_bins=None):
    """This function returns a load histogram from the load expressed as
    a list of load at each interval during the mission time. If no
    number_of_bins is given, a bin for each load level between the maximum
    and the minimum will be used.

    The histogram is represented with a dictionary where the key is the
    load level and the value is the number of instants that that load
    level was found during the mission time.

    This function is useful to convert a trace of the load for each period
    into the histogram required for the optimization problem, as expected
    by the constructor of Llovia.
    """
    load = np.array(load)

    if max_bins is None:
        number_of_bins = int(max(load))
    else:
        bin_size = (max(load) - min(load)) / max_bins
        number_of_bins = int(max(load) / bin_size)

    epsilon = 1e-6
    h = np.histogram(load-epsilon, bins=number_of_bins,
                     range=(0, max(load)))

    # Convert it to pandas for taking out the values not observed
    p = pd.DataFrame(h[0], index=h[1][1:], columns=["Count"])
    d = p[p.Count > 0]     # take out values not observed

    return LlooviaHistogram(d.to_records())


class LlooviaMaxPerformance:
    """This class solves a linear programming problem that takes as
    input a set of Instance Classes and obtains the maximum performance
    they can give. The VMs have to fulfill their limiting set restrictions.
    There is also an optional maximum cost restriction.
    """
    def __init__(self, instances, title="Optimize performance",
                 maximum_cost=None, relaxed=False):
        """Initalizes the optimization problem. The parameters are:
        'instances': list of InstanceClass to consider in the deployment. They
              must be ony on-demand instances
        'maximum_cost': restriction for the maximum cost. Can be None
        'relaxed': boolean; if True, the problem uses continuous variables
              instead of integer ones
        'title': optional title for the linear programming problem.
        """
        self.instances = instances
        self.title = title
        self.maximum_cost = maximum_cost
        self.relaxed = relaxed

        # Check that no instance is reserved
        for i in self.instances:
            if i.reserved:
                raise Exception("Only on-demand instances can be used")

        # Compute the set of LimitingSets (clouds),
        # extracted from the instances
        self.clouds = set()
        for i in instances:
            self.clouds.add(i.cloud)

    def create_variables(self):
        """Creates the set of variables Y* and X* of the basic problem.
        Override it if you need to create extra variables (first use
        super().create_variables() to call the base clase method)"""
        if self.relaxed:
            kind = LpContinuous
        else:
            kind = LpInteger
        self.vms = LpVariable.dicts('X', self.instances, 0, None, kind)

    def objective_function(self):
        """Adds to the problem the function to optimize, which is the
        total performance of the deployment.
        """
        self.prob += lpSum([self.vms[_ic] * _ic.performance
                            for _ic in self.instances]
                           ), "Objective: maximize performance"

    def create_problem(self):
        """This method creates the PuLP problem, and calls other
        methods to add variables and restrictions to it.

        It initializes the following attributes, which can be
        useful in derived classes:

          self.prob: instance of the PuLP problem
        """

        # Create the linear programming problem
        self.prob = LpProblem(self.title, LpMaximize)

        # Create the linear programming variables for pulp
        self.create_variables()

        # Create the goal function
        self.objective_function()

        # Add all restrictions indicated with functions *_restriction
        # in this class
        self.add_all_restrictions()

        return self.prob

    def add_all_restrictions(self):
        """This functions uses introspection to discover all implemented
        methods whose name ends with `_restriction`, and runs them all"""
        for name in dir(self):
            attribute = getattr(self, name)
            if ismethod(attribute) and name.endswith("_restriction"):
                attribute()

    def cost_restriction(self):
        """Adds to the PuLP problem the restriction of cost, which
        consists on forcing that the cost of all deployed machines
        is less than the maximum_cost.
        """
        if self.maximum_cost is None:
            return

        self.prob += lpSum([self.vms[_ic] * _ic.price
                            for _ic in self.instances] <=
                           self.maximum_cost),\
            "Maximum cost"

    def limit_instances_per_class_restriction(self):
        """If the instance has a max_vms attribute, this is a limit for that
        variable."""
        for ic in self.instances:
            if ic.max_vms > 0:
                # There is a limit per instance class
                self.prob += lpSum(
                        self.vms[ic]) <= ic.max_vms, \
                        "Max instances for on-demand instance class %r" % (ic)

    def limit_instances_per_limiting_set_restriction(self):
        """If the limiting set provides a max_vms > 0, then the sum of all
        instances in that limiting set should be limited to that maximum"""
        for cloud in self.clouds:
            if cloud.max_vms > 0:
                self.prob += lpSum([self.vms[ic] for ic in self.instances
                                    if ic.cloud == cloud]) <= cloud.max_vms,\
                             "Max instances for limiting set %r" % (cloud)

    def limit_cores_per_limiting_set_restriction(self):
        """If the limiting set provides a max_cores > 0, then the sum of all
        instance cores in that region should be limited to that maximum"""
        for cloud in self.clouds:
            if cloud.max_cores > 0:
                self.prob += lpSum([self.vms[ic]*ic.cores
                                    for ic in self.instances
                                    if ic.cloud == cloud]) <=\
                                   cloud.max_cores,\
                                   "Max cores for limiting set %r" % (cloud)

    def solve(self, *args, **kwargs):
        """Calls PuLP solver. Accepts the same arguments as a pulp solver"""
        return self.prob.solve(*args, **kwargs)

    def cost(self):
        """Gets the cost of the deployment obtained after solving it"""
        if self.prob.status != pulp.LpStatusOptimal:  # Not solved
            raise Exception("Cannot get the cost of an unsolved problem")
        res = 0
        for i in self.instances:
            res += self.vms[i].varValue * i.price
        return res

    def max_perf(self):
        """Gets the maximum performance found"""
        if self.prob.status != pulp.LpStatusOptimal:  # Not solved
            raise Exception("Cannot get the performance "
                            "of an unsolved problem")
        res = 0
        for i in self.instances:
            res += self.vms[i].varValue * i.performance
        return res

    def get_soldf(self, only_used=False):
        """Returns the solution as a DataFrame. Each columns is an instance
        clases.  If only_used is True, instance classes never used are not
        included in the DataFrame.
        """

        if self.prob.status != pulp.LpStatusOptimal:  # Not solved
            raise Exception("Cannot get the solution for an unsolved problem")

        sol = OrderedDict()
        for i in self.instances:
            sol[i] = self.vms[i].varValue

        soldf = pd.DataFrame([sol])

        if only_used:
            return soldf[soldf.columns[(soldf != 0).any()]]
        else:
            return soldf

# The following function is used to monkeypatch part of PuLP code.
# This modification is aimed to get the value of the optimal best bound
# which is provided by CBC solver as part of the solution, even if
# the solution could not be found due to a time limit
#
# PuLP does not recover this value, but for our analysis is useful
# to estimate the worst-case error of our approximation when the
# exact solution cannot be found in a reasonable time.
#
# The code patches the part in which PuLP calls CBC, so that the standard
# output of CBC is redirected to a logfile. When CBC exits, the code
# inspects the logfile and locates the bestBound value, storing it
# as part of the problem to make it accessible to the python code.
#
# This patch only works when the solver is COIN.


def solve_CBC_patched(self, lp, use_mps=True):
    """Solve a MIP problem using CBC, patched from original PuLP function
    to save a log with cbc's output and take from it the best bound"""

    def takeBestBoundFromLog(filename):
        try:
            f = open(filename, "r")
        except:
            return None
        else:
            for l in f:
                if l.startswith("Lower bound:"):
                    return float(l.split(":")[-1])
            return None

    if not self.executable(self.path):
        raise PulpSolverError("Pulp: cannot execute %s cwd: %s" % (self.path,
                              os.getcwd()))
    if not self.keepFiles:
        pid = os.getpid()
        tmpLp = os.path.join(self.tmpDir, "%d-pulp.lp" % pid)
        tmpMps = os.path.join(self.tmpDir, "%d-pulp.mps" % pid)
        tmpSol = os.path.join(self.tmpDir, "%d-pulp.sol" % pid)
    else:
        tmpLp = lp.name+"-pulp.lp"
        tmpMps = lp.name+"-pulp.mps"
        tmpSol = lp.name+"-pulp.sol"
    if use_mps:
        vs, variablesNames, constraintsNames, objectiveName = lp.writeMPS(
                    tmpMps, rename=1)
        cmds = ' '+tmpMps+" "
        if lp.sense == LpMaximize:
            cmds += 'max '
    else:
        lp.writeLP(tmpLp)
        cmds = ' '+tmpLp+" "
    if self.threads:
        cmds += "threads %s " % self.threads
    if self.fracGap is not None:
        cmds += "ratio %s " % self.fracGap
    if self.maxSeconds is not None:
        cmds += "sec %s " % self.maxSeconds
    if self.presolve:
        cmds += "presolve on "
    if self.strong:
        cmds += "strong %d " % self.strong
    if self.cuts:
        cmds += "gomory on "
        # cbc.write("oddhole on "
        cmds += "knapsack on "
        cmds += "probing on "
    for option in self.options:
        cmds += option+" "
    if self.mip:
        cmds += "branch "
    else:
        cmds += "initialSolve "
    cmds += "printingOptions all "
    cmds += "solution "+tmpSol+" "
    # if self.msg:
    #     pipe = None
    # else:
    #     pipe = open(os.devnull, 'w')
    pipe = open(tmpLp + ".log", 'w')
    log.debug(self.path + cmds)
    cbc = subprocess.Popen((self.path + cmds).split(), stdout=pipe,
                           stderr=pipe)
    if cbc.wait() != 0:
        raise PulpSolverError("Pulp: Error while trying to execute " +
                              self.path)
    if not os.path.exists(tmpSol):
        raise PulpSolverError("Pulp: Error while executing "+self.path)
    if use_mps:
        lp.status, values, reducedCosts, shadowPrices, slacks =\
                self.readsol_MPS(tmpSol, lp, lp.variables(),
                                 variablesNames, constraintsNames,
                                 objectiveName)
    else:
        lp.status, values, reducedCosts, shadowPrices, slacks =\
                self.readsol_LP(tmpSol, lp, lp.variables())
    lp.assignVarsVals(values)
    lp.assignVarsDj(reducedCosts)
    lp.assignConsPi(shadowPrices)
    lp.assignConsSlack(slacks, activity=True)
    lp.bestBound = takeBestBoundFromLog(tmpLp + ".log")
    if not self.keepFiles:
        try:
            os.remove(tmpMps)
        except:
            pass
        try:
            os.remove(tmpLp)
        except:
            pass
        try:
            os.remove(tmpSol)
        except:
            pass
    return lp.status


# Monkeypatching
COIN_CMD.solve_CBC = solve_CBC_patched
