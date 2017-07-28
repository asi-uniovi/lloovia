"""Converter from lloovia to YAML"""

import typing
import re
from functools import lru_cache

import lloovia

def problems_to_yaml(problems: typing.List[lloovia.Problem]):
    '''Converts a list of lloovia problems to a YAML string.'''
    return _Converter().problems_to_yaml(problems)

def solutions_to_yaml(solutions: typing.List[lloovia.Solution]):
    '''Converts a list of lloovia solutions to a YAML string.'''
    return _Converter().solutions_to_yaml(solutions)

class _Converter(object):
    """Converts lloovia problems and solutions to YAML."""
    def __init__(self):
        self._workloads_lines = []
        self._apps_lines = []
        self._problem_lines = []
        self._limiting_sets_lines = []
        self._instances_lines = []
        self._perf_lines = []
        self._solution_lines = []

        self._ic_id_factory = _IdFactory(prefix="ic")
        self._ls_id_factory = _IdFactory(prefix="ls")
        self._solution_id_factory = _IdFactory(prefix="sol")
        self._problem_id_factory = _IdFactory(prefix="problem")

    def _compose_problem_lines(self):
        return ['Limiting_sets:', *self._limiting_sets_lines,
                'Instance_classes:', *self._instances_lines,
                'Apps:', *self._apps_lines,
                'Workloads:', *self._workloads_lines,
                'Performances:', *self._perf_lines,
                'Problems:', *self._problem_lines]

    def problems_to_yaml(self, problems: typing.List[lloovia.Problem]):
        '''Converts a list of lloovia problems to a YAML string.'''
        self._process_problems(problems)
        return "\n".join(self._compose_problem_lines())

    def solutions_to_yaml(self, solutions: typing.List[lloovia.Solution]):
        '''Converts a list of lloovia solutions to a YAML string.'''
        problems = []
        for solution in solutions:
            problems.append(solution.problem)

        self._process_problems(problems)

        for solution in solutions:
            self._process_solution(solution)

        lines = [*self._compose_problem_lines(),
                 *self._compose_solution_lines()]

        return "\n".join(lines)

    @staticmethod
    def _none_to_null(value):
        '''Changes "None" to "null" because YAML requires "null" instead of "None".'''
        if value is None:
            return "null"

        return value

    @staticmethod
    def _adapt_status(original_status):
        '''Changes "unknown_error" to "cbc_error" because that's the name in the schema.'''
        if original_status == "unknown_error":
            return "cbc_error"

        return original_status

    @staticmethod
    def _generate_algorithm_lines(solution):
        stats = solution.solving_stats

        binning = stats.max_bins != None
        if binning:
            histogram = lloovia.get_load_hist_from_load(solution.problem.workload, stats.max_bins)
            effective_bins = len(histogram)
            binning_lines = ([
                '          n_bins: {}'.format(stats.max_bins),
                '          effective_bins: {}'.format(effective_bins),
                ])
        else:
            binning_lines = []

        return ([
            '      algorithm:',
            '        lloovia:',
            '          binning: {}'.format(binning),
            '          status: {}'.format(_Converter._adapt_status(stats.status)),
            *binning_lines,
            '          frac_gap: {}'.format(_Converter._none_to_null(stats.frac_gap)),
            '          max_seconds: {}'.format(_Converter._none_to_null(stats.max_seconds)),
            '          lower_bound: {}'.format(_Converter._none_to_null(stats.lower_bound)),
            ])

    @staticmethod
    def _generate_phase_i_solving_stats_lines(solution):
        stats = solution.solving_stats

        return ([
            '      optimal_cost: {}'.format(_Converter._none_to_null(stats.optimal_cost)),
            '      creation_time: {}'.format(stats.creation_time),
            '      solving_time: {}'.format(stats.solving_time),
            *_Converter._generate_algorithm_lines(solution)
            ])

    @staticmethod
    def _generate_global_solving_stats_lines(solution):
        stats = solution.solving_stats

        return ([
            '      optimal_cost: {}'.format(_Converter._none_to_null(stats.global_cost)),
            '      creation_time: {}'.format(stats.global_creation_time),
            '      solving_time: {}'.format(stats.global_solving_time),
            '      status: {}'.format(stats.global_status)
            ])

    def _generate_reserved_allocation_lines(self, solution):
        status = solution.solving_stats.status
        if status == 'infeasible' or status == 'unknown_error' or status == 'aborted':
            return ['      instance_classes: []',
                    '      vms_number: []']

        df_allocation = solution.get_allocation(only_used=True)
        df_allocation.columns.name = 'VM'
        df_with_res_info = df_allocation.T.reset_index().assign(
            reserved=lambda x: x.VM.map(lambda x: x.reserved))
        df_res = df_with_res_info[df_with_res_info.reserved]

        ics = []
        vms_numbers = []
        for i, instance_class in enumerate(df_res.VM):
            ic_id = self._ic_id_factory.get_id(str(instance_class))
            ics.append('*{}'.format(ic_id))
            vms_numbers.append(df_res.iloc[i, 1])

        return ['      instance_classes: [{}]'.format(', '.join(ics)),
                '      vms_number: {}'.format(vms_numbers)]

    def _generate_allocation_lines(self, solution):
        df_allocation = solution.get_allocation(only_used=True)

        instance_classes = []
        ic_ids = [] # instance classes ids
        for i in df_allocation.columns:
            instance_classes.append(i)

            ic_id = self._ic_id_factory.get_id(str(i))
            ic_ids.append('*{}'.format(ic_id))

        # These values only make sense for phase I solutions
        workload_tuples = []
        repeats = []
        workload_histogram = solution.solving_stats.workload

        vms_numbers = [] # list for each workload level/timeslot with the list of vms_numbers
        for index, row in df_allocation.iterrows():
            workload_level = index
            workload_tuples.append('[{}]'.format(workload_level))
            repeats.append(str(workload_histogram[workload_level]))

            workload_level_allocation = []
            for i in instance_classes:
                workload_level_allocation.append(row[i])

            vms_numbers.append(workload_level_allocation)

        # Get a detailed description of the workload for phase I solutions.
        # Phase II solutions have the workload for each timeslot
        if isinstance(solution, lloovia.SolutionII):
            workload_desc_lines = []
        else: # SolutionI
            workload_desc_lines = [
                '      workload_tuples: [{}]'.format(', '.join(workload_tuples)),
                '      repeats: [{}]'.format(', '.join(repeats))
                ]

        result = ['      apps: [*App0]',
                  '      instance_classes: [{}]'.format(', '.join(ic_ids)),
                  *workload_desc_lines,
                  '      vms_number:']

        for i, vm_number in enumerate(vms_numbers):
            if isinstance(solution, lloovia.SolutionII):
                # Phase II solution uses timeslots
                comment = '# t: {}'.format(i)
            else:
                # Phase I solution uses workload levels
                comment = '# l: {}'.format(workload_tuples[i])

            result.extend(['        - {}'.format(comment),
                           '          - {}'.format(list(vm_number))])

        return result

    def _generate_solution_ii_particular_lines(self, solution):
        result = ([
            '    global_solving_stats:', *self._generate_global_solving_stats_lines(solution)
            ])

        status = solution.solving_stats.global_status
        if status not in ['infeasible', 'unknown_error', 'aborted']:
            result.extend([
                '    allocation:', *self._generate_allocation_lines(solution)
                ])

        return result


    def _generate_solution_i_particular_lines(self, solution):
        result = ([
            '    solving_stats:', *self._generate_phase_i_solving_stats_lines(solution),
            '    reserved_allocation:', *self._generate_reserved_allocation_lines(solution)
            ])

        status = solution.solving_stats.status
        if status not in ['infeasible', 'unknown_error', 'aborted']:
            result.extend([
                '    allocation:', *self._generate_allocation_lines(solution)
                ])

        return result

    def _process_solution(self, solution):
        if isinstance(solution, lloovia.SolutionII):
            phase = 2
            particular_lines = self._generate_solution_ii_particular_lines(solution)
        elif isinstance(solution, lloovia.Solution):
            # Solution is used instead of SolutionI because, although lloovia defines
            # a SolutionI class, it never uses it
            phase = 1
            particular_lines = self._generate_solution_i_particular_lines(solution)
        else:
            raise ValueError('Solution should be of type SolutionI or SolutionII. It is {}'.format(
                type(solution)))

        solution_id = self._solution_id_factory.get_id_from_object(solution)
        problem_id = self._problem_id_factory.get_id_from_object(solution.problem)

        self._solution_lines = ([
            '  - &{}'.format(solution_id),
            '    id: {}'.format(solution_id),
            '    phase: {}'.format(phase),
            '    problem: *{}'.format(problem_id),
            *particular_lines,
            ])

    def _compose_solution_lines(self):
        return ['Solutions:', *self._solution_lines]

    def _process_performance(self, app_id, problems):
        self._perf_lines = ([
            '    - &Performance1',
            '      id: Performance1',
            '      values:'])

        # take the first problem: the performance should be the same for all
        problem = problems[0]
        for instance in problem.instances:
            instance_id = str(instance)
            perf = instance.performance
            self._perf_lines.extend([
                '        - instance_class: *{}'.format(self._ic_id_factory.get_id(instance_id)),
                '          app: *{}'.format(app_id),
                '          value: {}'.format(perf)])

    def _process_problems(self, problems: typing.List[lloovia.Problem]):
        '''Receives a list of lloovia problems and returns a YAML string
        of problem descriptions.

        In lloovia, there is no concept of app, so a synthetich app
        will be created. In addtion, the performances are the same for
        all problems, so there is going to be only a Performance1 object.
        '''
        app_id = self._process_apps()

        # These sets are used to avoid duplications
        limiting_sets = set()
        instance_classes = set()

        workload_index = 0
        for problem in problems:
            self._process_problem(app_id, instance_classes, limiting_sets, problem,
                                  workload_index)
            workload_index += 1

        self._process_performance(app_id, problems)

    def _process_problem(self, app_id, instance_classes, limiting_sets, problem,
                         workload_index):
        workloads_per_problem = []

        problem_id = self._problem_id_factory.get_id_from_object(problem)
        workload_id = 'Workload' + str(workload_index)
        description = '    description: Workload for app {} in problem {}'
        self._workloads_lines.extend([
            '  - &{}'.format(workload_id),
            '    id: {}'.format(workload_id),
            description.format(app_id, problem_id),
            '    values: [{}]'.format(
                ", ".join(str(i) for i in problem.workload)),
            '    app: *{}'.format(app_id)])

        workloads_per_problem.append(workload_id)

        self._problem_lines.extend([
            '  - &{}'.format(problem_id),
            '    id: {}'.format(problem_id),
            '    name: {}'.format(problem_id),
            '    description: {}'.format(problem_id),
            '    instance_classes: [{}]'.format(
                ", ".join('*{}'.format(
                    self._ic_id_factory.get_id(str(i))) for i in problem.instances)),
            '    workloads: [*{}]'.format(workload_id),
            '    performances: *Performance1'])

        for instance in problem.instances:
            if instance in instance_classes:
                continue # Already generated
            instance_classes.add(instance)

            # This should be a unique identifier, unlike instance.name
            instance_id = self._ic_id_factory.get_id(str(instance))

            limiting_set_id = self._ls_id_factory.get_id(str(instance.cloud))

            self._instances_lines.extend([
                '  - &{}'.format(instance_id),
                '    id: {}'.format(instance_id),
                '    name: "{}"'.format(instance.name),
                '    max_vms: {}'.format(instance.max_vms),
                '    limiting_sets: [*{}]'.format(limiting_set_id),
                '    price: {}'.format(instance.price),
                '    is_reserved: {}'.format(instance.reserved)])

            if limiting_set_id in limiting_sets:
                continue # Already generated
            limiting_sets.add(limiting_set_id)

            self._limiting_sets_lines.extend([
                '  - &{}'.format(limiting_set_id),
                '    id: {}'.format(limiting_set_id),
                '    max_vms: {}'.format(instance.cloud.max_vms),
                '    max_cores: {}'.format(instance.cloud.max_cores)])

    def _process_apps(self):
        app_id = 'App0'
        app_name = app_id

        self._apps_lines.extend([
            '  - &{}'.format(app_id),
            '    id: {}'.format(app_id),
            '    name: "{}"'.format(app_name)])
        return app_id

class _IdFactory:
    '''Generates valid-YAML and unique identifiers from unique strings or objects'''
    def __init__(self, prefix="id"):
        self.count = 0
        self.prefix = prefix

    @lru_cache(maxsize=None)
    def get_id(self, name: str):
        '''Generates valid-YAML and unique identifiers from unique strings,
        which are assumed to be unique names that can be invalid YAML. The
        identifiers are a prefix ("id" by default), plus a number different
        for each input string plus the string itself converted to valid
        YAML.
        '''
        result = "{}{}-{}".format(self.prefix,
                                  self.count,
                                  _IdFactory.id_to_valid_yaml(name))
        self.count += 1
        return result

    @lru_cache(maxsize=None)
    def get_id_from_object(self, obj: object):
        '''Generates a unique identifier for an object. The argument obj is
        not directly used, but is required to have a differnt id for each object in the cache.
        '''
        result = "{}{}".format(self.prefix,
                               self.count)
        self.count += 1
        return result


    @staticmethod
    def id_to_valid_yaml(identifier):
        '''YAML ids have to be alphanumeric or numeric characters, so the
        invalid characters are substituted by "-".'''
        return re.sub('[^0-9a-zA-Z]', '-', identifier)
