import typing
import re
from functools import lru_cache

import lloovia

class Converter(object):
    """Converts lloovia problems and solutions to YAML."""
    def __init__(self):
        self._workloads_lines = []
        self._apps_lines = []
        self._problem_lines = []
        self._limiting_sets_lines = []
        self._instances_lines = []
        self._perf_lines = []

        self._ic_id_factory = _IdFactory(prefix="ic")
        self._ls_id_factory = _IdFactory(prefix="ls")


    def problems_to_yaml(self, problems: typing.List[lloovia.Problem]):
        self.__process_problems(problems)
        lines = ['Limiting_sets:', *self._limiting_sets_lines,
                 'Instance_classes:', *self._instances_lines,
                 'Apps:', *self._apps_lines,
                 'Workloads:', *self._workloads_lines,
                 'Performances:', *self._perf_lines,
                 'Problems:', *self._problem_lines]

        return "\n".join(lines)

    def __process_performance(self, app_id, problems):
        problem = problems[0]
        self._perf_lines = ([
            '    - &Performance1',
            '      id: Performance1',
            '      values:'])

        for instance in problem.instances:
            instance_id = str(instance)
            perf = instance.performance
            self._perf_lines.extend([
                '        - instance_class: *{}'.format(self._ic_id_factory.get_id(instance_id)),
                '          app: *{}'.format(app_id),
                '          value: {}'.format(perf)])

    def __process_problems(self, problems: typing.List[lloovia.Problem]):
        '''Receives a list of lloovia problems and returns a YAML string
        of problem descriptions.

        In lloovia, there is no concept of app, so a synthetich app
        will be created. In addtion, the performances are the same for
        all problems, so there is going to be only a Performance1 object.
        '''
        # These sets are used to avoid duplications
        limiting_sets = set()
        instance_classes = set()

        app_id = self.__process_apps()

        workload_index = 0
        for i_problem, problem in enumerate(problems):
            self.__process_problem(app_id, i_problem, instance_classes, limiting_sets, problem,
                                   workload_index)
            workload_index += 1

        self.__process_performance(app_id, problems)

    def __process_problem(self, app_id, i_problem, instance_classes, limiting_sets, problem,
                          workload_index):
        workloads_per_problem = []

        workload_id = 'Workload' + str(workload_index)
        description = '    description: Workload for app {} in problem Problem{}'
        self._workloads_lines.extend([
            '  - &{}'.format(workload_id),
            '    id: {}'.format(workload_id),
            description.format(app_id, str(i_problem)),
            '    values: [{}]'.format(
                ", ".join(str(i) for i in problem.workload)),
            '    app: *{}'.format(app_id)])

        workloads_per_problem.append(workload_id)

        problem_id = 'Problem' + str(i_problem)
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

    def __process_apps(self):
        app_id = 'App0'
        app_name = app_id

        self._apps_lines.extend([
            '  - &{}'.format(app_id),
            '    id: {}'.format(app_id),
            '    name: "{}"'.format(app_name)])
        return app_id

class _IdFactory:
    '''Generates valid-YAML and unique identifiers from strings, which
    are assumed to be unique names that can be invalid YAML. The
    identifiers are a prefix ("id" by default), plus a number different
    for each input string plus the string itself converted to valid
    YAML.
    '''
    def __init__(self, prefix="id"):
        self.count = 0
        self.prefix = prefix

    @lru_cache(maxsize=None)
    def get_id(self, name: str):
        result = "{}{}-{}".format(self.prefix,
                                  self.count, _IdFactory.id_to_valid_yaml(name))
        self.count += 1
        return result

    @staticmethod
    def id_to_valid_yaml(identifier):
        '''YAML ids have to be alphanumeric or numeric characters, so the
        invalid characters are substituted by "-".'''
        return re.sub('[^0-9a-zA-Z]', '-', identifier)
