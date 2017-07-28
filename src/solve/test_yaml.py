'''Module to test conversion from lloovia to YAML'''
import unittest
import yaml
import jsonschema
import pandas as pd

import lloovia
import lloovia_yaml

class TestLlooviaYaml(unittest.TestCase):
    '''Tests functions to convert to YAML'''

    def prepare_problems(self, workload_phase_i, workload_phase_ii):
        '''Creates a problem for phase I and another for phase II as fields
        of the class. They use the workload from the parameters. The instances
        are stored in self.instances.'''
        ls_us_east = lloovia.LimitingSet("US East (N. California)", max_vms=20, max_cores=0)
        ls_us_west_m4 = lloovia.LimitingSet("us_west_m4", max_vms=5, max_cores=0)

        ic1 = lloovia.InstanceClass(name="m3.large", cloud=ls_us_east, max_vms=20,
                                    reserved=True, price=2.3, performance=5)

        ic2 = lloovia.InstanceClass(name="m4.medium", cloud=ls_us_west_m4, max_vms=5,
                                    reserved=False, price=4.5, performance=6)

        ic3 = lloovia.InstanceClass(name="m4.large", cloud=ls_us_west_m4, max_vms=5,
                                    reserved=False, price=8.2, performance=7)
        self.instances = [ic1, ic2, ic3]

        self.problem_phase_i = lloovia.Problem(instances=self.instances,
                                               workload=workload_phase_i)
        self.problem_phase_ii = lloovia.Problem(instances=self.instances,
                                                workload=workload_phase_ii)

    @staticmethod
    def convert_and_validate_solution(solution_phase_i):
        '''Converts a solution to YAML and validates that the schema is followed.'''
        yaml_output = lloovia_yaml.solutions_to_yaml([solution_phase_i])

        converted_yaml = yaml.safe_load(yaml_output)

        with open('malloovia.schema.yaml') as file:
            yaml_schema = yaml.safe_load(file)
        jsonschema.validate(converted_yaml, schema=yaml_schema)

    def test_problems_to_yaml(self):
        '''Test that problems_to_yaml() can create a valid YAML file with two simple problems,
        one for Phase I and another for Phase II.
        '''
        workload_phase_i = [1, 22, 5, 6, 10, 20, 50, 20]
        workload_phase_ii = [5, 2, 9, 9, 99, 999, 88, 60]

        self.prepare_problems(workload_phase_i, workload_phase_ii)

        yaml_output = lloovia_yaml.problems_to_yaml(
            [self.problem_phase_i, self.problem_phase_ii])

        converted_yaml = yaml.safe_load(yaml_output)

        with open("problem.schema.yaml") as file:
            yaml_schema = yaml.safe_load(file)
        jsonschema.validate(converted_yaml, schema=yaml_schema)

    def test_optimal_solution_to_yaml(self):
        '''Tests that solutions_to_yamls() can create a valid YAML file with optimal solutions.'''
        workload_phase_i = [1, 22, 5, 6, 10, 20, 50, 20]
        workload_phase_ii = [5, 2, 9, 9, 99, 999, 88, 60]

        self.prepare_problems(workload_phase_i, workload_phase_ii)

        load_hist = lloovia.get_load_hist_from_load(workload_phase_i)
        solving_stats = lloovia.SolvingStatsI(max_bins=None,
                                              workload=load_hist,
                                              frac_gap=None,
                                              max_seconds=600,
                                              creation_time=123.4,
                                              solving_time=5421.98,
                                              status='optimal',
                                              lower_bound=None,
                                              optimal_cost=100.6
                                             )

        allocation = pd.DataFrame({'load': [1, 5, 6, 10, 20, 22, 50],
                                   self.instances[0]: ([4]*7),
                                   self.instances[1]: ([0, 0, 0, 0, 0, 1, 5]),
                                   self.instances[2]: ([0]*7)}).set_index('load', drop=True)

        solution_phase_i = lloovia.SolutionI(
            problem=self.problem_phase_i,
            solving_stats=solving_stats,
            allocation=allocation)

        TestLlooviaYaml.convert_and_validate_solution(solution_phase_i)

    def test_infeasible_solution_to_yaml(self):
        '''Tests that solutions_to_yamls() can create a valid YAML file with infeasible
        solutions.'''
        workload_phase_i = [1, 22, 5, 6, 10, 20, 50, 2000]
        workload_phase_ii = [5, 2, 9, 9, 99, 999, 88, 60]

        self.prepare_problems(workload_phase_i, workload_phase_ii)

        load_hist = lloovia.get_load_hist_from_load(workload_phase_i)
        solving_stats = lloovia.SolvingStatsI(max_bins=None,
                                              workload=load_hist,
                                              frac_gap=None,
                                              max_seconds=None,
                                              creation_time=0.0852697420325198,
                                              solving_time=0.053043919593364935,
                                              status='infeasible',
                                              lower_bound=None,
                                              optimal_cost=None
                                             )

        solution_phase_i = lloovia.SolutionI(
            problem=self.problem_phase_i,
            solving_stats=solving_stats,
            allocation=None)

        TestLlooviaYaml.convert_and_validate_solution(solution_phase_i)

    def test_aborted_solution_to_yaml(self):
        '''Tests that solutions_to_yamls() can create a valid YAML file with aborted solutions.'''
        workload_phase_i = [1, 22, 5, 6, 10, 20, 50, 2000]
        workload_phase_ii = [5, 2, 9, 9, 99, 999, 88, 60]

        self.prepare_problems(workload_phase_i, workload_phase_ii)

        load_hist = lloovia.get_load_hist_from_load(workload_phase_i, max_bins=6000)
        solving_stats = lloovia.SolvingStatsI(max_bins=6000,
                                              workload=load_hist,
                                              frac_gap=0.01,
                                              max_seconds=600,
                                              creation_time=15.493958642000052,
                                              solving_time=338.72762610200016,
                                              status='aborted',
                                              lower_bound=501.1,
                                              optimal_cost=None
                                             )

        solution_phase_i = lloovia.SolutionI(
            problem=self.problem_phase_i,
            solving_stats=solving_stats,
            allocation=None)

        TestLlooviaYaml.convert_and_validate_solution(solution_phase_i)

    def test_optimal_solution_phase2_to_yaml(self):
        '''Tests that solutions_to_yamls() can create a valid YAML file with a phaseII
        solution where all the individual statuses are trivial or optimal.'''
        workload_phase_i = [1, 22, 5, 6, 10, 20, 50, 20]
        workload_phase_ii = [5, 2, 9, 9, 11, 5, 11, 50]

        self.prepare_problems(workload_phase_i, workload_phase_ii)

        allocation = {
            self.instances[0]: {0: 4.0, 1: 4.0, 2: 4.0, 3: 4.0, 4: 4.0, 5: 4.0, 6: 4.0, 7: 4.0},
            self.instances[1]: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 5.0},
            }

        allocation_df = pd.DataFrame.from_dict(allocation)

        solving_stats = lloovia.SolvingStatsII(workload=workload_phase_ii,
                                               default_frac_gap=None,
                                               default_max_seconds=None,
                                               global_creation_time=0.007148156645641279,
                                               global_solving_time=0.03285847801969055,
                                               global_status='optimal',
                                               global_cost=22.5,
                                               individual_status=lloovia.StatusList(
                                                   ['trivial']*7 + ['optimal'])
                                              )

        solution_phase_ii = lloovia.SolutionII(
            problem=self.problem_phase_i,
            solving_stats=solving_stats,
            allocation=allocation_df)

        TestLlooviaYaml.convert_and_validate_solution(solution_phase_ii)

if __name__ == '__main__':
    unittest.main()
