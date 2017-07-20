﻿'''Module to test conversion from lloovia to YAML'''
import unittest
import yaml
import jsonschema

import lloovia
import lloovia_yaml

class TestLlooviaYaml(unittest.TestCase):
    '''Tests functions to convert to YAML'''

    def setUp(self):
        ls_us_east = lloovia.LimitingSet("US East (N. California)", max_vms=20, max_cores=0)
        ls_us_west_m4 = lloovia.LimitingSet("us_west_m4", max_vms=5, max_cores=0)

        ic1 = lloovia.InstanceClass(name="m3.large", cloud=ls_us_east, max_vms=20,
                                    reserved=True, price=2.3, performance=5)

        ic2 = lloovia.InstanceClass(name="m4.medium", cloud=ls_us_west_m4, max_vms=5,
                                    reserved=False, price=4.4, performance=6)

        ic3 = lloovia.InstanceClass(name="m4.large", cloud=ls_us_west_m4, max_vms=5,
                                    reserved=False, price=8.2, performance=7)

        instances = [ic1, ic2, ic3]

        workload_phase_i = [1, 22, 5, 6, 10, 20, 50, 20]
        workload_phase_ii = [5, 2, 9, 9, 99, 999, 88, 60]

        self.problem_phase_i = lloovia.Problem(instances=instances, workload=workload_phase_i)

        self.problem_phase_ii = lloovia.Problem(instances=instances, workload=workload_phase_ii)

    def test_problems_to_yaml(self):
        '''Test that problems_to_yaml() can create a valid YAML file with two simple problems,
        one for Phase I and another for Phase II.
        '''
        converter = lloovia_yaml.Converter()
        yaml_output = converter.problems_to_yaml(
            [self.problem_phase_i, self.problem_phase_ii])

        converted_yaml = yaml.safe_load(yaml_output)

        with open("problem.schema.yaml") as file:
            yaml_schema = yaml.safe_load(file)
        jsonschema.validate(converted_yaml, schema=yaml_schema)

if __name__ == '__main__':
    unittest.main()