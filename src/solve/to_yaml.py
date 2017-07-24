import os
import sys
import argparse
import pickle

import jsonschema
import yaml

import lloovia
import lloovia_yaml

def convert_files(command, directory):
    if directory[-1] == '/' or directory[-1] == '\\':
        directory = directory[:-1]

    num_files = sum([len(files) for r, d, files in os.walk(directory)])
    num_processed = 0
    num_skipped = 0
    for root, dirs, files in os.walk(directory):
        for name in files:
            filename, file_extension = os.path.splitext(name)

            if file_extension != '.pickle':
                print('Skipping {}. Extension is not .pickle'.format(name))
                num_skipped += 1
                continue

            try:
                num_processed += 1
                print('Converting {} ({}/{})'.format(name, num_processed, num_files))
                convert(command, root, filename)
            except Exception as exception:
                print('Skipping {}. Exception: {}'.format(name, exception))
                num_skipped += 1

    print('Number of processed files:', num_processed)
    print('Number of skipped files:', num_skipped)

def convert(command, root_dir, base_filename):
    pickle_name = root_dir + '/' + base_filename + ".pickle"

    if command == 'problems':
        schema_name = "problem.schema.yaml"
    elif command == 'solutions':
        schema_name = "malloovia.schema.yaml"
    else:
        raise ValueError("The command must be 'problems' or 'solutions'")

    with open(schema_name) as schema_file:
        yaml_schema = yaml.safe_load(schema_file)

        with open(pickle_name, "rb") as file:
            obj = pickle.load(file)

            if not (isinstance(obj, lloovia.Solution) or isinstance(obj, lloovia.Problem)):
                raise ValueError('Not a lloovia problem or solution')

            converter = lloovia_yaml.Converter()

            if isinstance(obj, lloovia.Solution):
                if command == 'solutions':
                   yaml_output = converter.solutions_to_yaml([obj])
                elif command == 'problems':
                    yaml_output = converter.problems_to_yaml([obj.problem])
            else: # It is a lloovia.Problem
                if command == 'solutions':
                    raise ValueError('Cannot save a Problem as Solution')
                else: # Command and object are problems
                    yaml_output = converter.problems_to_yaml([obj])

            converted_yaml = yaml.safe_load(yaml_output)
            
            try:
                jsonschema.validate(converted_yaml, schema=yaml_schema)
            except jsonschema.ValidationError as exception:
                print('Validation error for', pickle_name)
                print('error:', exception.message)
                return

        yaml_name = root_dir + '/' + base_filename + ".yaml"
        with open(yaml_name, "w") as file:
            file.write(yaml_output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert lloovia pickle files with problems or solutions to YAML')
    subparser = parser.add_subparsers(help='command', dest='command')
    subparser.required = True

    parser_solutions = subparser.add_parser('solutions', help='convert solutions')
    parser_problems = subparser.add_parser('problems', help='convert problems')

    parser.add_argument('root_dir', help='root directory to look for lloovia .pickle files',
                        default='convert_files')

    args = parser.parse_args()

    convert_files(args.command, args.root_dir)

