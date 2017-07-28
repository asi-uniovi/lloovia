"""Scritp to recursively convert lloovia problems and solutions in a directory to YAML"""
import os
import argparse
import pickle

import jsonschema
import yaml
import colorama

import lloovia
import lloovia_yaml

def convert_files(command, directory):
    if directory[-1] == '/' or directory[-1] == '\\':
        directory = directory[:-1]

    num_files = sum([len(files) for r, d, files in os.walk(directory)])
    num_processed = 0
    num_skipped = 0
    num_errors = 0
    for root, _, files in os.walk(directory):
        for name in files:
            filename, file_extension = os.path.splitext(name)

            if file_extension != '.pickle':
                print(YELLOW + 'Skipping {}. Extension is not .pickle'.format(name),
                      RESET)
                num_skipped += 1
                continue

            try:
                num_processed += 1
                print('Converting {} ({}/{})'.format(name, num_processed, num_files))
                convert(command, root, filename)
            except Exception as exception:
                print(RED + '    Skipping {}. Exception: {}'.format(name, exception),
                      RESET)
                num_skipped += 1
                num_errors += 1

    print('\nNumber of processed files:', num_processed)
    print('Number of skipped files:', num_skipped)
    print('Number of files with errors:', num_errors)

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
                raise ValueError('Not a lloovia problem or solution. Type: {}'.format(
                    type(obj)))

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
                print(RED + 'Validation error for', pickle_name)
                print('error:', exception.message, RESET)
                return

        yaml_name = root_dir + '/' + base_filename + ".yaml"
        with open(yaml_name, "w") as file:
            file.write(yaml_output)

def main():

    parser = argparse.ArgumentParser(
        description='Convert lloovia pickle files with problems or solutions to YAML')
    subparser = parser.add_subparsers(help='command', dest='command')
    subparser.required = True

    subparser.add_parser('solutions', help='convert solutions')
    subparser.add_parser('problems', help='convert problems')

    parser.add_argument('root_dir', help='root directory to look for lloovia .pickle files',
                        default='convert_files')

    args = parser.parse_args()

    convert_files(args.command, args.root_dir)

if __name__ == '__main__':
    colorama.init()
    YELLOW = colorama.Fore.YELLOW
    RED = colorama.Fore.RED
    RESET = colorama.Style.RESET_ALL

    main()
