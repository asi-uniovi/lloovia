import os
import sys
import pickle

import jsonschema
import yaml

import lloovia
import lloovia_yaml

def convert(root_dir, base_filename):
    pickle_name = root_dir + '/' + base_filename + ".pickle"

    with open("malloovia.schema.yaml") as schema_file:
        yaml_schema = yaml.safe_load(schema_file)

        with open(pickle_name, "rb") as file:
            solution = pickle.load(file)
            if not isinstance(solution, lloovia.Solution):
                raise Exception('Not a lloovia solution')

            converter = lloovia_yaml.Converter()
            yaml_output = converter.solutions_to_yaml([solution])
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
    if len(sys.argv) != 2:
        print("Missing root directory. Usage: convert_problems root_dir")
        sys.exit(1)

    directory = sys.argv[1]
    if directory[-1] == '/' or directory[-1] == '\\':
        directory = directory[:-1]

    num_files = sum([len(files) for r, d, files in os.walk(directory)])
    num_processed = 0
    num_skipped = 0
    for root, dirs, files in os.walk(directory):
        for name in files:
            filename, file_extension = os.path.splitext(name)

            try:
                num_processed += 1
                print('Converting {} ({}/{})'.format(name, num_processed, num_files))
                convert(root, filename)
            except Exception as exception:
                print('Skipping {}. Exception: {}'.format(name, exception))
                num_skipped += 1

    print('Number of processed files:', num_processed)
    print('Number of skipped files:', num_skipped)


