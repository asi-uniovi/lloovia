import os
import re
import pandas as pd
import logging
import json


def json2pd(json_results):
    """Receives a string in which each line is a valid json expression,
    parses each json to create a dictionary and creates a pandas DataFrame
    in which each row is that dictionary"""

    data = []
    for line in json_results.split("\n"):
        if line:
            data.append(json.loads(line))

    df = pd.DataFrame(data)
    # process some of the fields
    df.timestamp = pd.to_datetime(df.timestamp, unit="s")
    # drop rows whose "metric" is "Timestamp"
    df = df[["Timestamp" not in x for x in df.metric]]
    # Set a multiindex
    df = df.set_index(["test", "metric", "timestamp"])
    # Keep only some columns
    df = df[["labels", "value", "unit", "run_uri"]]
    return df


def json2pd_oldisim(json_results):
    """Receives a string in which each line is a JSON, resulting from
    oldisim benchmark. Processes each json line, extracts the relevant
    information and generates a pandas dataframe with the info"""

    def labels_to_dict(l):
        "Helper function to parse the 'labels' field"
        d = {}
        for data in l.split(","):
            k, v = data.strip("|").split(":")
            d[k] = v
        return d

    data = []
    for line in json_results.split("\n"):
        if line:     # Each line contains a different metric
            aux = json.loads(line)
            if "QPS" not in aux["metric"]:  # We are interested in QPS only
                continue
            # The field "labels" contains info about the CPU, cores...
            lab = labels_to_dict(aux["labels"])
            # Extract only some keys
            relevant_keys = ["machine_type", "zone", "cpu_info", "num_cpus"]
            dic = {key: lab[key] for key in relevant_keys if key in lab}
            dic["QPS"] = aux["value"]
            data.append(dic)

    # Return the list as dataframe
    return pd.DataFrame(data)


def process_oldisim_results(folder):
    """Reads all json files related with oldisim benchmark present in the
    given folder and creates a pandas DataFrame with the relevant information
    which includes the machine_type, the kind of CPU, the number of cores
    and the QPS measured by oldisim. Each machine_type can appear several times
    if the benchmark was run more than once."""
    log = logging.getLogger(__name__)
    instance_types = []
    jsons = []
    for filename in os.listdir(folder):
        if "oldisim" in filename:
            log.info("Processing %s", os.path.join(folder, filename))
            # Use the filename as the name of the VM type
            m = re.search(r"-(.*)\.json", filename)
            instance_types.append(m.group(1))
            jsons.append(open(os.path.join(folder, filename)).read())

    jsons = "\n".join(jsons)
    return json2pd_oldisim(jsons)


def get_qps_as_dict(folder):
    """Condenses the info retrieved by process_oldisim_results into
    a single number of QPS per machine_type and returns a dictionary
    whose keys are the machine_types and
    the values are the QPS"""
    df = process_oldisim_results(folder)
    return df.groupby("machine_type").mean().QPS.to_dict()


def generate_qps_json(input_folder, output_file):
    """Writes as json the result of get_qps_as_dict()"""
    dic = get_qps_as_dict(input_folder)
    with open(output_file, "w") as f:
        f.write(json.dumps(dic))


def generate_rph_csv(providers, output_file):
    """Reads qps for several providers and writes the result in
    a single csv file, to be used later by the analysis.

    providers is a list of dictionaries containing the fields
       'name' (eg. 'amazon' or 'azure'), 'filename' which is the json
       containing the QPS resulting from running oldisim benchmark
       in that provider, and 'vm_types' which is a list of machine types
       to be extracted for that provider.

    output_file is the name of the resulting csv"""

    result = []
    for provider in providers:
        with open(provider["filename"]) as f:
            data = json.load(f)
        for vm_type, qps in data.items():
            if vm_type in provider["vm_types"]:
                result.append(dict(
                    provider=provider["name"],
                    instance_names=vm_type,
                    instance_performances=qps * 60 * 60
                    ))
    data = (pd.DataFrame(result)[["instance_names",
                                  "instance_performances",
                                  "provider"]]
              .sort_values(by="instance_names")
            )
    data.to_csv(output_file, index=False)


if __name__ == "__main__":
    # Only for testing purposes. This script is not meant to be
    # run from command line, but from Snakefile
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    generate_qps_json("../../data/raw/oldisim/azure", "/tmp/qps_azure.json")
    generate_qps_json("../../data/raw/oldisim/amazon", "/tmp/qps_amazon.json")
    providers = [
            dict(name="Azure",
                 filename="/tmp/qps_azure.json",
                 vm_types=["A5", "A6", "Basic_A0", "Basic_A1", "Basic_A2",
                           "Basic_A3", "ExtraSmall", "Large", "Medium",
                           "Small", "Standard_D11_v2", "Standard_D12_v2",
                           "Standard_D1_v2", "Standard_D2_v2", "Standard_D3_v2"
                           ]
                 ),
            dict(name="Amazon",
                 filename="/tmp/qps_amazon.json",
                 vm_types=["c4.2xlarge", "c4.xlarge", "m4.2xlarge", "m4.large",
                           "m4.xlarge"]
                 ),
            ]
    df = generate_rph_csv(providers, "/tmp/kk.csv")
