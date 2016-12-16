from cloud_providers import (
        read_amazon_data, generate_amazon_region_instances,
        read_azure_data, generate_azure_instances
        )
import pandas as pd
import pickle
from create_infrastructure_for_experiments import load_infrastructure


def create_instances_binning_analysis(benchmark_data, provider_data, save_to):
    instances = load_infrastructure(
                        benchmark_data, provider_data,
                        regions=[("US East (N. Virginia)", 5),
                                 ("US West (N. California)", 3),
                                 ("US West (Oregon)", 3),
                                 ("EU (Ireland)", 3),
                                 ("EU (Frankfurt)", 2),
                                 ("Asia Pacific (Tokyo)", 3),
                                 ("Asia Pacific (Seoul)", 2),
                                 ("Asia Pacific (Singapore)", 2),
                                 ("Asia Pacific (Sydney)", 3),
                                 ("South America (Sao Paulo)", 3)
                                 ],
                        limits=(50, 100))
    with open(save_to, "wb") as f:
        pickle.dump(instances, f)


def create_instances_multicloud_analysis(benchmark_data, provider_data,
                                         provider, output_file):
    data = pd.read_csv(benchmark_data).set_index("instance_names")
    perf = data.query('provider=="%s"' % provider.capitalize())\
               .instance_performances.to_dict()
    if provider == "amazon":
        amazon_data = read_amazon_data(provider_data)
        instances = generate_amazon_region_instances(
                            amazon_data, "US East (N. Virginia)",
                            max_inst_per_type=20,
                            max_inst_per_group=20,
                            availability_zones=3,
                            perf=perf)
    elif provider == "azure":
        azure_data = read_azure_data(provider_data)
        instances = generate_azure_instances(
                             azure_data, "us-east-2", 20, perf)
    else:
        raise Exception("Unknown provider '%s'" % provider)
    with open(output_file, "wb") as f:
        pickle.dump(instances, f)
