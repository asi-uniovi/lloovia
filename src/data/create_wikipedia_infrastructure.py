from cloud_providers import read_amazon_data, generate_amazon_region_instances
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

