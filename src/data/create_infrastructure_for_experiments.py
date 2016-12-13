from cloud_providers import read_amazon_data, generate_amazon_region_instances
import pandas as pd
import pickle


def load_infrastructure(benchmark, provider, regions, limits):
    """Reads data from different sources and combines it to create a cloud
    infrastructure composed by Amazon regions, with different availability
    zones in each one and all types of VMs available in each region.

    Performance data is from Wikibench test. VM characteristics and prices
    are from Amazon.

    Inputs:
       regions: is a list of tuples, being the first element the name
          of the region and the second one the number of availability
          zones in it
       limits: is a tuple, being the first element the limit in the number
           of instances per VM type, and the second one the limit of
           instances per Limiting Set (either region or zone)
    Output:
        Returns a list with instance classes, ready to be used as parameter
        to Llovia constructor."""

    df_perf = pd.read_csv(benchmark).set_index("instance_names")
    perf = df_perf.to_dict()['instance_performances']
    amazon_data = read_amazon_data(provider)
    instances = []
    for region, zones in regions:
        instances.extend(generate_amazon_region_instances(
                                amazon_data, region,
                                max_inst_per_type=limits[0],
                                max_inst_per_group=limits[1],
                                availability_zones=zones, perf=perf))
    return instances


def create_instances(benchmark_data, provider_data, save_to):
    instances = load_infrastructure(
                        benchmark_data, provider_data,
                        regions=[("US West (Oregon)", 3),
                                 ("EU (Ireland)", 3)],
                        limits=(20, 20))
    with open(save_to, "wb") as f:
        pickle.dump(instances, f)

