import requests
from io import StringIO
import pandas as pd
import collections

from lloovia import LimitingSet, InstanceClass

# Amazon

def save_amazon_data(path = ""):
    '''Gets the data about amazon EC2 and saves it as a CSV file
    in the provided path (a string)
    '''
    # Get all of Amazon services
    services = requests.get("https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/index.json")

    # Retrieve the URL for EC2
    ec2_url = services.json()["offers"]["AmazonEC2"]["currentVersionUrl"]

    # Get, from the previous URL, all information about EC2. We use the CSV
    # version because it is easier to parse than the json version
    instances_info = requests.get("https://pricing.us-east-1.amazonaws.com" + ec2_url[:-4] + "csv")

    # Convert the data to Pandas. use StringIO to convert the data from
    # string to something read_csv can parse
    csv = StringIO(instances_info.content.decode("utf-8"))
    all_amazon_data = pd.read_csv(csv,
                                  header=5) # Skip the first 5 lines: they are not data

    # Save only some columns and rename them
    interesting_columns = ["Instance Type", "Location", "PricePerUnit", 
                "vCPU", "Memory", "Storage",  "Operating System",
                "Unit", "LeaseContractLength", "PurchaseOption", "Tenancy", "PriceDescription", ]
    amazon_data = all_amazon_data[interesting_columns].copy()
    amazon_data.columns = ["Type", "Region", "Price", 
                "Cores", "Mem", "Disk",  "OS", 
                 "Unit", "Rsv", "Popt", "Tenancy", "Pdesc"]

    # Clean up data changing some NaN with more significant values
    amazon_data.Rsv = amazon_data.Rsv.fillna("No")
    amazon_data.Popt = amazon_data.Popt.fillna("N/A")

    # Save to CSV
    if path == "":
        filepath = "amazon_data.csv";
    else:
        filepath = path + "/amazon_data.csv";
    amazon_data.to_csv(filepath)

def read_amazon_data(path):
    '''Returns the data read from a csv file given as a path (string).
    NaN values in Rsv are changed to "No" and "N/A" in Popt to "N/A"
    '''
    amazon_data = pd.read_csv(path)
    amazon_data.Rsv = amazon_data.Rsv.fillna("No")
    amazon_data.Popt = amazon_data.Popt.fillna("N/A")
    return amazon_data

def get_amazon_prices(amazon_data, instance, os):
    """Receives a pandas dataframe with all the required data (columns
    "Type", "Region", "Price", "Cores", "Mem", "Disk",  "OS", "Unit",
    "Rsv", "Popt", "Tenancy", "Pdesc")  and the name of a instance type and OS
    and returns pricing information per hour as a pandas dataframe with multi-indexed
    rows by region, tenancy, leasing, and payment opts"""
    filtered = amazon_data[(amazon_data.Type == instance) & (amazon_data.OS.str.contains(os))]
    
    # Removed instances with type "Dedicated Host" because we don't use them
    filtered = filtered[filtered.Tenancy != "Host"]
    
    index_keys = ["Region", "Tenancy", "Rsv", "Popt", "Unit"]
    
    # Group using the previous keys and keep only data about price
    grouped = filtered.sort_values(by=index_keys).set_index(index_keys)[["Price"]]
    
    # Change into columns the rows with have the price per hour and
    # per unit (last element of the index, and thus the use of -1).
    # In addition, fill NaN with zeros
    grouped = grouped[["Price"]].unstack(-1).fillna(0).reset_index()
    
    # Compute the price per hour for the instances reserved for one year.
    # In the column Price/Hrs we have the upfront price divided into the
    # hours of one year plus the price per hour after the upfront, so
    # we get the total price per hour
    yr1 = grouped[grouped.Rsv=="1yr"].copy()
    yr1["Price/h"] = yr1[("Price", "Hrs")] + yr1[("Price", "Quantity")] /365 / 24

    # Idem for instances reserved for three years
    yr3 = grouped[grouped.Rsv=="3yr"].copy()
    yr3["Price/h"] = yr3[("Price", "Hrs")] + yr3[("Price", "Quantity")] /365 /24 /3

    # For on-demand instances, the price per hour is the one we already had in Price
    on_demand = grouped[grouped.Rsv=="No"].copy()
    on_demand["Price/h"] = on_demand[("Price", "Hrs")]
    
    # Join together the data from the previous computations
    prices = yr1.merge(yr3, how="outer").merge(on_demand, how="outer")
    
    # Reorder and reindex the table using the same keys as before,
    # except the last one, that has been removed in the process. Now
    # all instance types have the price per hour
    index_keys = index_keys[:-1]
    prices = prices.sort_values(by=index_keys).set_index(index_keys)

    return prices[["Price/h"]]

def get_simplified_amazon_prices(amazon_data, instance, os):
    '''Receives a pandas dataframe with all the required data (columns
    "Type", "Region", "Price", "Cores", "Mem", "Disk",  "OS", "Unit",
    "Rsv", "Popt", "Tenancy", "Pdesc")  and the name of a instance type and OS
    and returns pricing information per hour as a dictionary with two
    rows: "on_demand" and "reserved"
    '''
    prices = get_amazon_prices(amazon_data, instance, os)
    on_demand = prices.xs(("Shared", "No", "N/A"), level=(1,2,3))["Price/h"]
    reserved = prices.xs(("Shared", "1yr", "All Upfront"), level=(1,2,3))["Price/h"]
    return dict(on_demand = on_demand.to_dict(), reserved = reserved.to_dict())

def generate_amazon_region_instances(amazon_data, region_name, max_inst_per_type,
                                     max_inst_per_group, availability_zones, perf):
    '''Arguments:
    - amazon_data: dataframe with information about price. 
    - region_name: name of the region where we are generating the instances
    - max_inst_per_type: maximum number of on-demand instances per type
    - max_inst_per_group: maximum number of on-demand instances per region and
                and of reserved instances per availability zone
    - availability_zones: number of availability zones inside the region
    - perf: a dictionary where the key is the name of an instance type and the value,
            its performance

    Returns:
    - A list of InstaceClasses with the given region, limits, availability zones and
      performance
    '''
    region_data = collections.OrderedDict()
    
    for instance_name in sorted(perf):
        r = get_simplified_amazon_prices(amazon_data, instance_name, "Linux")
        region_data[instance_name] = dict(on_demand = r["on_demand"].get(region_name, None),
                                          reserved = r["reserved"].get(region_name, None),
                                          perf = perf[instance_name],
                                          max_vms = max_inst_per_type)
    
    ins = []
    # The limiting set "region" sets a limit for on-demand instances
    region = LimitingSet(region_name, max_vms = max_inst_per_group)
    for i,dat in region_data.items():
        if dat["on_demand"] == None:
            continue # Not all regions have all instance types
        ins.append(InstanceClass(i, region, performance=dat["perf"],
                       price=dat["on_demand"], max_vms=dat["max_vms"], 
                       reserved = False))
        
    # The limiting set "availability zone" sets a limit for reserved instances
    availability_zones = [ 
        LimitingSet(region_name + "_AZ%d" % z, max_vms = max_inst_per_group)
        for z in range(1,availability_zones + 1) 
    ]
    for i, dat in region_data.items():
        for zone in availability_zones:
            if dat["reserved"] == None:
                continue # Not all regions have all instance types
            ins.append(InstanceClass(i, zone, performance=dat["perf"],
                       price=dat["reserved"], 
                       max_vms=0, # Reserved instances don't have limit per type
                       reserved = True))
    return ins

# Azure

def read_azure_data(path):
    '''Returns the data read from a csv file given as a path (string).
    '''
    return pd.read_csv(path)

def generate_azure_instances(azure_data, region_name, max_cores, perf):
    '''Arguments:
    - azure_data: dataframe with information about price and VM characteristics. 
    - region_name: name of the region where we are generating the instances.
      Can be 'eu' or 'us-east-2'.
    - max_cores: maximum number of cores per region
    - perf: a dictionary where the key is the name of an instance type and the value,
            its performance

    Returns:
    - A list of InstaceClasses with the given region, limits and performance'''
    ins = []
    ls = LimitingSet(region_name, max_cores = max_cores)
    for i,dat in azure_data.iterrows():
        ins.append(InstanceClass(dat["Type"], ls,
                               performance=perf[dat["Type"]],
                               price=dat["price-"+region_name],
                               reserved = False,
                               provides={"cpus": dat["Cores"]}))

    return ins