# vim: syntax=python tabstop=4 expandtab
# coding: utf-8

# import logging
# log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# logging.basicConfig(level=logging.INFO, format=log_fmt)

CASES = ['decreasing', 'everything', 'increasing', 'once',
         'periodic', 'static', 'unpredictable']

synthetic_LTWP = expand("data/processed/{case}_LTWP.csv", case=CASES) 
synthetic_STWP = expand("data/processed/{case}_STWP.csv", case=CASES) 
paper_LTWP = expand("data/paper/{case}_LTWP.csv", case = CASES)
paper_STWP = expand("data/paper/{case}_STWP.csv", case = CASES)


#############################################################################
# SYNTHETIC WORKLOADS for the experiments
#############################################################################

# This ruleorder ensures that the data used is the same than
# the one used for the paper. If you prefer instead to generate
# new synthetic workloads, invert the order of the rule
ruleorder: use_pregenerated_synthetic_workload > generate_synthetic_workload

rule all:
     input: synthetic_LTWP, synthetic_STWP, "data/processed/en_wikipedia.csv",
            "data/processed/azure_data.csv", "data/processed/amazon_data.csv"

rule use_pregenerated_synthetic_workload:
    input: "data/paper/{case}_{prediction,[A-Z]+}.csv"
    output: "data/processed/{case}_{prediction,[A-Z]+}.csv"
    message: "Copying pre-generated synthetic {wildcards.prediction} for case {wildcards.case}"
    shell: "cp {input} {output}"

rule generate_synthetic_workload:
    output: "data/processed/{case,[^\W\d_]}_{prediction,[A-Z]}.csv"
    message: "Generating new sythetic {wildcards.prediction} for case {wildcards.case}"
    run:
        import src.data.make_dataset as m
        m.generate_case(wildcards.case, [10000, 50000, 100000, 1000000, 300000],
                        extension="_" + wildcards.prediction, folder="data/processed")


##############################################################################
# WIKIPEDIA DATA
##############################################################################

# This ruleorder ensures that the data used as wikipedia traces is the
# same than the one used in the paper. If you prefer instead to donwload
# again the data from wikipedia servers and process it, change the order
# of this rule (beware, the download takes a long time and uses 345M)

# The raw wikipedia traces are stored in /tmp, but the processed ones
# are kept in data/processed. Raw traces can be deleted once processed

ruleorder: use_already_processed_wikipedia_traces > process_wikipedia_traces

rule use_already_processed_wikipedia_traces:
    output: "data/processed/en_wikipedia.csv"
    input: "data/paper/en_wikipedia.csv"
    message: "Copying preprocessed wikipedia traces"
    shell: "cp {input} {output}"


# Recompute wikipedia traces
#
wikipedia_years = ["%d"%y for y in range(2010, 2016)]
wikipedia_months = ["%02d"%(m+1) for m in range(12)]
raw_wikipedia_traces = expand("/tmp/logs-{year}/{year}-{month}.tbz", year=wikipedia_years, month=wikipedia_months)
processed_wikipedia_years = expand("data/processed/en_wikipedia_{year}.csv", year=wikipedia_years)
def tbz_in_a_year(w):
    year=w.year
    return ["/tmp/logs-{}/{}-{:02}.tbz".format(year, year, month+1) for month in range(12)]

rule download_wikipedia_traces:
    input: raw_wikipedia_traces

rule download_wikipedia_month:
    output: "/tmp/logs-{year}/{year}-{month}.tbz"
    message: "Downloading wikipedia traces for {wildcards.year}-{wildcards.month}"
    run:
        year, month = int(wildcards.year), int(wildcards.month)
        import src.data.wikipedia_download as wd
        w = wd.WikipediaLogDownloader("/tmp/logs-{}".format(year))
        w.download_month(year, month)

rule process_wikipedia_year:
    input: tbz_in_a_year
    output: "data/processed/en_wikipedia_{year}.csv"
    message: "Processing wikipedia data for year {wildcards.year}"
    run:
        import src.process_data.process_wikipedia_traces as wp
        w = wp.WikipediaAnalyzer()
        w.analyze("/tmp/logs-{}".format(wildcards.year))
        (w.getdf()
         .query("project == 'en'")
         .views
         .to_csv("{}".format(output), sep=";")
         )

rule process_wikipedia_traces:
    input: processed_wikipedia_years
    output: "data/processed/en_wikipedia.csv"
    shell: "cat {input} > {output}"


##############################################################################
# Cloud providers data
##############################################################################

providers = ["amazon", "azure"]

ruleorder: copy_provider_data > process_provider_data > process_amazon_data

rule providers_data:
    input: expand("data/processed/{provider}_data.csv", provider=providers)

rule copy_provider_data:
    input: "data/paper/{provider}_data.csv"
    output: "data/processed/{provider}_data.csv"
    message: "Copying provider data for {wildcards.provider}"
    shell: "cp {input} {output}"

rule process_provider_data:
    input: "data/processed/{provider}_data.csv"

rule process_amazon_data:
    input: "/tmp/raw_amazon_data.csv.gz"
    output: "data/processed/amazon_data.csv"
    run:
        import src.process_data.amazon_data as ad
        ad.simplify_amazon_data(str(input), str(output))

rule download_amazon_data:
    output: "/tmp/raw_amazon_data.csv.gz"
    run:
        import src.data.amazon_donwload as ad
        ad.download_amazon_data(str(output))
