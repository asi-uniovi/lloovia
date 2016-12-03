# vim: syntax=python tabstop=4 expandtab
# coding: utf-8

CASES = ['decreasing', 'everything', 'increasing', 'once',
         'periodic', 'static', 'unpredictable']

synthetic_LTWP = expand("data/processed/{case}_LTWP.csv", case=CASES) 
synthetic_STWP = expand("data/processed/{case}_STWP.csv", case=CASES) 
raw_LTWP = expand("data/raw/{case}_LTWP.csv", case = CASES)
raw_STWP = expand("data/raw/{case}_STWP.csv", case = CASES)
wikipedia_traces = []


rule all:
     input: synthetic_LTWP, synthetic_STWP

# Following rule copies pre-generated synthetic load to analysis folder
# Comment this rule and uncomment the next one if you want to generate and use 
# use a different set of syntethic loads
rule use_pregenerated_synthetic_workload:
    input: "data/raw/{case}_{prediction}.csv"
    output: "data/processed/{case}_{prediction}.csv"
    message: "Copying pre-generated synthetic {wildcards.prediction} for case {wildcards.case}"
    shell: "cp {input} {output}"

# Following rule re-generates a new set of synthetic workloads
# Uncomment it and comment previous rule to use a new set of synthetic workloads
 
# rule generate_synthetic_workload:
#     output: "data/processed/{case}_{prediction}.csv"
#     message: "Generating new sythetic {wildcards.prediction} for case {wildcards.case}"
#     run:
#         import src.data.make_dataset as m
#         m.generate_case(wildcards.case, [10000, 50000, 100000, 1000000, 300000],
#                         extension="_" + wildcards.prediction, folder="data/processed")


wikipedia_years = ["%d"%y for y in [2010, 2011, 2012, 2013, 2014, 2015]]
wikipedia_months = ["%02d"%(m+1) for m in range(12)]
wikipedia_traces = expand("/tmp/logs-{year}/{year}-{month}.tbz", year=wikipedia_years, month=wikipedia_months)

rule download_wikipedia_data:
    input: wikipedia_traces

rule download_wikipedia_month:
    output: "/tmp/logs-{year}/{year}-{month}.tbz"
    message: "Downloading wikipedia traces for {wildcards.year}-{wildcards.month}"
    run:
        year, month = int(wildcards.year), int(wildcards.month)
        import src.data.wikipedia_download as wd
        w = wd.WikipediaLogDownloader("/tmp/logs-{}".format(year))
        w.download_month(year, month)
