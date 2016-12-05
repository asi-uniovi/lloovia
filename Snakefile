# vim: syntax=python tabstop=4 expandtab
# coding: utf-8

import logging
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)

CASES = ['decreasing', 'everything', 'increasing', 'once',
         'periodic', 'static', 'unpredictable']

synthetic_LTWP = expand("data/processed/{case}_LTWP.csv", case=CASES) 
synthetic_STWP = expand("data/processed/{case}_STWP.csv", case=CASES) 
paper_LTWP = expand("data/paper/{case}_LTWP.csv", case = CASES)
paper_STWP = expand("data/paper/{case}_STWP.csv", case = CASES)


rule all:
     input: synthetic_LTWP, synthetic_STWP, "data/processed/en_wikipedia.csv"

# Following rule copies pre-generated synthetic load to analysis folder
# Comment this rule and uncomment the next one if you want to generate and use 
# use a different set of syntethic loads
rule use_pregenerated_synthetic_workload:
    input: "data/paper/{case}_{prediction}.csv"
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
processed_wikipedia_traces = expand("data/processed/en_wikipedia_{year}.csv", year=wikipedia_years)

rule download_wikipedia_traces:
    input: wikipedia_traces


def tbz_in_a_year(w):
    year=w.year
    return ["/tmp/logs-{}/{}-{:02}.tbz".format(year, year, month+1) for month in range(12)]

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

# rule process_wikipedia_traces:
#     input: wikipedia_traces
#     output: processed_wikipedia_traces


rule processed_wikipedia_traces:
    input: processed_wikipedia_traces
    output: "data/processed/en_wikipedia.csv"
    shell: "cat {input} > {output}"

rule download_wikipedia_month:
    output: "/tmp/logs-{year}/{year}-{month}.tbz"
    message: "Downloading wikipedia traces for {wildcards.year}-{wildcards.month}"
    run:
        year, month = int(wildcards.year), int(wildcards.month)
        import src.data.wikipedia_download as wd
        w = wd.WikipediaLogDownloader("/tmp/logs-{}".format(year))
        w.download_month(year, month)
