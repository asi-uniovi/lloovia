# vim: syntax=python tabstop=4 expandtab
# coding: utf-8

CASES = ['decreasing', 'everything', 'increasing', 'once',
         'periodic', 'static', 'unpredictable']

PROVIDERS = ['amazon', 'azure']

BENCHMARKS = ['oldisim', 'wikibench']

synthetic_LTWP = expand("data/processed/traces_synthetic_{case}_LTWP.csv", case=CASES) 
synthetic_STWP = expand("data/processed/traces_synthetic_{case}_STWP.csv", case=CASES) 
paper_LTWP = expand("data/paper/traces_synthetic_{case}_LTWP.csv", case = CASES)
paper_STWP = expand("data/paper/traces_synthetic_{case}_STWP.csv", case = CASES)

processed_wikipedia = "data/processed/traces_en_wikipedia.csv"
paper_wikipedia = "data/paper/traces_en_wikipedia.csv"

processed_providers = expand("data/processed/providers_{provider}_data.csv", provider=PROVIDERS)
paper_providers = expand("data/paper/providers_{provider}_data.csv", provider=PROVIDERS)

processed_benchmarks = expand("data/processed/benchmarks_{benchmark}.csv", benchmark=BENCHMARKS)
paper_benchmarks = expand("data/paper/benchmarks_{benchmark}.csv", benchmark=BENCHMARKS)


#############################################################################
# SYNTHETIC WORKLOADS for the experiments
#############################################################################

rule all:
     input: synthetic_LTWP, synthetic_STWP, processed_wikipedia,
            processed_providers, processed_benchmarks

include: "Snakefile.rules"

ruleorder: copy_pregenerated_synthetic_workload > generate_synthetic_workload
ruleorder: copy_already_processed_wikipedia_traces > join_wikipedia_years
ruleorder: copy_preprocessed_provider_data > process_amazon_data
ruleorder: copy_benchmark_results > summarize_oldisim 


