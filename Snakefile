# vim: syntax=python tabstop=4 expandtab
# coding: utf-8

include: "Snakefile.definitions"

rule create_input_data:
     input: synthetic_LTWP, synthetic_STWP, processed_wikipedia,
            processed_providers, processed_benchmarks

include: "Snakefile.rules"

ruleorder: copy_pregenerated_synthetic_workload > generate_synthetic_workload
ruleorder: copy_already_processed_wikipedia_traces > join_wikipedia_years
ruleorder: copy_preprocessed_provider_data > process_amazon_data
ruleorder: copy_benchmark_results > summarize_oldisim 


