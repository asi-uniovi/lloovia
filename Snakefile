# vim: syntax=python tabstop=4 expandtab
# coding: utf-8

# The following setting forces the use of the data in 'data/paper', so all results
# should be the same than the ones shown in the paper.

USE_PAPER_DATASET = True
 
# If you change this to False, then several datasets will be regenerated:
# 
# * Wikipedia traces are downloaded and processed again
# * Amazon and Azure prices and VM types are downloaded from their websites and processed again
# * Synthetic workloads are randomly regenerated
#
# Note that as a consequence the analysis results would vary. It is even possible 
# that the analysis cannot be performed because VM types in Amazon or Azure were 
# renamed/removed from their clouds. In this case the code which generates sets
# of instances for the analysis should be changed


include: "Snakefile.definitions"

rule create_input_data:
     input: synthetic_LTWP, synthetic_STWP, synthetic_example,
            processed_providers, processed_benchmarks, processed_wikipedia

include: "Snakefile.rules"
