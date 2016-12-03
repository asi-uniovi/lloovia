# -*- coding: utf-8 -*-
import os
import logging
import pandas as pd
import numpy as np
from collections import namedtuple
from functools import partial


def generate_workload(slots, base_level,
                      noise_instant_range=0, noise_cummulative_range=0,
                      peak_prob=0, peak_size=0, peak_length=0,
                      long_cycles=0, long_cycle_size=0,
                      short_cycles=0, short_cycle_size=0,
                      shorter_cycles=0, shorter_cycle_size=0,
                      tendency=0, tendency_exp=1):
    '''Arguments:
    - slots: number of slots to generate. For instance, for a trace of one
    year with data each hour, it would be 265*24 = 8760
    - base_level: base level (number of requests)
    - noise_instant: a value gaussian noise with variance 1/3rd of this
        parameter is added to each value
    - noise_cummulative: a value uniformingly distributed between 0 and this
        parameter will be added or substracted to the workload and accumulated
        to the next slot
    - peak_prob: probability of having a peak in a slot
    - peak_size: size of the peak
    - peak_length: length (in slots) of the peak
    - long_cycles: number of cycles for long periods, for instance, 4 for each
        season in a year
    - long_cycle_size: amplitude of the long cycle
    - short_cycles: number of cycles for short periods, for instance, 52 for
        each week in a year
    - short_cycle_size: amplitude of the short cycle
    - short_cycles: number of cycles for even shorter periods, for instance,
        265 for a cycle each day
    - shorter_cycle_size: amplitude of the shorter cycle
    - tendency: value to add cummulatively to the base_level in each slot to
        represent a tendency. If it is negative, it will represent a negative
        tendency
    - tendency_exp: exponent of the tendency. If it is 0, no tendency; 1,
        linear; 2, squared; 3, cubic; etc.

    Returns:
    - A list of values with load levels for each slot
    '''
    result = []
    long_cycle_load = np.sin(
        np.arange(slots)/slots*2*np.pi*long_cycles) * long_cycle_size
    short_cycle_load = np.sin(
        np.arange(slots)/slots*2*np.pi*short_cycles) * short_cycle_size
    shorter_cycle_load = np.sin(
        np.arange(slots)/slots*2*np.pi*shorter_cycles) * shorter_cycle_size
    current_peak_length = 0
    current_noise_c = 0
    for i in range(0, slots):
        l = base_level
        # Tendency
        l += ((i+1)**tendency_exp)*tendency

        # Cycles
        l += l * (
            long_cycle_load[i] + short_cycle_load[i] + shorter_cycle_load[i])

        # Peak
        if current_peak_length == 0 and np.random.random() <= peak_prob:
            current_peak_length = peak_length

        if current_peak_length != 0 and current_peak_length > 0:
            l += l * peak_size
            current_peak_length -= 1

        # Instant noise
        if noise_instant_range > 0:
            noise_i = np.random.normal(0, noise_instant_range / 3)
            l += noise_i

        # Cummulative noise
        if noise_cummulative_range > 0:
            noise_c = np.random.normal(0, noise_cummulative_range / 3)
            current_noise_c += noise_c
            l += current_noise_c

        # The load cannot be less than 0
        if (l < 0):
            l = 0

        result.append(int(l))
    return result


def generate_case(case, levels, slots=24*365,
                  extension="_base-level", folder="/tmp",
                  logger=logging.getLogger(__name__)):
    """This function generates the csv for one scenario, using hardcoded
    values for the parameters of the generator, to reproduce a set
    of scenarios similar to the ones in the paper.

     case = The case to generate, one of 'Static', 'Periodic', 'Once'
            'Unpredictable', 'Decreasing', 'Increasing' or 'Everything'
     levels = array of base-levels to generate, for example [50000, 100000,
            1000000, 3000000] to use the same ones than in the paper
     slots = Number of timeslots in each synthetic trace,
     extension = suffix to add to the filenames (to separate loads used
             as predictions from the ones used as realization)
     folder = folder where the traces will be written
    """

    def generate_single_level(case, level, slots=slots):
        Case = namedtuple("Case", ["name", "filename", "function"])
        base_level = level
        cases = [
            Case("static",  "static",
                 partial(generate_workload,
                         noise_instant_range=base_level * 0.2)
                 ),
            Case("periodic", "periodic",
                 partial(generate_workload,
                         noise_instant_range=base_level * 0.1,
                         long_cycles=2, long_cycle_size=0.1,
                         short_cycles=52, short_cycle_size=0.1,
                         shorter_cycles=365, shorter_cycle_size=0.1)
                 ),
            Case("once", "once",
                 partial(generate_workload,
                         noise_instant_range=base_level * 0.1,
                         peak_prob=2.0/(24*365.0),
                         peak_size=1,
                         peak_length=slots*0.01)
                 ),
            Case("unpredictable", "unpredictable",
                 partial(generate_workload,
                         noise_cummulative_range=base_level * 0.02)
                 ),
            Case("decreasing", "decreasing",
                 partial(generate_workload,
                         noise_instant_range=base_level * 0.1,
                         tendency=-base_level * 0.00005)
                 ),
            Case("increasing", "increasing",
                 partial(generate_workload,
                         noise_instant_range=base_level * 0.3,
                         tendency=base_level * 0.000001, tendency_exp=1.5)
                 ),
            Case("everything", "everything",
                 partial(generate_workload,
                         noise_instant_range=base_level * 0.1,
                         noise_cummulative_range=base_level * 0.005,
                         peak_prob=2/(24*365.0), peak_size=0.6,
                         peak_length=10,
                         long_cycles=2, long_cycle_size=0.1,
                         short_cycles=52, short_cycle_size=0.1,
                         shorter_cycles=365, shorter_cycle_size=0.1,
                         tendency=base_level * 0.00007, tendency_exp=1)
                 )
        ]
        dict_cases = {}
        for c in cases:
            dict_cases[c.name] = c
        if case not in dict_cases.keys():
            raise ValueError("{} case is unknown, use one of {}".
                             format(case, [c.name for c in cases]))
        c = dict_cases[case]
        return c.filename, c.function(slots=slots, base_level=level)

    r = []
    for l in levels:
        # generate all levels in a list
        f_name, load = generate_single_level(case, l)
        r.append(load)
    filename = f_name + extension
    # Use pandas to convert the list to csv
    df = pd.DataFrame(r).T
    df.columns = levels
    path = "{}/{}.csv".format(folder, filename)
    df.to_csv(path)
    logger.info("Synthetic workload created and saved in %s", path)
    return df


def create_synthetic_workload(output_path, extension="", regenerate=False,
                              logger=logging.getLogger(__name__)):
    """ Creates a set of csv containing synthetic workload for
    experiments on the influence of the binning in the result.
    For reproducibility this function should not be called, and use
    instead the .csv provided in data/raw
    If you want to force the re-generation of the synthetic traces
    pass regenerate=True to this function
    """
    if not regenerate:
        logger.warn("Skipping the generation of synthetic traces, "
                    "since they are already pre-generated")
        logger.warn("Pass regenerate=True to create new synthetic traces")
        return
    levels = [50000, 100000, 1000000, 3000000]
    cases = ['decreasing', 'everything', 'increasing', 'once',
             'periodic', 'static', 'unpredictable']
    for case in cases:
        generate_case(case, levels, extension=extension, folder=output_path)


def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Creating synthetic workload prediction (LTWP)')
    create_synthetic_workload(output_filepath, extension='_LTWP',
                              regenerate=True)
    logger.info('Creating synthetic workload realization (STWP)')
    create_synthetic_workload(output_filepath, extension='_STWP',
                              regenerate=True)


def download_wikipedia_data(years):
    if type(years) != list:
        years = [years]
    logger = logging.getLogger(__name__)
    logger.info("Downloading access data from wikipedia, for years %s", years)
    from wikipedia_download import WikipediaLogDownloader
    for year in years:
        w = WikipediaLogDownloader("logs-%s" % year)
        for month in [1,2]:
            logger.info("Downloading wikipedia traces %d-%02d", year, month)
            w.download_month(year, month)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    # main("", os.path.join(project_dir, "data", "test"))
    download_wikipedia_data(2012)
