import os
import os.path
import pandas as pd
import logging
import tarfile


class WikipediaAnalyzer:

    def __init__(self):
        self.all_views = []
        self.df = None
        self.log = logging.getLogger(__name__)

    def add_data_from_file(self, filename, f):
        '''The filename has to have the form YYYY-MM-DD_HH.any_extension
        '''
        basename = os.path.splitext(os.path.basename(filename))[0]
        date, hour = basename.split('_')
        time = "{} {}:00:00".format(date, hour)

        num_errors = 0
        for line in f:
            fields = line.split()
            if len(fields) == 0:
                continue
            if len(fields) < 3:
                self.log.warning("Missing fields in file: %s", filename)
                num_errors = num_errors + 1
                if num_errors > 3:
                    self.log.error("Too many errors for this file."
                                   " Abandoning %s", filename)
                    return
                continue
            project = str(fields[0], "utf-8", errors="replace")
            views = fields[2]
            try:
                self.all_views.append(dict(project=project, time=time,
                                      views=int(views)))
            except:
                self.log.warning("Wrong value %s, for project %s",
                                 views, project)
                num_errors = num_errors + 1
                if num_errors > 3:
                    self.log.error("Too many errors for this file."
                                   " Abandoning %s", filename)
                    return

    def add_data_from_tbz(self, filename):
        '''Opens a .tbz file (compressed tar) and for each file inside
        calls add_data_from_file() for that file'''
        with tarfile.open(filename, "r:bz2") as t:
            self.log.info("Processing %s", filename)
            for member in t.getnames():
                self.add_data_from_file(member, t.extractfile(member))

    def analyze(self, folder):
        '''Analyzes all the files in a folder assuming they are Wikipedia logs with the
        name YYYY-MM.tbz'''
        for root, _, files in os.walk(folder):
            for f in files:
                if f.endswith("tbz"):
                    self.add_data_from_tbz(folder + "/" + f)

    def getdf(self):
        # Converts all accumulated data into a proper pandas dataframe
        df = (pd.DataFrame(data=self.all_views)
              .assign(time=lambda x: pd.to_datetime(x.time))
              .set_index("time")
              .sort_index())
        return df


def concatenate_all_csv_in_single_dataframe(list_of_csvs):
    # Not used
    aux = []
    for csv in list_of_csvs:
        df = (pd.read_csv(csv, sep=";", header=None,
                          names=["time", "rph"])
              .assign(time=lambda x: pd.to_datetime(x.time))
              .set_index("time").reindex()
              )
        aux.append(df)
    return pd.concat(aux).set_index("time")

if __name__ == "__main__":
    # Only for testing purposes. This script is not meant to be
    # run from command line, but from Snakefile
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    for year in [2012, 2013, 2014, 2015]:
        w = WikipediaAnalyzer()
        w.analyze("/tmp/logs-{}".format(year))
        (w.getdf()
         .query("project == 'en'")
         .views
         .to_csv("/tmp/{}.csv".format(year), sep=";")
         )
