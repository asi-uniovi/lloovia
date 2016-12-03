import lxml.html
import datetime
import dateutil.relativedelta
import logging
import requests
import os
import tarfile
from io import BytesIO


class WikipediaLogDownloader:

    def __init__(self, folder=""):
        self.folder = folder
        if not os.path.exists(folder):
                os.makedirs(folder)
        self.log = logging.getLogger(__name__)
        logging.getLogger("requests").setLevel(logging.WARNING)

    def download_file(self, file_url, file_date, hour, tgz=None):
        r = requests.get(file_url)
        if r.status_code != 200:
            self.log.error("Can't get file %s", file_url)
            return

        if self.folder != "":
            folder = self.folder + "/"
        else:
            folder = self.folder


        if tgz is None:
            try:
                with open(folder + str(file_date) +
                          "_" + hour + ".log", "w") as f:
                    f.write(r.text)
            except:
                self.log.error("Error for file " + file_url)
                with open(folder + "error_log.txt", "a") as ef:
                    ef.write("Error for file " + file_url)
                return
        else:
            try:
                info = tarfile.TarInfo(str(file_date) + "_" + hour + ".log")
                info.size = len(r.content)
                tgz.addfile(info, BytesIO(bytes(r.content)))
            except:
                self.log.error("Error for file " + file_url)
                with open(folder + "error_log.txt", "a") as ef:
                    ef.write("Error for file " + file_url)
                return

    def obsolete_download_month(self, current_date, start_date, end_date):
        '''Downloads all the files for the month of current_date as long as they are
        before end_date. Writes all of them in a single .tgz file.
        All dates are datetime.date objects.
        '''
        base_url = 'http://dumps.wikimedia.org/other/pagecounts-raw'

        year = str(current_date.year)
        month = "{:02d}".format(current_date.month)

        index_url = base_url + '/' + year + '/' + year + "-" + month

        dom = lxml.html.fromstring(requests.get(index_url).text)

        tgz = tarfile.open(self.folder + "/" +
                           datetime.date(int(year), int(month)) +
                           ".tgz", mode="w:bz2")
        for link in dom.xpath('//a/@href'):
            # select the url in href for all a tags
            if (link.startswith('projectcounts')):
                hour = link[-6:-4]
                day = link[-9:-7]
                file_url = index_url + '/' + link
                file_date = datetime.date(int(year), int(month), int(day))
                if start_date <= file_date < end_date:
                    self.download_file(file_url, file_date, hour, tgz=tgz)
        tgz.close()

    def download_month(self, year, month):
        '''Downloads all the files for the month of current_date as long as they are
        before end_date. Writes all of them in a single .tgz file.
        All dates are datetime.date objects.
        '''
        base_url = 'http://dumps.wikimedia.org/other/pagecounts-raw'

        start_date = datetime.date(year, month, 1)
        end_date = start_date + dateutil.relativedelta.relativedelta(months=+1)
        year = str(year)
        month = "{:02d}".format(month)

        index_url = base_url + '/' + year + '/' + year + "-" + month

        dom = lxml.html.fromstring(requests.get(index_url).text)

        tgz = tarfile.open("{}/{}-{}.tbz".format(self.folder, year, month),
                           mode="w:bz2")
        for link in dom.xpath('//a/@href'):
            # select the url in href for all a tags
            if (link.startswith('projectcounts')):
                hour = link[-6:-4]
                day = link[-9:-7]
                file_url = index_url + '/' + link
                file_date = datetime.date(int(year), int(month), int(day))
                if start_date <= file_date < end_date:
                    self.download_file(file_url, file_date, hour, tgz=tgz)
        tgz.close()

    def download(self, start_year, start_month, start_day,
                 number_of_days, folder=""):
        start_date = datetime.date(start_year, start_month, start_day)
        end_date = start_date + datetime.timedelta(days=number_of_days)

        self.folder = folder

        # Iterate over the months, as there is one index page per month
        current_date = start_date
        while current_date < end_date:
            self.download_month(current_date, start_date, end_date)
            current_date = (current_date +
                            dateutil.relativedelta.relativedelta(months=+1))
