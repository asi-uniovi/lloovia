"""This module performs web scraping to obtain prices and characterestics
of Azure virtual machines"""

import logging
import json
from lxml import etree
from io import StringIO


class AzurePrices:
    def __init__(self, filename):
        self.filename = filename
        self.log = logging.getLogger(__name__)
        self.log.info("Filename=%r", filename)

    def parse_vm(self, node):
        """Receives a xtree node from which all the data about a VM type can
        be extracted. This code is very fragile because it depends on the
        structure of the web page, which can be changed by Microsoft"""

        dic = {}
        # Get data about this VM type, which is two cells above
        cells = node.xpath("td")

        # First four cells in the row contain hardware features
        cols = ["Instance", "Cores", "RAM", "DISK"]
        for i, c in enumerate(cells[:-1]):
            dic[cols[i]] = " ".join([data.strip()
                                     for data in c.xpath(".//text()")
                                     ])
        # VM tier is more difficult. It is extracted from the last h3 title
        dic["Tier"] = node.xpath("preceding::h3/text()")[-1].strip()

        # Last cell is a <span data-amount="xxx">, being xxx a JSON string
        # which contains prices and regions
        data_amount = cells[-1].xpath("./span/@data-amount")[0]
        if not data_amount.startswith("{"):
            return None
        price_data = json.loads(data_amount)
        # Default price for all regions
        if "default" in price_data:
            dic["default"] = price_data["default"]
        # Specific price for some regions
        for region, price in price_data["regional"].items():
            dic[region] = float(price)
        return dic

    def scrape(self):
        with open(self.filename, "r") as f:
            r = f.read()
        parser = etree.HTMLParser()
        self.log.info("Scraping data from Azure web")
        tree = etree.parse(StringIO(r), parser)
        data = []
        tiers = tree.xpath("//*[contains(@class, 'table-width-even')]")
        for tier in tiers:
            vms = tier.xpath(".//tr")
            for vm in vms[1:]:
                data.append(self.parse_vm(vm))
        return json.dumps(data)

    def scrape_and_save(self, filename):
        self.log.info("About to scrap %s", self.filename)
        data = self.scrape()
        self.log.info("About to save %s", filename)
        with open(filename, "w") as f:
            f.write(data)


if __name__ == "__main__":
    # For testing purpose only. This module is not intended
    # to be run from command line, but from Snakefile
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    df = AzurePrices("/tmp/azure-web.html").scrape_azure()
