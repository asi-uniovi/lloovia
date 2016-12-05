import logging
import requests
import gzip


def download_amazon_data(filename):
    '''Gets the data about amazon EC2 and saves it as a compressed CSV file
    in the provided filename.
    '''
    log = logging.getLogger(__name__)
    # Get all of Amazon services
    services = requests.get("https://pricing.us-east-1.amazonaws.com/"
                            "offers/v1.0/aws/index.json")

    # Retrieve the URL for EC2
    ec2_url = services.json()["offers"]["AmazonEC2"]["currentVersionUrl"]

    # Get, from the previous URL, all information about EC2. We use the CSV
    # version because it is easier to parse than the json version
    info = requests.get("https://pricing.us-east-1.amazonaws.com" +
                        ec2_url[:-4] + "csv", stream=True)
    if info.status_code != 200:
        log.error("Cannot download Amazon data")
        return

    log.info("Downloading Amazon data into %s", filename)
    with gzip.open(filename, mode="wb") as f:
        for block in info.iter_content(1024*32):
            f.write(block)


if __name__ == "__main__":
    # For testing purpose only. This module is not intended
    # to be run from command line, but from Snakefile
    download_amazon_data("/tmp/raw_amazon_data.csv.gz")
