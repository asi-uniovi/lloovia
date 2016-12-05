import pandas as pd


def simplify_amazon_data(f_input, f_output):
    '''Reads the csv downloaded from Amazon and extracts the columns
    relevant for the analysis. Writes the result in a new csv.
    '''

    # Pairs of desired columns and new abridged names
    wanted_data = [
            ("Instance Type", "Type"),
            ("Location", "Region"),
            ("PricePerUnit", "Price"),
            ("vCPU", "Cores"),
            ("Memory", "Mem"),
            ("Storage", "Disk"),
            ("Operating System", "OS"),
            ("Unit", "Unit"),
            ("LeaseContractLength", "Rsv"),
            ("PurchaseOption", "Popt"),
            ("Tenancy", "Tenancy"),
            ("PriceDescription", "Pdesc")
            ]
    # Read csv with pandas, select and rename desired columns and
    # fill missing values wit sensible defaults
    all_data = (pd.read_csv(f_input, header=5,
                            dtype={"Max IOPS/volume": str,
                                   "Max IOPS Burst Performance": str,
                                   "Provisioned": str})
                .loc[:, [c[0] for c in wanted_data]]
                .rename(columns=dict(wanted_data))
                .fillna({"Rsv": "No", "Popt": "N/A"})
                )
    # Write result to new csv
    all_data.to_csv(f_output)


if __name__ == "__main__":
    # For testing purpose only. This module is not intended
    # to be run from command line, but from Snakefile
    simplify_amazon_data("/tmp/raw_amazon_data.csv.gz",
                         "/tmp/amazon_data.csv")
