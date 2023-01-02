import pandas as pd
import pyarrow.parquet as pq
import numpy as np
import matplotlib.pyplot as plt
import vaex
import dask
#from utils import compute_trade_sign

import glob
from dask.distributed import Client
import time

def compute_trade_sign(events:pd.DataFrame):
    """ Computes the sign of a trade for each trade in a intraday trade dataframe. The sign of a trade represents whether the trade was buy-initiated or sell-initiated.
        Trade sign is defined as: 
        1. if trade price above mid-price => buy
        2. if trade price below mid-price => sell
        3. if trade price equal to mid-price => apply tick-test

    Args:
        events (pd.DataFrame): Intraday trade data, assumes time-ordering of the trades

    Returns:
        _type_: intraday trade data with additional column "s", representing the sign of the trade
    """

    events["mid"] = (events["bid-price"] + events["ask-price"]) * 0.5
    events = events.fillna(method="ffill").dropna()
    events["s"] = events["trade_price"] - events["mid"]
    #print(events["s"])
    events["s"] = np.sign(events["s"])

    print(
        "Percentage of unclassifiable trades",
        f"{((events.s == 0.0).sum()/ len(events))*100:.2f}%",
    )

    ## we need to resolve case where trade_price = mid_price (by using tick test described in the paper) following Lee's algo https://onlinelibrary.wiley.com/doi/full/10.1111/j.1540-6261.1991.tb02683.x

    # The tick test is a technique which infers the direction of a trade by-comparing its price to the price of the preceding trade(s).
    # The test classifies each trade into four categories: an uptick, a downtick, a zero-uptick, and a zero-downtick.
    # A trade is an uptick (downtick) if the price is higher (lower) than the price of the previous trade. When the price is the same as the previous trade (a zero tick),
    # if the last price change was an uptick, then the trade is a zero-uptick.
    # Similarly, if the last price change was a downtick, then the trade is a zero-downtick.
    # A trade is classified as a buy if it occurs on an uptick or a zero-uptick; otherwise it is classified as a sell.

    uptick = pd.Series(
        (events["trade_price"].shift(-1) - events["trade_price"]).iloc[:-1].values,
        index=events.iloc[1:].index,
    )

    ## important to set nan first, since False == 0.0 in pandas
    uptick[uptick == 0.0] = np.nan
    uptick[uptick > 0.0] = True
    uptick[uptick < 0.0] = False

    ## now that we have clbasassified the upticks, if uptick = True and s=0.0 => it is a buy-trade, if uptick=False and s=0.0 => it is a sell-trade, if uptick = NaN and s=0.0 => take last trade classification

    ## use ffill to take last trade classification
    events["uptick"] = uptick.ffill()

    ## applying the rule described above
    idx = events[(events.s == 0.0)].index
    events["new_s"] = np.sign(events.loc[idx]["uptick"])
    events["new_s"] = events["new_s"].fillna(0.0)
    events["s"] = events["s"] + events["new_s"]

    ## cleaning up after
    events.drop(columns=["uptick", "new_s"], inplace=True)

    return events

@dask.delayed
def load_and_compute_trade_sign(path):
    print("Processing",path)
    df = vaex.open(path).to_pandas_df()
    df = compute_trade_sign(df)
    vaex.from_pandas(df, copy_index=True).export_arrow(path)
    del df


if __name__ == "__main__":

    client = Client(n_workers = 1, threads_per_worker=4)

    client.amm.start()

    datasets = glob.glob("data/clean/DOW/*")

    print(len(datasets))
    t1 = time.time()
    print("Computing trade sign of", len(datasets), "datasets")
    all_promises=[]
    for dataset in datasets:
        all_promises.append(load_and_compute_trade_sign(dataset))
    dask.compute(all_promises, optimize_graph=False)
    t2 = time.time()
    print("Computation took", (t2-t1), "seconds")