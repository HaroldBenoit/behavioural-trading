import glob
import re
import time

import dask
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import vaex
from dask.distributed import Client
import argparse

from utils import compute_R_fast

import os.path as osp

# from utils import compute_trade_sign


def compute_trade_sign(events: pd.DataFrame):
    """Computes the sign of a trade for each trade in a intraday trade dataframe. The sign of a trade represents whether the trade was buy-initiated or sell-initiated.
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
    # print(events["s"])
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
    ## 2*int(False)-1 = -1 | 2*int(True)-1=1
    events["new_s"] = 2 * (events.loc[idx]["uptick"].astype(int)) - 1
    events["new_s"] = events["new_s"].fillna(0.0)
    events["s"] = events["s"] + events["new_s"]

    ## cleaning up after
    events.drop(columns=["uptick", "new_s"], inplace=True)

    return events


def extract_digit(df, k=0):
    """Extract the k_th digit of the trade price, where the unit has index 0, the first decimal has index 1 and the tends digits has index -1.

    Args:
        df (pd.Dataframe): Events dataframe with trade_price as a column
        k (int, optional): Defaults to 0, for the unit digit.

    Returns:
        pd.Series: Series mapping each trade price to its k-th digit
    """

    return ((10**k) * df.trade_price).astype(int) % 10


@dask.delayed
def load_and_compute_trade_sign(path, save=False):
    print("Processing", path)
    df = vaex.open(path).to_pandas_df()
    df = compute_trade_sign(df)
    if save:
        path = path.replace("events", "events_w_s")
        print("Saving to ", path)
        # df.to_pickle(path)
        df_v = vaex.from_pandas(df)
        # print(df_v.schema)
        df_v.export_arrow(path)


if __name__ == "__main__":

    ## Init
    parser = argparse.ArgumentParser(prog="pipeline")
    parser.add_argument("--process", action="store_true")
    parser.add_argument('--plot_path', default="behavioural-trading/plots/")
    parser.add_argument('--digit', type=int, default=-1)
    args = parser.parse_args()

    if args.process:
        client = Client(n_workers=1, threads_per_worker=4)

        client.amm.start()

        datasets = glob.glob("data/clean/DOW/*events.arrow")

        print(len(datasets))
        ## TRADE SIGN
        t1 = time.time()
        print("Computing trade sign of", len(datasets), "datasets")
        all_promises = []
        for dataset in datasets:
            all_promises.append(load_and_compute_trade_sign(dataset, False))
        dask.compute(all_promises, optimize_graph=False)
        t2 = time.time()
        print("Computation took", (t2 - t1), "seconds")

    datasets = glob.glob("data/clean/DOW/*events_w_s.arrow")
    ## RESPONSE FUNCTION
    k=args.digit
    for dataset in datasets:
        events = vaex.open(dataset).to_pandas_df()
        events["unit_digit"] = pd.Categorical(extract_digit(events, k=k))
        events["trade_sign"] = pd.Categorical(events["s"].apply(lambda x : "BUY" if x == 1 else "SELL"))

        response_functions = events.groupby(["trade_sign", "unit_digit"]).apply(
            lambda x: compute_R_fast(x, tau_max=1000)
        )

        response_functions = pd.pivot_table(response_functions.apply(pd.Series), columns=response_functions.index)

        

        f,a = plt.subplots(5,2, figsize=(30,15), dpi=200)

        ticker = re.search(".*\/(.*)\-events.*", dataset).groups(0)[0]
        for i,ax in zip(range(10),a.flatten()):
            #re.match("/(.*)-events*", dataset)[0]
            #print(response_functions.iloc[:,0+i::10])
            curr_response = response_functions.iloc[:,0+i::10]
            if not(curr_response.empty):
                curr_response.plot(ax=ax)
                
        f.suptitle(f"{ticker} unit digit response function")
            
        plot_path = osp.join(args.plot_path,f"{ticker}-{k}th-digit-response.png")
        plt.savefig(plot_path)
