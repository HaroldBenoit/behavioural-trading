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
import os

from utils import compute_R_over_time

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

    ## now that we have clasassified the upticks, if uptick = True and s=0.0 => it is a buy-trade, if uptick=False and s=0.0 => it is a sell-trade, if uptick = NaN and s=0.0 => take last trade classification

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
        print("Saving to " +path)
        df_v.export_arrow(path)


def plot_response_functions(response_functions, ticker, plot_path, freq = None, month=None, quarter = None):
    response_functions = pd.pivot_table(response_functions.apply(pd.Series), columns=response_functions.index)

    f,a = plt.subplots(5,2, figsize=(30,15), dpi=200)

    print(response_functions.shape)
    print(f"Plotting ticker: {ticker}")
    for i,ax in zip(range(10),a.flatten()):
        #re.match("/(.*)-events*", dataset)[0]
        #print(response_functions.iloc[:,0+i::10])
        curr_response = response_functions.iloc[:,0+i::10]
        if not(curr_response.empty):
            curr_response.plot(ax=ax)
    
    timescale = "Trade" if freq is None else f"Freq {args.freq}"        
    if k == -1:
        digit_string = "Tens digit"
    elif k == 0:
        digit_string = "Unit digit"
    elif k ==1:
        digit_string = "1st decimal"
    elif k==2:
        digit_string= "2nd decimal"
    elif k == 3:
        digit_string= "3rd decimal"
    elif k > 3:
        digit_string= f"{k}th decimal"
        
    
    averaging_window = "Month" if month else f"Q{q}" if quarter else "Yearly"
    
    plot_title = f" {averaging_window} {ticker} {digit_string} response function - Timescale: {timescale} - Price summary mean:{mean:.3f}, min:{min:.3f}, max: {max:.3f}, Tot. volume: {total_volume:.2E}"
    f.suptitle(plot_title)
    
       
    os.makedirs(plot_path, exist_ok=True)
    plot_path = osp.join(plot_path, f"{averaging_window} {timescale}-{ticker}-{k}th-digit-response.png")
    plt.savefig(plot_path)


if __name__ == "__main__":

    ## Init
    parser = argparse.ArgumentParser(prog="Python script to compute digit response functions on the DOW Jones")
    parser.add_argument("--process", action="store_true", help="If flag specified, the raw data needs to be processed into clean data")
    parser.add_argument('--plot_path', default="behavioural-trading/plots/", help="Absolute or relative path to appropriate folder to store the resulting plots")
    parser.add_argument('--digit', type=int, default=-1, help="Digit of the price to extract, here the unit has index 0, the first decimal has index 1 and the tends digits has index -1.")
    parser.add_argument('--tau_max', type=int, default=1000,help="How many shifts we compute for the response function")
    parser.add_argument('--freq', default=None,help="If argument given, it specifies that we want to use physical time scale instead of trade time scale, and it specifies the frequency or precision .e.g 1s, 2s, 1min, 2min ")
    parser.add_argument('--monthly', action="store_true", help="If argument given, the script will compute the and plot the response function for each month in the year")
    parser.add_argument('--quarterly', action="store_true", help="If argument given, the script will compute the and plot the response function for each quarter in the year")

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
            all_promises.append(load_and_compute_trade_sign(dataset, True))
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
        events.set_index("index",inplace=True)
        events["month"] = events.index.month
        
        #computing statstics before modifying events because of physical time scale
        mean, min, max, total_volume = events.mid.mean(), events.mid.min(), events.mid.max(), events.trade_volume.sum()

        ## enter if loop if we're using physical time scale
        if args.freq is not None:
            events = events.groupby(pd.Grouper(freq=args.freq)).last()
            events.loc[events.mid.isna(), 'mid']  = 0.0
            events.loc[events.s.isna(),'s'] = 0.0
            # when considering a precision of, for example 1 second, we consider the correct midprice to be the last one
            # if during a time step, no trade occured, we act as if the trade sign is equal to zero i.e. no influence on the response function 
            # (taken from paper "Price response functions and spread impact in correlated financial markets" https://arxiv.org/pdf/2010.15105.pdf)
        
        ticker = re.search(".*\/(.*)\-events.*", dataset).groups(0)[0]

        if args.monthly:
            events["month"] = pd.Categorical(events.month)

            for m in events.month.unique():
                response_functions = events[events.quarter==m].groupby(["trade_sign", "unit_digit"]).apply(
                    lambda x: compute_R_over_time(x, tau_max=args.tau_max)
                ) 
                plot_response_functions(
                    response_functions, 
                    ticker, 
                    plot_path = osp.join(args.plot_path,f"{ticker}", f"M{m}"), 
                    freq = args.freq, 
                    month = m,
                    )
        elif args.quarterly:
            events["quarter"]=pd.Categorical(((events["month"]-1)/3+1).astype(int))
            for q in events.quarter.unique():
                response_functions = events[events.quarter==q].groupby(["trade_sign", "unit_digit"]).apply(
                    lambda x: compute_R_over_time(x, tau_max=args.tau_max)
                ) 
                plot_response_functions(
                    response_functions, 
                    ticker, 
                    plot_path = osp.join(args.plot_path,f"{ticker}", f"Q{q}"), 
                    freq = args.freq, 
                    quarter = q
                    )

        else :
            response_functions = events.groupby(["trade_sign", "unit_digit"]).apply(
                lambda x: compute_R_over_time(x, tau_max=args.tau_max)
            )
            plot_response_functions(response_functions, ticker, args.plot_path, args.freq)


        
