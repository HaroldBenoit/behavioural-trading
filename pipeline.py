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

from functools import reduce


from utils import compute_R_over_time

import os.path as osp

# from utils import compute_trade_sign




def extract_digit(df, k=0):
    """Extract the k_th digit of the trade price, where the unit has index 0, the first decimal has index 1 and the tends digits has index -1.

    Args:
        df (pd.Dataframe): Events dataframe with trade_price as a column
        k (int, optional): Defaults to 0, for the unit digit.

    Returns:
        pd.Series: Series mapping each trade price to its k-th digit
    """

    return ((10**k) * df.trade_price).astype(int) % 10



        


        

def plot_response_functions(response_functions, ticker, plot_path, freq = None, month=None, quarter = None, total = False):
    response_functions = pd.pivot_table(response_functions.apply(pd.Series), columns=response_functions.index)

    f,a = plt.subplots(5,2, figsize=(15,30), dpi=200)

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
    
    # average over all tickers
    if total:
        plot_title = f" {averaging_window} {digit_string} response function over all DOW - Timescale: {timescale}"
    
    f.suptitle(plot_title)
    
       
    os.makedirs(plot_path, exist_ok=True)
    if not total:
        plot_path = osp.join(plot_path, f"{averaging_window} {timescale}-{ticker}-{k}th-digit-response.png")
    else:
        plot_path = osp.join(plot_path, f"Total-{averaging_window}-{timescale}--{k}th-digit-response.png")

    
    plt.savefig(plot_path, bbox_inches='tight')


if __name__ == "__main__":

    ## Init
    parser = argparse.ArgumentParser(prog="Python script to compute digit response functions on the DOW Jones")
    parser.add_argument('--plot_path', default="behavioural-trading/plots/", help="Absolute or relative path to appropriate folder to store the resulting plots")
    parser.add_argument('--digit', type=int, default=-1, help="Digit of the price to extract, here the unit has index 0, the first decimal has index 1 and the tends digits has index -1.")
    parser.add_argument('--tau_max', type=int, default=1000,help="How many shifts we compute for the response function")
    parser.add_argument('--freq', default=None,help="If argument given, it specifies that we want to use physical time scale instead of trade time scale, and it specifies the frequency or precision .e.g 1s, 2s, 1min, 2min ")
    parser.add_argument('--monthly', action="store_true", help="If argument given, the script will compute the and plot the response function for each month in the year")
    parser.add_argument('--quarterly', action="store_true", help="If argument given, the script will compute the and plot the response function for each quarter in the year")

    args = parser.parse_args()

    datasets = glob.glob("data/clean/DOW/*events_w_s.arrow")
    ## RESPONSE FUNCTION
    
    
    k=args.digit
    all_response_functions = []
    
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
            plot_path = osp.join(args.plot_path, f"{ticker}", "yearly")
            plot_response_functions(response_functions, ticker, plot_path, args.freq)
            
            ## we standardize by mean mid price
            all_response_functions.append(response_functions / mean)
            
            
    if len(all_response_functions) > 0:
            
        all_response_functions = reduce(lambda a,b: a +b, all_response_functions) / len(response_functions)
        
        plot_path = osp.join(args.plot_path)
        
        plot_response_functions(all_response_functions, ticker="", plot_path=plot_path, freq= args.freq, total=True)



        
