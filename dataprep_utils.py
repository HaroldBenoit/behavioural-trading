import glob
import os
import re
import time
import dask
import pandas as pd
import vaex
import numpy as np

import os.path as osp

import pyarrow.parquet as pq
import vaex
from dask.distributed import Client


dask.config.set(scheduler="processes")

import warnings

pd.set_option('mode.chained_assignment', None)

# warnings.filterwarnings("error")

# @dask.delayed
def load_TRTH_trade(
    filename,
    tz_exchange="America/New_York",
    only_non_special_trades=True,
    only_regular_trading_hours=True,
    open_time="09:30:00",
    close_time="16:00:00",
    merge_sub_trades=True,
):
    try:
        if re.search("(csv|csv\\.gz)$", filename):
            DF = pd.read_csv(filename)
        if re.search(r"arrow$", filename):
            DF = pd.read_arrow(filename)
        if re.search("parquet$", filename):
            DF = pd.read_parquet(filename)

    except Warning as w:
        print("Warning for ", filename)
    except Exception as e:
        print("load_TRTH_trade could not load " + filename)
        print(e)
        return None

    try:
        DF.shape
    except Exception as e:  # DF does not exist
        print("DF does not exist")
        print(e)
        return None

    if DF.shape[0] == 0:
        return None

    if only_non_special_trades:
        DF = DF[DF["trade-stringflag"] == "uncategorized"]

    DF.drop(columns=["trade-rawflag", "trade-stringflag"], axis=1, inplace=True)

    DF.index = pd.to_datetime(DF["xltime"], unit="d", origin="1899-12-30", utc=True)
    DF.index = DF.index.tz_convert(
        tz_exchange
    )  # .P stands for Arca, which is based at New York
    DF.drop(columns="xltime", inplace=True)

    if only_regular_trading_hours:
        DF = DF.between_time(
            open_time, close_time
        )  # warning: ever heard e.g. about Thanksgivings?

    if merge_sub_trades:
        DF = DF.groupby(DF.index).agg(
            trade_price=pd.NamedAgg(column="trade-price", aggfunc="mean"),
            trade_volume=pd.NamedAgg(column="trade-volume", aggfunc="sum"),
        )

    return DF


# @dask.delayed
def load_TRTH_bbo(
    filename,
    tz_exchange="America/New_York",
    only_regular_trading_hours=True,
    merge_sub_trades=True,
):
    try:
        if re.search(r"(csv|csv\.gz)$", filename):
            DF = pd.read_csv(filename)
        if re.search(r"arrow$", filename):
            DF = pd.read_arrow(filename)
        if re.search(r"parquet$", filename):
            DF = pd.read_parquet(filename)

    except Warning as w:
        print("Warning for ", filename)

    except Exception as e:
        print("load_TRTH_bbo could not load " + filename)
        return None

    try:
        DF.shape
    except Exception as e:  # DF does not exist
        print("DF does not exist")
        print(e)
        return None

    if DF.shape[0] == 0:
        # print("Empty DF ", filename)
        return None

    DF.index = pd.to_datetime(DF["xltime"], unit="d", origin="1899-12-30", utc=True)
    DF.index = DF.index.tz_convert(
        tz_exchange
    )  # .P stands for Arca, which is based at New York
    DF.drop(columns="xltime", inplace=True)

    if only_regular_trading_hours:
        DF = DF.between_time("09:30:00", "16:00:00")  # ever heard about Thanksgivings?

    if merge_sub_trades:
        DF = DF.groupby(DF.index).last()

    return DF


@dask.delayed
def load_merge_trade_bbo(
    ticker,
    date,
    bbo_path,
    trade_path,
    country="US",
    dirBase="data/flash_crash_DJIA/US",
    suffix="parquet",
    suffix_save=None,
    dirSaveBase="data/clean/flash_crash_DJIA/US/events",
    saveOnly=False,
    doSave=False,
):

    trades = load_TRTH_trade(trade_path)
    bbos = load_TRTH_bbo(bbo_path)
    if trades is None:
        # print("Trade is none : ", trade_path )
        return None
    if bbos is None:
        # print("BBO is none")
        return None
    try:
        trades.shape + bbos.shape
    except:
        print("Couln't broadcast trade shapes to bbos")
        return None

    events = trades.join(bbos, how="left")

    if doSave:
        dirSave = dirSaveBase + "/" + country + "/events/" + ticker
        if not os.path.isdir(dirSave):
            os.makedirs(dirSave)

        if suffix_save:
            suffix = suffix_save

        file_events = dirSave + "/" + date + "-" + ticker + "-events" + "." + suffix
        # pdb.set_trace()

        saved = False

        """ if suffix == "arrow":
            events = vaex.from_pandas(events, copy_index=True)
            events.export_arrow(file_events)
            saved = True
        if suffix == "parquet":
            #   pdb.set_trace()
            events.to_parquet(file_events, use_deprecated_int96_timestamps=True)
            saved = True

        if not saved:
            print("suffix " + suffix + " : format not recognized")

        if saveOnly:
            return saved """
    return events



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



def compute_trade_sign_all_datasets():
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
        