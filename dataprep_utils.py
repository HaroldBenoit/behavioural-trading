import glob
import os
import re

import dask
import pandas as pd
import vaex

dask.config.set(scheduler="processes")

import warnings

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
