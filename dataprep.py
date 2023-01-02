import glob
import os
import re

import dask
import dataprep_utils
import numpy as np
import pandas as pd
import vaex
from dask.distributed import Client

if __name__ == "__main__":

    client = Client(n_workers=1, threads_per_worker=8)

    bbo_files=glob.glob("data/flash_crash_DJIA/US/bbo/*/*")
    trade_files=glob.glob("data/flash_crash_DJIA/US/trade/*/*")
    bbo_files.sort()
    trade_files.sort()


    securities = list(zip(bbo_files, trade_files))

    paths=pd.DataFrame(securities, columns=["BBO_file", "TRADE_file"])

    paths["ticker"] = paths["BBO_file"].apply(lambda x: re.match(".*\/bbo\/(.*)\..*\/.*", x).groups()[0])
    paths["date"] = paths.apply(lambda x: re.match(f".*\/(.*)-{x['ticker']}.*", x["BBO_file"]).groups()[0], axis=1)

    #for ticker in ["RTX"]:

    for ticker in paths["ticker"].unique():
        if ticker not in ["HD", "RTX", "WBA"]:
            print(f"Merging TICKER :{ticker}")
            all_promises=paths[paths["ticker"]==ticker].apply(lambda x: dataprep_utils.load_merge_trade_bbo(x['ticker'], x['date'], x['BBO_file'], x['TRADE_file']), axis=1)
            events= dask.compute(all_promises.values.tolist())[0]
            events = pd.concat(events)

            events = vaex.from_pandas(events, copy_index=True)
            events = events[~events.trade_price.isnan()]
            dirSave = "data/clean/DOW/"
            fileSave =  dirSave + ticker + "-events" + ".arrow" 
            if not os.path.isdir(dirSave):
                os.makedirs(dirSave)
            events.export_arrow(fileSave)
        else:
            print(f"skipping {ticker}")

    print ("finished")


    