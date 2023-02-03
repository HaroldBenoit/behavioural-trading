import numpy as np
import pandas as pd
from scipy.signal import convolve
import dask
from dask.distributed import Client

from functools import reduce


def extract_digit(df, k=0):
    """Extract the k_th digit of the trade price, where the unit has index 0, the first decimal has index 1 and the tends digits has index -1.

    Args:
        df (pd.Dataframe): Events dataframe with trade_price as a column
        k (int, optional): Defaults to 0, for the unit digit.

    Returns:
        pd.Series: Series mapping each trade price to its k-th digit
    """

    return ((10**k) * df.trade_price).astype(int) % 10


def compute_R(events, tau_max=1000, dtau=1):
    taus = range(1, tau_max, dtau)
    R = []
    R_plus = []
    R_minus = []
    for tau in taus:
        events_mid_shifted = events["mid"].shift(-tau)
        R.append(np.nanmean(events["s"] * (events_mid_shifted - events["mid"])))
    return np.array(R)


def compute_R_fast(events: pd.DataFrame, tau_max=1000):
    # R definition is R(tau) = E[s_n(M_{n+tau} -M_n) ]
    
    if len(events) == 0:
        return np.full(tau_max, np.nan)

    s = events["s"]
    mid = events["mid"]

    N = len(mid)

    ## we can move out demeaning factor out of the response function equation
    ## just need to make sure we have the right amount of corresponding s*mid in the expected value =>
    ## need to use cumsum
    ## need to be careful to reverse the order of the sum because for example
    # the last term in mid will contribute only once (for the tau=0 computation)
    # and the first term in mid will contribute to all computations

    demean_const = np.cumsum((s * mid))[::-1]
    ## number of terms used for each tau, to normalize correctly the expected value computation
    division_const = np.array([i for i in range(1, N + 1)])[::-1]

    ## as we're taking the expected value,
    # the number of factors inside the expectation decreases linearly as the shift increases.
    # The most extreme example being the biggest shift where only one element contributes

    middle = N - 1

    ## need to reverse mid as we want cross-correlation and not convolution
    conv = convolve(s, mid[::-1])

    ## we want only want half of the convolution and in causal order, thus we take first half and reverse
    conv = conv[: middle + 1][::-1]

    # print("conv",conv)
    # print()
    # print("s*mid ",s*mid)
    # print()
    # print("demean", demean_const)

    response_function = (conv - demean_const) / division_const
    # print()
    # print("response", response_function)

    cutoff = min(len(response_function), tau_max)

    return np.array(response_function[:cutoff].values)


@dask.delayed
def compute_R_correctly(events: pd.DataFrame, k:int, tau_max:int=1000):
    
    num_trades = len(events)
    
    digits = extract_digit(events,k=k)
    
    order_types = events["s"].apply(lambda x : "BUY" if x == 1 else "SELL")
    
    digits_response ={"BUY":[np.zeros(tau_max) for i in range(10)], "SELL":[np.zeros(tau_max) for i in range(10)]}
    
    digits_num_appearance = {"BUY":[0 for i in range(10)], "SELL":[0 for i in range(10)]}
     
    
    
    for i in range(num_trades - tau_max):
                
        window = events.iloc[i:i + tau_max]
        
        digit = digits[i]
        order_type = order_types[i]
        
        ## count num appearance
        
        digits_num_appearance[order_type][digit] = digits_num_appearance[order_type][digit] + 1
        
        ## computing contribution to response
        curr_mid_price = events.iloc[i].mid
        response = (window.s *(window.mid - curr_mid_price)).to_numpy()
        
        digits_response[order_type][digit] = digits_response[order_type][digit] + response
    
    
    ## averaging things out 
    
    all_responses=[]
    for order_type in ["BUY", "SELL"]:
        for i in range(10):
            if  digits_num_appearance[order_type][i] > 0: 
                avg_response = digits_response[order_type][i]/ digits_num_appearance[order_type][i]
            else:
                avg_response = np.zeros(tau_max)
            
            all_responses.append(avg_response)
            
    
    index = pd.MultiIndex.from_product([["BUY","SELL"],list(range(10))], names=["trade_sign", "digit"])

    return pd.Series(all_responses,index=index)
    


def compute_R_over_time_correctly(events: pd.DataFrame, k:int, tau_max:int=1000):
    
    client = Client(n_workers=1, threads_per_worker=8)
    
    promises = []
    
    for day in events.index.day_of_year.unique():
        
        curr_events = events.iloc[events.index.day_of_year == day]
        promises.append(compute_R_correctly(curr_events,k=k,tau_max=tau_max))
        
    response_functions_all_days = dask.compute(promises)[0]
    
    response_functions = reduce(lambda a,b: a +b, response_functions_all_days) / len(response_functions_all_days)

    return response_functions


def compute_R_over_time(events: pd.DataFrame, tau_max=1000):
       
    ## computing for each day, make sure to drop the nan days as there are only 252 trading days
    responses = events.groupby(pd.Grouper(freq="1D", origin='start_day')).apply(lambda x: compute_R_fast(x, tau_max=tau_max))
    
    #keeping only trading days
    
    trading_days = events.index.day_of_year.drop_duplicates()
    trading_days_mask = np.isin(responses.index.day_of_year.to_numpy(), trading_days)

    responses = responses.iloc[trading_days_mask]
    
    max_len = responses.apply(lambda x: len(x)).max()
    
    ## we pad every trading day response function up to max_len viewed (usually equal to tau_max)
    final_response = np.nanmean(np.array([np.pad(a, (0,max_len-a.shape[0]) , mode='constant', constant_values=np.nan) for a in responses.to_numpy()]),axis=0)
    
    return final_response


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

    events["mid"] = (events.bid + events.ask) * 0.5
    events["s"] = np.sign(events["trade.price"] - events["mid"])

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
        (events["trade.price"].shift(-1) - events["trade.price"]).iloc[:-1].values,
        index=events.iloc[1:].index,
    )

    ## important to set nan first, since False == 0.0 in pandas
    uptick[uptick == 0.0] = np.nan
    uptick[uptick > 0.0] = True
    uptick[uptick < 0.0] = False

    ## now that we have classified the upticks, if uptick = True and s=0.0 => it is a buy-trade, if uptick=False and s=0.0 => it is a sell-trade, if uptick = NaN and s=0.0 => take last trade classification

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
