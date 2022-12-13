
import pandas as pd
import numpy as np



def compute_R(events: pd.DataFrame, tau_max=1000,dtau=1):
    taus=range(1,tau_max,dtau)
    R=[]
    R_plus=[]
    R_minus=[]
    for tau in taus:
        events_mid_shifted=events["mid"].shift(-tau)
        R.append(np.nanmean(events["s"]*(events_mid_shifted-events["mid"])))
    return np.array(R)   



from scipy.signal import convolve

def compute_R_fast(events: pd.DataFrame, tau_max=1000):
    #R definition is R(tau) = E[s_n(M_{n+tau} -M_n) ]

    s = events["s"]
    mid = events["mid"]

    N=len(mid)


    ## we can move out demeaning factor out of the response function equation
    ## just need to make sure we have the right amount of corresponding s*mid in the expected value =>
    ## need to use cumsum
    ## need to be careful to reverse the order of the sum because for example
    # the last term in mid will contribute only once (for the tau=0 computation)
    # and the first term in mid will contribute to all computations

    demean_const = np.cumsum((s*mid))[::-1]
    ## number of terms used for each tau, to normalize correctly the expected value computation
    division_const= np.array([i for i in range(1,N+1)])[::-1]


    ## as we're taking the expected value, 
    #the number of factors inside the expectation decreases linearly as the shift increases.
    # The most extreme example being the biggest shift where only one element contributes

    middle = N-1

    ## need to reverse mid as we want cross-correlation and not convolution
    conv = convolve(s,mid[::-1])

    ## we want only want half of the convolution and in causal order, thus we take first half and reverse
    conv = conv[:middle+1][::-1]

    #print("conv",conv)
    #print()
    #print("s*mid ",s*mid)
    #print()
    #print("demean", demean_const)

    response_function = (conv- demean_const)/division_const
    #print()
    #print("response", response_function)
    
    cutoff = min(len(response_function), tau_max)
    
    return np.array(response_function[:cutoff].values)




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
    

    events["mid"]=(events.bid+events.ask)*0.5
    events["s"]=np.sign(events["trade.price"]-events["mid"])

    print("Percentage of unclassifiable trades", f"{((events.s == 0.0).sum()/ len(events))*100:.2f}%")

    ## we need to resolve case where trade_price = mid_price (by using tick test described in the paper) following Lee's algo https://onlinelibrary.wiley.com/doi/full/10.1111/j.1540-6261.1991.tb02683.x

    #The tick test is a technique which infers the direction of a trade by-comparing its price to the price of the preceding trade(s).
    #The test classifies each trade into four categories: an uptick, a downtick, a zero-uptick, and a zero-downtick.
    #A trade is an uptick (downtick) if the price is higher (lower) than the price of the previous trade. When the price is the same as the previous trade (a zero tick),
    #if the last price change was an uptick, then the trade is a zero-uptick.
    #Similarly, if the last price change was a downtick, then the trade is a zero-downtick. 
    #A trade is classified as a buy if it occurs on an uptick or a zero-uptick; otherwise it is classified as a sell.

    uptick = pd.Series((events["trade.price"].shift(-1) -events["trade.price"]).iloc[:-1].values, index = events.iloc[1:].index)

    ## important to set nan first, since False == 0.0 in pandas
    uptick[uptick == 0.0] = np.nan
    uptick[uptick > 0.0] = True
    uptick[uptick < 0.0] = False

    ## now that we have classified the upticks, if uptick = True and s=0.0 => it is a buy-trade, if uptick=False and s=0.0 => it is a sell-trade, if uptick = NaN and s=0.0 => take last trade classification

    ## use ffill to take last trade classification
    events["uptick"] = uptick.ffill()

    ## applying the rule described above
    idx =  events[(events.s == 0.0)].index
    events["new_s"] = np.sign(events.loc[idx]["uptick"])
    events["new_s"] = events["new_s"].fillna(0.0)
    events["s"] = events["s"] + events["new_s"]

    ## cleaning up after
    events.drop(columns=["uptick","new_s"],inplace=True)
    
    return events




