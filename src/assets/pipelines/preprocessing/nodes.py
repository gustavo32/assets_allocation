import pandas as pd
import numpy as np

from stockstats import StockDataFrame as Sdf
from functools import reduce


def _get_change_between_days(df):
    stock = Sdf.retype(df.copy())
    df["Close"] = stock["change"].values
    return df


def resample_data(index, resample_rate=None, strategy=None):
    assert resample_rate.upper() in ["W", "M", "Q", "Y", None]#, "resample rate needs to be W, M, Q, Y or None"
    assert strategy.upper() in ["SUM", "KEEP_LAST", None]#, "just SUM, KEEP_LAST and None strategies are implemented"

    if strategy is None:
        return index
    elif strategy.upper() == "SUM":
        return index.resample(resample_rate).sum()
    elif strategy.upper() == "KEEP_LAST":
        daily_df = index.asfreq("D", method="ffill")
        df = daily_df.asfreq(resample_rate)
        if "Volume" in index.columns:
            df["Volume"] = index.resample(resample_rate).sum()["Volume"]

        return df


def _make_signal_from_return(sreturn, index, resample_rate="W", total_business_days=252):
    df = pd.DataFrame({"Close": [0]*len(index)}, index=index)
    df["Close"] = sreturn/total_business_days
    df = resample_data(df, resample_rate, "SUM")
    return df


def create_tech_indicators(df, indicators, return_risk, add_return_risk):
    stock = Sdf.retype(df.copy())
    feat_dict = {}

    for ind in indicators:
        feat_dict[ind] = stock[ind].values

    df_ind = pd.DataFrame(feat_dict, index=df.index).fillna(0)
    if add_return_risk:
        df_ind["return_risk"] = df_ind.close

    #     if return_risk.add_risk:
    #         series_risk = df_ind[return_risk.risk_col].fillna(0)
    #         total_risk = series_risk[series_risk < 0]
    #         total_risk = 0 if len(total_risk) == 0 else total_risk
    #         df_ind["return_risk"] = df_ind.return_risk/(total_risk+0.5)
    #     df_ind["return_risk"] = df_ind.return_risk.clip(0, return_risk.clip_max)
    return df_ind


def prefixado_signal(ipca, prefixado_return, business_dates, resample_rate):
    prefixado_return_signal = _make_signal_from_return(prefixado_return, business_dates.index,
                                                       resample_rate=resample_rate)
    ipca["Close"] =  prefixado_return_signal.Close - ipca.Close
    return ipca


def posfixado_cdi_signal(cdi, ipca, cdi_percentage):
    cdi["Close"] = (cdi.Close * cdi_percentage / 100) - ipca.Close
    return cdi


def posfixado_ipca_signal(ipca, ipca_return, business_dates, resample_rate):
    ipca_return_signal = _make_signal_from_return(ipca_return, business_dates.index, resample_rate=resample_rate)
    ipca["Close"] = (ipca.Close + ipca_return_signal.Close) * 0.8 - ipca.Close
    return ipca


def bvsp_signal(bvsp, ipca):
    bvsp = _get_change_between_days(bvsp)
    bvsp["Close"] = bvsp.Close - ipca.Close
    return bvsp


def global_vix_signal(global_vix, ipca):
    global_vix = _get_change_between_days(global_vix)
    global_vix["Close"] = global_vix.Close - ipca.Close
    return global_vix


def brazil_vix_signal(brazil_vix, ipca):
    brazil_vix = _get_change_between_days(brazil_vix)
    brazil_vix["Close"] = brazil_vix.Close - ipca.Close
    return brazil_vix


def ifix_signal(ifix, ipca):
    ifix = _get_change_between_days(ifix)
    ifix["Close"] = ifix.Close - ipca.Close
    return ifix


def ivvb11_signal(ivvb11, ipca):
    ivvb11 = _get_change_between_days(ivvb11)
    ivvb11["Close"] = ivvb11.Close - ipca.Close
    return ivvb11
