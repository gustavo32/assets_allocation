"""
This is a boilerplate pipeline 'ingestion'
generated using Kedro 0.17.7
"""
import pandas as pd
import yfinance as yf
import requests
import investpy as inv
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta


def _parse_month_value(df, current_date_format, stock_dates):
    df["month"] = pd.to_datetime(df.month, format=current_date_format)

    start = stock_dates.min()
    start = datetime(start.year, start.month, 1)
    end = stock_dates.max()
    index = pd.date_range(start, end)

    df = pd.DataFrame(df.set_index("month"), index=index)
    df = df.fillna(method="ffill")
    return pd.DataFrame(df, index=stock_dates)


def _next_day_if_none(end_date):
    return (datetime.today() + timedelta(days=1)).date() if end_date is None else end_date


def _date_to_brazil_str_format(date):
    return date.strftime("%d/%m/%Y")


def _prepare_dates_to_investpy(start_date, end_date):
    end_date = _next_day_if_none(end_date)
    end_date = _date_to_brazil_str_format(end_date)
    start_date = _date_to_brazil_str_format(start_date + timedelta(days=1))
    return start_date, end_date


def _format_output_data(df):
    return df[["Close", "Volume"]]


def _get_month_year(series):
    str_series = series.astype(str)
    return str_series.str[:4] + str_series.str[5:7]


def _shift_one_month(df, month_col):
    df[month_col] = pd.to_datetime(df[month_col], format="%Y%m")
    df[month_col] = df[month_col].apply(lambda x: x + relativedelta(months=1))
    df[month_col] = _get_month_year(df[month_col])
    return df


def _propagate_to_last_month(raw_df, df):
    df["month"] = _get_month_year(df.index)
    df["month"] = df.index.astype(str)
    df["month"] = df.month.str[:4] + df.month.str[5:7]

    counts = df.groupby("month").count()
    counts.columns = ["counts"]

    merged_df = counts.merge(raw_df, on="month", how="inner")
    merged_df["Close"] = merged_df.Close.astype(float) / merged_df.counts

    df = df.reset_index()
    df = df[["Date", "month"]]

    return df.merge(merged_df, on="month", how="left").set_index("Date").fillna(method="ffill")["Close"]


def get_ifix_data(start_date, end_date=None):
    index = "BM&Fbvsp Real Estate IFIX"
    start_date, end_date = _prepare_dates_to_investpy(start_date, end_date)
    df = inv.get_index_historical_data(index, country="Brazil", from_date=start_date, to_date=end_date)
    return _format_output_data(df)


def get_bvsp_data(start_date, end_date=None):
    return _format_output_data(yf.download("^BVSP", start=start_date, end=end_date))


def get_global_vix_data(start_date, end_date=None):
    return _format_output_data(yf.download("^VIX", start=start_date, end=end_date))


def get_brazil_vix_data(start_date, end_date=None):
    index = "CBOE Brazil Etf Volatility"
    start_date, end_date = _prepare_dates_to_investpy(start_date, end_date)
    df = inv.get_index_historical_data(index, country="united states", from_date=start_date, to_date=end_date)
    return _format_output_data(df)


def get_ivvb11_data(start_date, end_date=None):
    df = _format_output_data(yf.download("^GSPC", start=start_date, end=end_date))
    return df, df.index.to_series()


def get_ipca_data(stock_dates):
    stock_dates = stock_dates.index
    response = requests.get("https://apisidra.ibge.gov.br/values/t/1737/n1/1/v/63/p/all?formato=json")
    print(response)
    raw_df = pd.DataFrame.from_records(response.json()[1:])[["V", "D3C"]]

    raw_df.columns = ["Close", "month"]

    shifted_df = _shift_one_month(raw_df, "month")
    df = _parse_month_value(shifted_df.copy(), current_date_format="%Y%m", stock_dates=stock_dates)
    df = _propagate_to_last_month(shifted_df, df)

    return df


def get_cdi_data(stock_dates):
    stock_dates = stock_dates.index
    start_date = _date_to_brazil_str_format(stock_dates.min())
    end_date = _date_to_brazil_str_format(stock_dates.max())
    response = requests.get(
        f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.11/dados?formato=json&dataInicial={start_date}&dataFinal={end_date}")

    df = pd.DataFrame.from_records(response.json())
    df.columns = ["month", "Close"]
    return _parse_month_value(df, current_date_format="%d/%m/%Y", stock_dates=stock_dates)
