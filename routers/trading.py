from typing import Any, Dict, List
from fastapi import APIRouter, Body, Depends, File, Query, UploadFile, HTTPException
import numpy as np
import pandas as pd

router = APIRouter()

def __get_period(df: pd.DataFrame) -> int:
    start_date = pd.to_datetime(df.index[0])
    end_date = pd.to_datetime(df.index[-1])
    days_between = (end_date - start_date).days
    return days_between

def __annualize(rate, period):
    if period < 360:
        rate = ((rate - 1) / period * 365) + 1
    elif period > 365:
        rate = rate ** (365 / period)
    return round(rate, 4)

def __get_sharpe_ratio(df, rf_rate):
    '''
    Calculate sharpe ratio
    :param df:
    :param rf_rate:
    :return: Sharpe ratio
    '''
    period = __get_period(df)
    rf_rate_daily = rf_rate / 365 + 1
    df['exs_rtn'] = df['acc_rtn'] - rf_rate_daily
    exs_rtn_annual = (__annualize(df['acc_rtn'].iloc[-1], period) - 1) - rf_rate
    sharpe_ratio = exs_rtn_annual / (df['exs_rtn'].std() * np.sqrt(365))
    return sharpe_ratio

@router.post("/indicator_to_signal")
async def indicator_to_signal(
    df: List[Dict[str, Any]] = Body(...), 
    factor: str = Query(...), 
    buy: float = Query(...), 
    sell: float = Query(...)
):
    '''
    Makes buy or sell signals according to factor indicator
    :param df: The dataframe containing stock prices and indicator data
    :param factor: The indicator to determine how to trade
    :param buy: The price level to buy
    :param sell: The price level to sell
    :return: The dataframe containing trading signal
    
    특정 factor를 이용해 트레이딩 했을 경우 성과가 어땠는지 확인하는 과정
    buy(매수),sell(공매도),zero(무)
    '''
    df = pd.DataFrame(df).set_index('index')
    df['trade'] = np.nan
    if buy >= sell:
        df['trade'].mask(df[factor]>buy, 'buy', inplace=True)
        df['trade'].mask(df[factor]<sell, 'zero', inplace=True)
    else:
        df['trade'].mask(df[factor]<buy, 'buy', inplace=True)
        df['trade'].mask(df[factor]>sell, 'zero', inplace=True)
    df['trade'].fillna(method='ffill', inplace=True)
    df['trade'].fillna('zero', inplace=True)
    # df=df['trade']
    return df.reset_index().to_dict(orient='records')

@router.post("/band_to_signal")
async def band_to_signal(
    df: List[Dict[str, Any]] = Body(...), 
    buy: str = Query(...), 
    sell: str = Query(...)
):
    '''
    Makes buy or sell signal according to band formation
    :param df: The dataframe containing stock prices and band data
    :param buy: The area in band to buy
    :param sell: The area in band to sell
    :return: The dataframe containing trading signal
    '''
    df = pd.DataFrame(df).set_index('index')
    symbol = df.columns[0]
    df['trade'] = np.nan

    # buy signals
    if buy == 'A':
        df['trade'].mask(df[symbol] > df['ub'], 'buy', inplace=True)
    elif buy == 'B':
        df['trade'].mask((df['ub'] > df[symbol]) & (df[symbol] > df['center']), 'buy', inplace=True)
    elif buy == 'C':
        df['trade'].mask((df['center'] > df[symbol]) & (df[symbol] > df['lb']), 'buy', inplace=True)
    elif buy == 'D':
        df['trade'].mask(df['lb'] > df[symbol], 'buy', inplace=True)

    # sell signals
    if sell == 'A':
        df['trade'].mask(df[symbol] > df['ub'], 'zero', inplace=True)
    elif sell == 'B':
        df['trade'].mask((df['ub'] > df[symbol]) & (df[symbol] > df['center']), 'zero', inplace=True)
    elif sell == 'C':
        df['trade'].mask((df['center'] > df[symbol]) & (df[symbol] > df['lb']), 'zero', inplace=True)
    elif sell == 'D':
        df['trade'].mask(df['lb'] > df[symbol], 'zero', inplace=True)

    df['trade'].fillna(method='ffill', inplace=True)
    df['trade'].fillna('zero', inplace=True)

    return df.reset_index().to_dict(orient='records')

def combine_signal_and(df, *cond):
    '''
    Combine signals as intersection
    :param df: Dataframe containing historical prices
    :param cond: Columns to be combined
    :return: Dataframe of selected signals
    '''
    for c in cond:
        df['trade'].mask((df['trade'] == 'buy') & (df[c] == 'buy'), 'buy', inplace=True)
        df['trade'].mask((df['trade'] == 'zero') | (df[c] == 'zero'), 'zero', inplace=True)
    return df


def combine_signal_or(df, *cond):
    '''
    Combine signals as union
    :param df: Dataframe containing historical prices
    :param cond: Columns to be combined
    :return: Dataframe of selected signals
    '''
    for c in cond:
        df['trade'].mask((df['trade'] == 'buy') | (df[c] == 'buy'), 'buy', inplace=True)
        df['trade'].mask((df['trade'] == 'zero') & (df[c] == 'zero'), 'zero', inplace=True)
    return df

@router.post("/position")
async def position(
    df: List[Dict[str, Any]] = Body(...), 
    ):
    '''
    Determine the position of portfolio according to trading signals
    :param df: The dataframe containing trading signal
    :return: The dataframe containing trading position
    
    포트폴리오 포지션 결정
    '''
    df = pd.DataFrame(df).set_index('index')
    df['position'] = ''
    df['position'].mask((df['trade'].shift(1)=='zero') & (df['trade']=='zero'), 'zz', inplace=True)
    df['position'].mask((df['trade'].shift(1)=='zero') & (df['trade']=='buy'), 'zl', inplace=True)
    df['position'].mask((df['trade'].shift(1)=='buy') & (df['trade']=='zero'), 'lz', inplace=True)
    df['position'].mask((df['trade'].shift(1)=='buy') & (df['trade']=='buy'), 'll', inplace=True)
    
    df['position_chart'] = 0
    df['position_chart'].mask(df['trade']=='buy', 1, inplace=True)
    # df=df['position']
    return df.reset_index().to_dict(orient='records')

@router.post("/evaluate")
async def evaluate(
    df: List[Dict[str, Any]] = Body(...), 
    cost: float = 0.001
):
    '''
    Calculate daily returns and MDDs of portfolio
    :param df: The dataframe containing trading position
    :param cost: Transaction cost when sell
    :return: Returns, MDD
    
    일일 수익률 및 포트폴리오의 최대낙폭(MDD) 계산
    매개변수(df): 거래 포지션이 포함된 데이터프레임
    매개변수(cost): 매도 시 거래 비용
    반환값: 수익률, 최대낙폭(MDD)
    '''
    df = pd.DataFrame(df).set_index('index')
    df['signal_price'] = np.nan

    df['signal_price'].mask(df['position'] == 'zl', df.iloc[:, 0], inplace=True)
    df['signal_price'].mask(df['position'] == 'lz', df.iloc[:, 0], inplace=True)

    print(df.tail(5))  # Debugging print statement to check the tail of the dataframe

    # Ensure signal_price is float and handle NaN values
    df['signal_price'] = pd.to_numeric(df['signal_price'], errors='coerce')

    record = df[['position', 'signal_price']].dropna()
    record['rtn'] = 1

    record['rtn'].mask(record['position'] == 'lz', (record['signal_price'] * (1 - cost)) / record['signal_price'].shift(1), inplace=True)
    record['acc_rtn'] = record['rtn'].cumprod()

    df['signal_price'].mask(df['position'] == 'll', df.iloc[:, 0], inplace=True)
    df['signal_price'] = pd.to_numeric(df['signal_price'], errors='coerce')

    df['rtn'] = record['rtn']
    df['rtn'].fillna(1, inplace=True)
    df['daily_rtn'] = 1
    df['daily_rtn'].mask(df['position'] == 'll', df['signal_price'] / df['signal_price'].shift(1), inplace=True)
    df['daily_rtn'].mask(df['position'] == 'lz', (df['signal_price'] * (1 - cost)) / df['signal_price'].shift(1), inplace=True)
    df['daily_rtn'].fillna(1, inplace=True)
    df['acc_rtn'] = df['daily_rtn'].cumprod()
    df['acc_rtn_dp'] = ((df['acc_rtn'] - 1) * 100).round(2)
    df['mdd'] = (df['acc_rtn'] / df['acc_rtn'].cummax() - 1).round(4)
    df['bm_mdd'] = (df.iloc[:, 0] / df.iloc[:, 0].cummax() - 1).round(4)
    df.drop(columns='signal_price', inplace=True)
    return df.reset_index().to_dict(orient='records')

@router.post("/performance")
async def performance(
    df: List[Dict[str, Any]] = Body(...), 
    rf_rate: float = 0.01
):
    df = pd.DataFrame(df).set_index('index')
    df.index = pd.to_datetime(df.index)
    
    rst = {}
    rst['no_trades'] = int((df['position'] == 'zl').sum())
    rst['no_win'] = int((df['rtn'] > 1).sum())
    rst['acc_rtn'] = float(df['acc_rtn'].iloc[-1].round(4))
    rst['hit_ratio'] = float(round((df['rtn'] > 1).sum() / rst['no_trades'], 4)) if rst['no_trades'] > 0 else 0
    rst['avg_rtn'] = float(round(df[df['rtn'] != 1]['rtn'].mean(), 4))
    rst['period'] = __get_period(df)
    rst['annual_rtn'] = float(__annualize(rst['acc_rtn'], rst['period']))
    rst['bm_rtn'] = float(round(df.iloc[-1, 0] / df.iloc[0, 0], 4))
    rst['sharpe_ratio'] = float(__get_sharpe_ratio(df, rf_rate))
    rst['mdd'] = float(df['mdd'].min())
    rst['bm_mdd'] = float(df['bm_mdd'].min())

    result = {
        "CAGR(연평균수익률)": '{:.2%}'.format(rst['annual_rtn'] - 1),
        "Accumulated return(누적수익률)": '{:.2%}'.format(rst['acc_rtn'] - 1),
        "Average return(평균수익률)": '{:.2%}'.format(rst['avg_rtn'] - 1) if not pd.isna(rst['avg_rtn']) else 'nan%',
        "Benchmark return(전략x의 경우 수익률)": '{:.2%}'.format(rst['bm_rtn'] - 1),
        "거래횟수": rst['no_trades'],
        "성공 횟수(플러스 수익률)": rst['no_win'],
        "Hit ratio(전략이 성공한 확률 성공횟수/거래횟수)": '{:.2%}'.format(rst['hit_ratio']),
        "Investment period (years)(테스트를 진행한 총 투자기간)": '{:.1f} yrs'.format(rst['period'] / 365),
        "Sharpe ratio(위험 대비 수익 비율)": '{:.2f}'.format(rst['sharpe_ratio']),
        "MDD(최대낙폭)": '{:.2%}'.format(rst['mdd'] - 1),
        "Benchmark MDD(전략x의 경우 최대낙폭)": '{:.2%}'.format(rst['bm_mdd'] - 1)
    }
    return result

# @router.post("/performance")
# async def performance(
#     df: List[Dict[str, Any]] = Body(...), 
#     rf_rate: float = 0.01
# ):
#     '''
#     Calculate additional information of portfolio
#     :param df: The dataframe with daily returns
#     :param rf_rate: Risk free interest rate
#     :return: Number of trades, Number of wins, Hit ratio, Sharpe ratio, ...
#     '''
#     df = pd.DataFrame(df).set_index('index')
#     df.index = pd.to_datetime(df.index)
    
#     rst = {}
#     rst['no_trades'] = int((df['position'] == 'zl').sum())
#     rst['no_win'] = int((df['rtn'] > 1).sum())
#     rst['acc_rtn'] = float(df['acc_rtn'].iloc[-1].round(4))
#     rst['hit_ratio'] = round((df['rtn'] > 1).sum() / rst['no_trades'], 4) if rst['no_trades'] > 0 else 0
    
    
#     avg_rtn_series = df[df['rtn'] != 1]['rtn']
#     if not avg_rtn_series.empty and pd.api.types.is_numeric_dtype(avg_rtn_series):
#         rst['avg_rtn'] = round(avg_rtn_series.mean(), 4)
#     else:
#         rst['avg_rtn'] = float('nan')
    
#     rst['period'] = __get_period(df)
#     rst['annual_rtn'] = __annualize(rst['acc_rtn'], rst['period'])
#     rst['bm_rtn'] = round(df.iloc[-1, 0] / df.iloc[0, 0], 4)
#     rst['sharpe_ratio'] = float(__get_sharpe_ratio(df, rf_rate))
#     rst['mdd'] = float(df['mdd'].min())
#     rst['bm_mdd'] = float(df['bm_mdd'].min())

    # results = {
    #     "CAGR(연평균수익률)": '{:.2%}'.format(rst['annual_rtn'] - 1),
    #     "Accumulated return(누적수익률)": '{:.2%}'.format(rst['acc_rtn'] - 1),
    #     "Average return(평균수익률)": '{:.2%}'.format(rst['avg_rtn'] - 1) if not pd.isna(rst['avg_rtn']) else 'nan%',
    #     "Benchmark return(전략x의 경우 수익률)": '{:.2%}'.format(rst['bm_rtn'] - 1),
    #     "거래횟수": rst['no_trades'],
    #     "성공 횟수(플러스 수익률)": rst['no_win'],
    #     "Hit ratio(전략이 성공한 확률 성공횟수/거래횟수)": '{:.2%}'.format(rst['hit_ratio']),
    #     "Investment period (years)(테스트를 진행한 총 투자기간)": '{:.1f} yrs'.format(rst['period'] / 365),
    #     "Sharpe ratio(위험 대비 수익 비율)": '{:.2f}'.format(rst['sharpe_ratio']),
    #     "MDD(최대낙폭)": '{:.2%}'.format(rst['mdd'] - 1),
    #     "Benchmark MDD(전략x의 경우 최대낙폭)": '{:.2%}'.format(rst['bm_mdd'] - 1)
    # }

#     return results