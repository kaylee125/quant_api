from typing import Any, Dict, List
from fastapi import APIRouter, Depends, File, Query, UploadFile, HTTPException
import numpy as np
import pandas as pd

router = APIRouter()

@router.post("/rsi")
async def rsi(
    df: List[Dict[str, Any]], 
    w=14):
    '''
    Calculate RSI indicator
    :param df: Dataframe containing historical prices
    :param w: Window size
    :return: Series of RSI values
    '''
    df = pd.DataFrame(df).set_index('index')
    pd.options.mode.chained_assignment = None
    symbol = df.columns[0]
    df.fillna(method='ffill', inplace=True)  # 들어온 데이터의 구멍을 메꿔준다
    df.replace([float('inf'), float('-inf')], 0, inplace=True)  # 무한대 값을 0으로 대체

    if len(df) > w:
        df['diff'] = df.iloc[:, 0].diff()   # 일별 가격차이 계산
        df['au'] = df['diff'].where(df['diff'] > 0, 0).rolling(w).mean()
        df['ad'] = df['diff'].where(df['diff'] < 0, 0).rolling(w).mean().abs()
        for r in range(w + 1, len(df)):
            df['au'].iloc[r] = (df['au'].iloc[r - 1] * (w - 1) + df['diff'].where(df['diff'] > 0, 0).iloc[r]) / w
            df['ad'].iloc[r] = (df['ad'].iloc[r - 1] * (w - 1) + df['diff'].where(df['diff'] < 0, 0).abs().iloc[r]) / w
        df['rsi'] = (df['au'] / (df['au'] + df['ad']) * 100).round(2)
        
        # NaN 값과 무한대 값을 적절한 값으로 대체
        df['rsi'].fillna(0, inplace=True)
        df.replace([float('inf'), float('-inf')], 0, inplace=True)
        
        print('df', df)
        return df[[symbol, 'rsi']].reset_index().to_dict(orient='records')
    else:
        return None

@router.post("/macd")
async def macd(
    df: List[Dict[str, Any]],
    short: int = 12,  
    long: int =26, 
    signal: int =9
    ):
    '''
    Calculate MACD indicators
    :param df: Dataframe containing historical prices
    :param short: Day length of short term MACD
    :param long: Day length of long term MACD
    :param signal: Day length of MACD signal
    :return: Dataframe of MACD values
    '''
    df = pd.DataFrame(df).set_index('index')
    symbol = df.columns[0]
    print('symbol',symbol)
    #ewm:pandas -지수이동평균 구하는 함수
    df['ema_short'] = df[symbol].ewm(span=short).mean()
    df['ema_long'] = df[symbol].ewm(span=long).mean()
    df['macd'] = (df['ema_short'] - df['ema_long']).round(2)
    df['macd_signal'] = df['macd'].ewm(span=signal).mean().round(2)
    df['macd_oscillator'] = (df['macd'] - df['macd_signal']).round(2)
    df=df[[symbol, 'macd','macd_signal','macd_oscillator']]
    print('macd df',df.tail(5))
    # return df[[symbol, 'macd','macd_signal','macd_oscillator']]
    return df.reset_index().to_dict(orient='records')

@router.post("/envelope")
def envelope(
    df: List[Dict[str, Any]], 
    w: int = 50, 
    spread: float = 0.05):
    '''
    Calculate Envelope indicators
    :param df: Dataframe containing historical prices
    :param w: Window size
    :param spread: % difference from center line to determine band width
    :return: Dataframe of Envelope values
    '''
    df = pd.DataFrame(df).set_index('index')
    symbol = df.columns[0]
    df['center'] = df[symbol].rolling(w).mean()
    df['ub'] = df['center'] * (1 + spread)
    df['lb'] = df['center'] * (1 - spread)
    df = df[[symbol, 'center', 'ub', 'lb']]
    
    # NaN 값을 처리 (NaN 값을 0으로 대체)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return df.reset_index().to_dict(orient='records')

@router.post("/bollinger")
async def bollinger(
    df: List[Dict[str, Any]],
    w: int = 20, 
    k: int = 2):
    '''
    Calculate bollinger band indicators
    :param df: Dataframe containing historical prices
    :param w: Window size
    :param k: Multiplier to determine band width
    :return: Dataframe of bollinger band values
    '''
    df = pd.DataFrame(df)
    df['index'] = pd.to_datetime(df['index'])
    df.set_index('index', inplace=True)
    
    symbol = df.columns[0]
    df['center'] = df[symbol].rolling(w).mean()
    df['sigma'] = df[symbol].rolling(w).std()
    df['ub'] = df['center'] + k * df['sigma']
    df['lb'] = df['center'] - k * df['sigma']
    
    # NaN 및 무한 값 처리
    df.fillna(0, inplace=True)
    df.replace([np.inf, -np.inf], 0, inplace=True)
    
    print(df[[symbol, 'center','ub','lb']].tail(5))
    
    return df.reset_index().to_dict(orient='records')

def stochastic(df, symbol, n=14, m=3, t=3):
    '''
    Calculate stochastic indicators
    :param df: Dataframe containing historical prices
    :param symbol: Symbol or ticker of equity by finance.yahoo.com
    :param n: Day length of fast k stochastic
    :param m: Day length of slow k stochastic
    :param t: Day length of slow d stochastic
    :return: Dataframe of stochastic values
    '''
    try:
        df['fast_k'] = ( ( df['Close'] - df['Low'].rolling(n).min() ) / ( df['High'].rolling(n).max() - df['Low'].rolling(n).min() ) ).round(4) * 100
        df['slow_k'] = df['fast_k'].rolling(m).mean().round(2)
        df['slow_d'] = df['slow_k'].rolling(t).mean().round(2)
        df.rename(columns={'Close':symbol}, inplace=True)
        df.drop(columns=['High','Open','Low','Volume','Adj Close','fast_k'], inplace=True)
        return df[[symbol, 'slow_k', 'slow_d']]
    except:
        return 'Error. The stochastic indicator requires OHLC data and symbol. Try get_ohlc() to retrieve price data.'

