from datetime import date
from typing import Any, Dict, List, Optional, Union

from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from . import data_prep,trend
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FixedLocator
from config import settings
from fastapi import APIRouter, Depends, File, Query, UploadFile, HTTPException
import matplotlib.dates as mdates
from io import BytesIO

router = APIRouter()

ScalarFormatter().set_scientific(False)
plt.style.use('bmh')
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.grid'] = True
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.7
plt.rcParams['lines.antialiased'] = True
plt.rcParams['figure.figsize'] = [10.0, 5.0]
plt.rcParams['savefig.dpi'] = 96
plt.rcParams['font.size'] = 12
plt.rcParams['legend.fontsize'] = 'medium'
plt.rcParams['figure.titlesize'] = 'medium'
plt.rcParams['axes.formatter.useoffset'] = True
plt.rcParams['axes.formatter.use_mathtext'] = True

def str_to_list(s: Union[str, List[str]]) -> List[str]:
    if isinstance(s, str):
        return [s]
    return s
    
@router.post("/draw_chart")
async def draw_chart(
    df: List[Dict[str, Any]], 
    left: Optional[Union[str, List[str]]] = Query(None), 
    right: Optional[Union[str, List[str]]] = Query(None)
):
    log = False
    df = pd.DataFrame(df)
    df['index'] = pd.to_datetime(df['index'])
    df.set_index('index', inplace=True)
    
    fig, ax1 = plt.subplots()
    x = df.index

    if left is not None:
        left = str_to_list(left)
        i = 6
        for c in left:
            ax1.plot(x, df[c], label=c, color='C'+str(i), alpha=1)
            i += 1
        if log:
            ax1.set_yscale('log')
            ax1.yaxis.set_major_formatter(ScalarFormatter())
            ax1.yaxis.set_minor_formatter(ScalarFormatter())
    else:
        ax1.axes.yaxis.set_visible(False)

    if right is not None:
        right = str_to_list(right)
        ax2 = ax1.twinx()
        i = 1
        for c in right:
            ax2.plot(x, df[c], label=c+'(R)', color='C'+str(i), alpha=1)
            ax1.plot(np.nan, label=c+'(R)', color='C'+str(i))
            i += 1
        ax1.grid(False, axis='y')
        if log:
            ax2.set_yscale('log')
            ax2.yaxis.set_major_formatter(ScalarFormatter())
            ax2.yaxis.set_minor_formatter(ScalarFormatter())

    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()
    
    ax1.legend(loc=2)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return StreamingResponse(buf, media_type="image/png")
    
# @router.post("/draw_chart")
# async def draw_chart(
#     df: List[Dict[str, Any]], 
#     left: Optional[str] = Query(None), 
#     right: Optional[str] = Query(None)
# ):
#     print('right',right,left)
#     log = False
#     df = pd.DataFrame(df)
#     df['index'] = pd.to_datetime(df['index'])
#     df.set_index('index', inplace=True)
    
#     fig, ax1 = plt.subplots()
#     x = df.index

#     if left is not None:
#         left = str_to_list(left)
#         i = 6
#         for c in left:
#             ax1.plot(x, df[c], label=c, color='C'+str(i), alpha=1)
#             i += 1
#         if log:
#             ax1.set_yscale('log')
#             ax1.yaxis.set_major_formatter(ScalarFormatter())
#             ax1.yaxis.set_minor_formatter(ScalarFormatter())
#     else:
#         ax1.axes.yaxis.set_visible(False)

#     if right is not None:
#         right = str_to_list(right)
#         print('right',right)
#         ax2 = ax1.twinx()
#         i = 1
#         for c in right:
#             ax2.plot(x, df[c], label=c+'(R)', color='C'+str(i), alpha=1)
#             ax1.plot(np.nan, label=c+'(R)', color='C'+str(i))
#             i += 1
#         ax1.grid(False, axis='y')
#         if log:
#             ax2.set_yscale('log')
#             ax2.yaxis.set_major_formatter(ScalarFormatter())
#             ax2.yaxis.set_minor_formatter(ScalarFormatter())

#     ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
#     ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
#     fig.autofmt_xdate()
    
#     ax1.legend(loc=2)
#     plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

#     buf = BytesIO()
#     plt.savefig(buf, format='png')
#     buf.seek(0)
#     plt.close(fig)
#     return StreamingResponse(buf, media_type="image/png")

@router.post("/draw_band_chart")
def draw_band_chart(
    df: List[Dict[str, Any]], 
    band=['lb', 'center', 'ub'], 
    log=False):
    '''
    :param df: Dataframe that contains data to plot
    :param band: List of columns to be plotted as [lower band, center line, upper band]
    :param log: Plot in log scale
    :return: Band chart
    '''
    df = pd.DataFrame(df)
    df['index'] = pd.to_datetime(df['index'])
    df.set_index('index', inplace=True)
    
    fig, ax1 = plt.subplots()
    x = df.index
    ax1.axes.yaxis.set_visible(False)
    
    # secondary y
    ax2 = ax1.twinx()
    
    ax2.fill_between(x, df[band[0]], df[band[2]], color='C0', alpha=.2)
    ax2.plot(x, df[band[1]], label=band[1], color='C0', alpha=.7)
    
    symbol = df.columns[0]
    ax2.plot(x, df[symbol], label=symbol, color='C1', alpha=1)
    
    ax1.grid(False, axis='y')
    
    if log:
        ax2.set_yscale('log')
        ax2.yaxis.set_major_formatter(ScalarFormatter())
        ax2.yaxis.set_minor_formatter(ScalarFormatter())
    
    ax2.legend(loc=2)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    return StreamingResponse(buf, media_type="image/png")

@router.post("/draw_trade_results")
async def draw_trade_results(
    df: List[Dict[str, Any]],
):
    '''
    Draw portfolio return and position changes
    :param df: Dataframe that contains data to plot
    :return: Portfolio return and position chart
    '''
    df = pd.DataFrame(df)
    df['index'] = pd.to_datetime(df['index'])
    df.set_index('index', inplace=True)
    
    fig, ax1 = plt.subplots()
    x = df.index
    
    # Plot accumulated return on primary y-axis
    ax1.plot(x, df['acc_rtn_dp'], label='Return', color='C6', alpha=.7)
    ax1.grid(False, axis='y')
    
    # Plot the first column on secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot(x, df.iloc[:, 0], label=df.columns[0], color='C1', alpha=1)
    ax1.plot(np.nan, label=df.columns[0] + '(R)', color='C1')
    
    # Plot position_chart on tertiary y-axis
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))  # Move the third y-axis to the right
    ax3.fill_between(x, 0, df['position_chart'], color='C2', alpha=.5)
    ax3.set_ylim(0, 10)
    ax3.axes.yaxis.set_visible(False)
    
    ax1.legend(loc=2)
    
    # Format x-axis
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    
    return StreamingResponse(buf, media_type="image/png")

    
def draw_price_multiple_band(df, multiple='PER', acct='EPS', log=False):
    '''
    Draw price multiple band chart
    :param df: Dataframe that contains data to plot
    :param multiple: Price multiple
    :param acct: Financial account to be used to calculate price multiple
    :param log: Plot in log scale
    :return: Price multiple band chart
    '''
    fig, ax1 = plt.subplots()
    x = df.index
    i_max = round((df['Price']/df[acct]).max(),1)
    i_min = round((df['Price']/df[acct]).min(),1)
    i_3 = round(i_min+(i_max-i_min)/4*3,1)
    i_2 = round(i_min+(i_max-i_min)/2,1)
    i_1 = round(i_min+(i_max-i_min)/4,1)
    ax1.plot(x, i_max*df[acct], label=multiple+str(i_max), color='C2', linewidth=1, alpha=.7)
    ax1.plot(x, i_3*df[acct], label=multiple+str(i_3), color='C3', linewidth=1, alpha=.7)
    ax1.plot(x, i_2*df[acct], label=multiple+str(i_2), color='C4', linewidth=1, alpha=.7)
    ax1.plot(x, i_1*df[acct], label=multiple+str(i_1), color='C5', linewidth=1, alpha=.7)
    ax1.plot(x, i_min*df[acct], label=multiple+str(i_min), color='C6', linewidth=1, alpha=.7)
    ax1.plot(x, df['Price'], label='Price', color='C1', alpha=1)

    if log:
        ax1.set_yscale('log')
        ax1.yaxis.set_major_formatter(ScalarFormatter())
        ax1.yaxis.set_minor_formatter(ScalarFormatter())

    ax1.legend(loc=2)


def draw_return(df, bm='^GSPC'):
    '''
    Draw portfolio return with benchmark return
    :param df: Dataframe that contains data to plot
    :param bm: Symbol of benchmark to be plotted together
    :return: Portfolio return chart
    '''
    end = (pd.to_datetime(df.index[-1]) + pd.tseries.offsets.QuarterEnd(0)).date()
    bm_ = data_prep.get_price(symbol=bm, start_date='2006-01-01', end_date=end)
    month_ends = bm_.loc[bm_.groupby([bm_.index.year, bm_.index.month]).apply(lambda s: np.max(s.index))]
    quarter_ends = bm_.loc[bm_.groupby([bm_.index.year, bm_.index.quarter]).apply(lambda s: np.max(s.index))]
    bm_idx = bm_.head(1)
    bm_idx = bm_idx.append(quarter_ends)
    bm_idx['term_rtn'] = bm_idx.pct_change() + 1
    bm_idx = bm_idx.loc[df.index[0]:df.index[-1]]
    bm_idx['acc_rtn'] = bm_idx[bm] / bm_idx[bm][0]

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    x = df.index
    x_ = np.arange(len(df))
    ax1.bar(x_-.2, (df['term_rtn']-1)*100, width=.4, label='Portfolio Rtn (term)', color='C5', linewidth=1, alpha=.5)
    ax1.bar(x_+.2, (bm_idx['term_rtn']-1)*100, width=.4, label='BM Rtn (term)', color='C6', linewidth=1, alpha=.5)
    ax2.plot(x_, (df['acc_rtn']-1)*100, label='Portfolio Rtn', color='C1', alpha=1)
    ax2.plot(x_, (bm_idx['acc_rtn']-1)*100, label='BM Rtn', color='C0', alpha=1, linestyle='dotted')
    ax1.plot(np.nan, label='Portfolio Rtn (R)', color='C1', alpha=1)
    ax1.plot(np.nan, label='BM Rtn (R)', color='C0', alpha=1, linestyle='dotted')
    ax1.axhline(0, linestyle='dotted', linewidth=1, color='k')
    ax1.legend(loc=2)
    ax1.grid(False)
    ax1.set_ylim(-100,100)
    plt.xticks(x_, x)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=90)


def str_to_list(s):
    '''
    Convert string to list
    :param s: String or List
    :return: List
    '''
    if type(s) == list:
        cds = s
    else:
        cds = []
        cds.append(s)
    return cds
