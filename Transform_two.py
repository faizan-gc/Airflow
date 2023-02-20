import pandas as pd
import os
from datetime import datetime
from pandas import Timestamp
import pyarrow
import pickle
import numpy as np
from mlfinlab.microstructural_features import encoding, entropy
from mlfinlab.microstructural_features import second_generation
from mlfinlab.microstructural_features.misc import get_avg_tick_size, vwap



def feats(df, tic, i):
    if tic not in ['BTC', 'ETH']:
        df_=df.between_time('9:30', '15:59')
        if len(df_)!=0:
            tradeableH=df_.price.quantile(0.98)
            tradeableL=df_.price.quantile(0.02)
        else:
            tradeableH=np.nan
            tradeableL=np.nan
    else:
        tradeableH=df.price.quantile(0.98)
        tradeableL=df.price.quantile(0.02)
    df.reset_index(inplace=True, drop=True)
    df['dv']=df['price']*df['size']
    O=list(df.price)[0]
    H=df.price.quantile(0.98)
    L=df.price.quantile(0.02)
    C=list(df.price)[-1]
    V=df['size'].sum()
    DV=df['dv'].sum()
    BV=df[df.tick_rule==1]['size'].sum()
    BDV=df[df.tick_rule==1].dv.sum()
# sigma encoding & lempel_ziv & shanon entropy
    rets=df.tic_pct
    try:
        ed=encoding.sigma_mapping(rets, 0.001)
        ePt1Pct=encoding.encode_array(rets, ed)
        lempel_ziv_Pt1Pct=entropy.get_lempel_ziv_entropy(ePt1Pct)
        shanon_Pt1Pct=entropy.get_shannon_entropy(ePt1Pct)
        
    except:
        lempel_ziv_Pt1Pct=np.nan
        shanon_Pt1Pct=np.nan        
    try:
        ed=encoding.sigma_mapping(rets, 0.005)
        ePt5Pct=encoding.encode_array(rets, ed)
        
        lempel_ziv_Pt5Pct=entropy.get_lempel_ziv_entropy(ePt5Pct)
        shanon_Pt5Pct=entropy.get_shannon_entropy(ePt5Pct)
    except:
        lempel_ziv_Pt5Pct=np.nan
        shanon_Pt5Pct=np.nan        
    try:
        ed=encoding.sigma_mapping(rets, 0.01)
        e1Pct=encoding.encode_array(rets, ed)
        lempel_ziv_1Pct=entropy.get_lempel_ziv_entropy(e1Pct)
        shanon_1Pct=entropy.get_shannon_entropy(e1Pct)    
    except:
        lempel_ziv_1Pct=np.nan
        shanon_1Pct=np.nan    
# second gen microstructural features
    tb_kyle=second_generation.get_trades_based_kyle_lambda(df.tic_diff, df['size'], df.tick_rule)
    tb_kyle_coef=tb_kyle[0]
    tb_kyle_t=tb_kyle[1]
    tb_amihud=second_generation.get_trades_based_amihud_lambda(df.log_ret, df['dv'])
    tb_amihud_coef=tb_amihud[0]
    tb_amihud_t=tb_amihud[1]
    tb_hasbrouk=second_generation.get_trades_based_hasbrouck_lambda(df.log_ret, df['dv'], df.tick_rule)
    tb_hasbrouk_coef=tb_hasbrouk[0]
    tb_hasbrouk_t=tb_hasbrouk[1]
    _vwap=vwap(df.dv, df['size'])
    avg_tick_size=get_avg_tick_size(df['size'])
    return [O, H, L, C, V, DV, BV, BDV, tradeableH, tradeableL,
           lempel_ziv_Pt1Pct, shanon_Pt1Pct, lempel_ziv_Pt5Pct, shanon_Pt5Pct, lempel_ziv_1Pct, shanon_1Pct,
           tb_kyle_coef, tb_kyle_t, tb_amihud_coef, tb_amihud_t, tb_hasbrouk_coef, tb_hasbrouk_t,
           _vwap, avg_tick_size]



def calculate_features(ticker_info: dict,
                       return_format: str = 'csv'):

    TIC = ticker_info["name"]
    resample_path = ticker_info['resample-path'] 
    impute_path = ticker_info["impute-path"] 

    barDf=pd.read_feather(resample_path)
    dates=barDf.date_time
    timestamps_new=[]
    for count, currDate in enumerate(list(dates)):
        if count >0:
            timestamps_new.append((dates[count-1],currDate))           
    timestamps_new=pd.Series(timestamps_new)
    timestamps_new.drop_duplicates(keep='first', inplace=True)
    timestamps_new=list(timestamps_new)            
    print("Loading DataSet ")
    df = pd.read_feather(impute_path) ## change path read date and time
    df['participant_timestamp'] = pd.to_datetime(df['participant_timestamp'])
    df.set_index('participant_timestamp', inplace=True)
    df['tic_diff']=df.price.diff()
    df['tic_pct']=df.price.pct_change()
    df['price_shift']=df['price'].shift()
    df['log_ret']=np.log(df['price']/df['price_shift'])
    df.loc[(df.tic_diff>0), 'tick_rule']=1
    df.loc[(df.tic_diff<0), 'tick_rule']=-1   
    df['tick_rule'] = df['tick_rule'].ffill()
    
    print('tick rule done ')
    df.dropna(inplace=True)
#     df.sort_index(ascending=True, inplace=True)
    print('Dataset Loaded...')   
    data_store = {}
    count=1
    for i in (timestamps_new):
        x = df.loc[i[0]:i[1]].iloc[1:]
        x=feats(x, TIC, i)
        data_store[i[1]] = x
        count+=1
    if return_format.lower() == 'csv':
        return pd.DataFrame.from_dict(data_store, orient='index')
    else:
        return data_store


def transform_stage_two(ticker_info):
    tic = ticker_info["name"]
    ticdict=dict()
    fdf=calculate_features(ticker_info)
    print("THE LENGTH OF FDF IS ", len(fdf))
    print("THE FDF IS", fdf.head())
    ticdict[tic]=fdf
    ticdict[tic].rename(columns={ticdict[tic].columns[0]: 'O', 
                                ticdict[tic].columns[1]: 'H',
                                ticdict[tic].columns[2]: 'L',
                                ticdict[tic].columns[3]: 'C',
                                ticdict[tic].columns[4]: 'V', 
                                ticdict[tic].columns[5]: 'DV',
                                ticdict[tic].columns[6]: 'BV', 
                                ticdict[tic].columns[7]: 'BDV',
                                ticdict[tic].columns[8]: 'tradeableH',
                                ticdict[tic].columns[9]: 'tradeableL',
                                
                                ticdict[tic].columns[10]: 'lempel_ziv_Pt1Pct', 
                                ticdict[tic].columns[11]: 'shanon_Pt1Pct',
                                ticdict[tic].columns[12]: 'lempel_ziv_Pt5Pct',
                                ticdict[tic].columns[13]: 'shanon_Pt5Pct',
                                ticdict[tic].columns[14]: 'lempel_ziv_1Pct',
                                ticdict[tic].columns[15]: 'shanon_1Pct',
                                
                                ticdict[tic].columns[16]: 'tb_kyle_coef', 
                                ticdict[tic].columns[17]: 'tb_kyle_t',
                                ticdict[tic].columns[18]: 'tb_amihud_coef',
                                ticdict[tic].columns[19]: 'tb_amihud_t',
                                ticdict[tic].columns[20]: 'tb_hasbrouk_coef',
                                ticdict[tic].columns[21]: 'tb_hasbrouk_t',
                                
                                ticdict[tic].columns[22]: 'vwap',
                                ticdict[tic].columns[23]: 'avg_tick_size',

                                }, inplace=True)
    pickle.dump(ticdict[tic], open(f'docker_storage/Tick_Data/Const_Resampled_2017/dolDF_const_10_BarsPerDay_wFeats/{tic}_dolDF_const_10_BarsPerDay_wFeats.pkl', 'wb'))    
    return tic


def update_transform_stage_two(complete_ticker_info):
    new_lst = []
    for ticker in complete_ticker_info.keys():
        tic = transform_stage_two(complete_ticker_info[ticker])
        new_lst.append(tic)
    print("NEW lst", new_lst)
    return new_lst