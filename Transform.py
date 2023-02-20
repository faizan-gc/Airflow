import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
from mlfinlab.data_structures import imbalance_data_structures, standard_data_structures
from datetime import datetime
from dateutil.relativedelta import relativedelta, MO
import gc
import talib


def step_one(ticker_info):
    tic = ticker_info["name"]
    path = ticker_info["path"]
    start_date = ticker_info["start-date"]
    end_date = ticker_info["end-date"]

    if tic not in ['BTC', 'ETH']:
        print(tic)
        raw_merged_feather = pd.read_feather(path)
        raw_merged_feather.reset_index(inplace=True, drop=True)
        #temp=raw_merged_feather[raw_merged_feather.participant_timestamp<'2017-05-01'].copy()               #UNCOMMENT WHEN ON SERVER!
        temp = raw_merged_feather.copy()        
        temp=temp[['participant_timestamp', 'price', 'size']]
        temp.dropna(inplace=True)
        temp['dv']=temp['price']*temp['size']
        volthresh=temp.groupby([temp['participant_timestamp'].dt.date]).sum().dv.describe(percentiles=[0.6])['60%']
#-------------------------------------------------------------------------
        temp=raw_merged_feather[raw_merged_feather.participant_timestamp>'2017-01-01'].copy()        
        temp['sma_price']=talib.SMA(temp['price'], 10)
        temp['sma_price_pct']=abs(((temp['price']-temp['sma_price'])/temp['sma_price'])*100)
        temp=temp[temp['sma_price_pct']<7.5].reset_index(drop=True)
        temp=temp[['participant_timestamp', 'price', 'size']]
        temp.dropna(inplace=True)
        temp=temp[temp.participant_timestamp>'2017-05-01'].copy()
        temp.reset_index(drop=True, inplace=True)
        temp.to_feather(f'./docker_storage/Tick_Data/AdjustedData/2k17OnAndImputed/{tic}-|{start_date}_{end_date}|-Tick-Data.ftr')
        
        gc.collect()
        for barsperday in [10]:
            t=(volthresh/barsperday)
            dolDF=standard_data_structures.get_dollar_bars(temp, threshold=t, verbose=False, batch_size=100000)
            dolDF.to_feather(f'./docker_storage/Tick_Data/Const_Resampled_2017/{tic}_dolDF_const_{barsperday}_BarsPerDay.ftr')
            gc.collect()
        gc.collect()
    else:
        print(tic)
        raw_merged_feather = pd.read_feather(path)
        raw_merged_feather.reset_index(inplace=True, drop=True)
        temp=raw_merged_feather[raw_merged_feather.participant_timestamp<'2018-05-01'].copy()
        temp=temp[['participant_timestamp', 'price', 'size']]
        temp.dropna(inplace=True)
        temp['dv']=temp['price']*temp['size']
        volthresh=temp.groupby([temp['participant_timestamp'].dt.date]).sum().dv.describe(percentiles=[0.6])['60%']
#-------------------------------------------------------------------------
        temp=raw_merged_feather[raw_merged_feather.participant_timestamp>'2018-01-01'].copy()        
        temp['sma_price']=talib.SMA(temp['price'], 10)
        temp['sma_price_pct']=abs(((temp['price']-temp['sma_price'])/temp['sma_price'])*100)
        temp=temp[temp['sma_price_pct']<7.5].reset_index(drop=True)
        temp=temp[['participant_timestamp', 'price', 'size']]
        temp.dropna(inplace=True)
        temp=temp[temp.participant_timestamp>'2018-05-01'].copy()
        temp.reset_index(drop=True, inplace=True)
        temp.to_feather(f'./docker_storage/Tick_Data/AdjustedData/2k17OnAndImputed/{tic}-{start_date}_{end_date}-Tick-Data.ftr')

        gc.collect()
        for barsperday in [10]:
            t=(volthresh/(barsperday*3.69))
            dolDF=standard_data_structures.get_dollar_bars(temp, threshold=t, verbose=False, batch_size=100000)
            dolDF.to_feather(f'/docker_storage/Tick_Data/Const_Resampled_2017/{tic}_dolDF_const_{barsperday}_BarsPerDay.ftr')
            gc.collect()
        gc.collect()
    ticker_info['resample-path'] = f"./docker_storage/Tick_Data/Const_Resampled_2017/{tic}_dolDF_const_{barsperday}_BarsPerDay.ftr"
    ticker_info["impute-path"] = f"./docker_storage/Tick_Data/AdjustedData/2k17OnAndImputed/{tic}-|{start_date}_{end_date}|-Tick-Data.ftr"
    return ticker_info