import numpy as np
import pandas as pd
from mlfinlab.data_structures import imbalance_data_structures, standard_data_structures
from pyarrow import feather
import pickle
from datetime import datetime
from dateutil.relativedelta import relativedelta, MO
from datetime import timedelta
# import exchange_calendars as xcals
# from statsmodels.tsa.stattools import adfuller
from scipy.stats import iqr
import warnings
import talib
from mlfinlab.features.fracdiff import frac_diff_ffd
from mlfinlab.microstructural_features import first_generation
from mlfinlab.filters import filters
# from pandarallel import pandarallel
# pandarallel.initialize(progress_bar=False, nb_workers=32)  #Uncomment on server




def getDailyVol(close,span0=100):
    # daily vol reindexed to close
    df0=close.index.searchsorted(close.index-pd.Timedelta(days=1))
    df0=df0[df0>0]   
    df0=(pd.Series(close.index[df0-1], 
                   index=close.index[close.shape[0]-df0.shape[0]:]))   
    try:
        df0=close.loc[df0.index]/close.loc[df0.values].values-1 # daily rets
    except Exception as e:
        print(f'error: {e}\nplease confirm no duplicate indices')
    df0=df0.ewm(span=span0).std().rename('dailyVol')
    return df0


def co(curr, prev):
    if prev<1 and curr>1:
        return 1
    else:
        return 0


def TradeSim(row, count, slM, tpM):
    bp=row['C']
    dailyVol=row['feature_dailyVol']
    
    sl=-slM*dailyVol
    tp=tpM*dailyVol
    
    for i in np.arange(count):
               
        currLowPct=(row[f'low+{i+1}']-bp)/bp
        currHighPct=(row[f'high+{i+1}']-bp)/bp
        
        currBarTime=row[f'index+{i+1}']
        
        
        if currLowPct<sl:
            return [currLowPct, -1, i, currBarTime]
        
        
        elif currHighPct>=tp:
            return [currHighPct, 1, i, currBarTime]
        
    currClosePct=(row[f'close+{count}']-bp)/bp
    return [currClosePct, 0, count, currBarTime]



def make_all_features(tic):
  secDict=dict()
  try:
      print('------------------------------------------------------------------')
      print(tic)
      df=pickle.load(open(f'docker_storage/Tick_Data/Const_Resampled_2017/dolDF_const_10_BarsPerDay_wFeats/{tic}_dolDF_const_10_BarsPerDay_wFeats.pkl', 'rb'))
      secDict[tic]=df
      print(len(df))
      print(secDict[tic].dropna(subset=['tradeableH', 'tradeableL']).isna().sum().sum())

      #     Make Features


      #     ffd ohlc, vwap
      # for col in ['O', 'H', 'L', 'C']:
      #     secDict[tic][f'feature_ffd_log_{col}']=frac_diff_ffd(np.log(secDict[tic][[col]]), 0.6, thresh=1e-5)

      # secDict[tic][f'feature_ffd_log_vwap']=frac_diff_ffd(np.log(secDict[tic][['vwap']]), 0.6, thresh=1e-5)

      # avg tick size adjusted w price
      secDict[tic]['feature_avg_tick_price*C']=secDict[tic]['avg_tick_size']*secDict[tic]['C']


      # ratios of ohlcv, wvap
      secDict[tic]['feature_O/C']=secDict[tic]['O']/secDict[tic]['C']
      secDict[tic]['feature_O/H']=secDict[tic]['O']/secDict[tic]['H']
      secDict[tic]['feature_O/L']=secDict[tic]['O']/secDict[tic]['L']
      secDict[tic]['feature_H/L']=secDict[tic]['H']/secDict[tic]['L']
      secDict[tic]['feature_wvap/C']=secDict[tic]['vwap']/secDict[tic]['C']



      #    BBs
      ub, middleband, lb = talib.BBANDS(secDict[tic].C, timeperiod=21, nbdevup=2, nbdevdn=2, matype=0)
      # secDict[tic][f'feature_UB']=ub
      # secDict[tic][f'feature_LB']=lb
      secDict[tic]['feature_C/UB']=secDict[tic]['C']/ub
      secDict[tic]['feature_C/LB']=secDict[tic]['C']/lb
      secDict[tic]['feature_UB/LB']=ub/lb




      # emas
      # secDict[tic][f'feature_EMA10']=talib.EMA(secDict[tic].C, 10)
      # secDict[tic][f'feature_EMA21']=talib.EMA(secDict[tic].C, 21)
      # secDict[tic][f'feature_EMA55']=talib.EMA(secDict[tic].C, 55)
      ema5=talib.EMA(secDict[tic].C, 5)
      ema10=talib.EMA(secDict[tic].C, 10)
      ema21=talib.EMA(secDict[tic].C, 21)
      ema55=talib.EMA(secDict[tic].C, 55)
      sma500=talib.SMA(secDict[tic].C, 500)
      secDict[tic]['feature_ema10/ema21']=ema10/ema21
      secDict[tic]['feature_ema10/ema55']=ema10/ema55
      secDict[tic]['feature_ema21/ema55']=ema21/ema55
      secDict[tic]['feature_C/ema10']=secDict[tic]['C']/ema10
      secDict[tic]['feature_C/ema21']=secDict[tic]['C']/ema21
      secDict[tic]['feature_C/ema55']=secDict[tic]['C']/ema55



      #     adx
      secDict[tic][f'feature_adx21']=talib.ADX(secDict[tic].H, secDict[tic].L, secDict[tic].C, 21)
      secDict[tic][f'feature_adx63']=talib.ADX(secDict[tic].H, secDict[tic].L, secDict[tic].C, 63)

      # mfi
      secDict[tic][f'feature_mfi14']=talib.MFI(secDict[tic].H, secDict[tic].L, secDict[tic].C, secDict[tic].DV, 14)
      secDict[tic][f'feature_mfi63']=talib.MFI(secDict[tic].H, secDict[tic].L, secDict[tic].C, secDict[tic].DV, 63)

      # macd
      macd, macdsignal, macdhist = talib.MACD(secDict[tic].C, fastperiod=12, slowperiod=26, signalperiod=9)
      secDict[tic][f'feature_macd']=macd/sma500
      secDict[tic][f'feature_macdsignal']=macdsignal/sma500
      secDict[tic][f'feature_macdhist']=macdhist/sma500

      # rsi
      secDict[tic][f'feature_rsi14']=talib.RSI(secDict[tic].C, 14)
      secDict[tic][f'feature_rsi63']=talib.RSI(secDict[tic].C, 63)

      # stochrsi
      fastk, fastd = talib.STOCHRSI(secDict[tic].C, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
      secDict[tic][f'feature_fastk']=fastk
      secDict[tic][f'feature_fastd']=fastd

      # UO
      secDict[tic][f'feature_UO']=talib.ULTOSC(secDict[tic].H, secDict[tic].L, secDict[tic].C, timeperiod1=7, timeperiod2=14, timeperiod3=28)

      # atr
      secDict[tic][f'feature_atr14']=talib.ATR(secDict[tic].H, secDict[tic].L, secDict[tic].C, timeperiod=14)/sma500
      secDict[tic][f'feature_atr63']=talib.ATR(secDict[tic].H, secDict[tic].L, secDict[tic].C, timeperiod=63)/sma500


      #tsf
      secDict[tic]['feature_tsf14']=talib.TSF(secDict[tic].C, 14)/secDict[tic].C
      secDict[tic]['feature_tsf63']=talib.TSF(secDict[tic].C, 63)/secDict[tic].C


      # # cycle inds
      # secDict[tic][f'feature_HT_DCPERIOD']=talib.HT_DCPERIOD(secDict[tic].C)
      # secDict[tic][f'feature_HT_DCPHASE']=talib.HT_DCPHASE(secDict[tic].C)
      # inphase, quadrature=talib.HT_PHASOR(secDict[tic].C)
      # secDict[tic][f'feature_HT_PHASOR_IN']=inphase
      # secDict[tic][f'feature_HT_PHASOR_QUAD']=quadrature
      # sine, leadsine=talib.HT_SINE(secDict[tic].C)
      # secDict[tic][f'feature_HT_SINE']=sine
      # secDict[tic][f'feature_HT_LEADSINE']=leadsine
      # secDict[tic][f'feature_HT_TRENDMODE']=talib.HT_TRENDMODE(secDict[tic].C)



      for i in secDict[tic].columns:
          if 'feature_' in i:
      #         secDict[tic][f'{i}_pct']=secDict[tic][i].pct_change()
              secDict[tic][f'{i}_angle5']=talib.LINEARREG_ANGLE(secDict[tic][i], 5)
      #         secDict[tic][f'{i}_slope10']=talib.LINEARREG_SLOPE(secDict[tic][i], 14)


      for i in secDict[tic].columns:
          if 'feature_' in i:        
              secDict[tic][f'{i}_var14']=talib.VAR(secDict[tic][i], 14)
      #         secDict[tic][f'{i}_square']=secDict[tic][i]**2
      #         secDict[tic][f'{i}_sin']=talib.SIN(secDict[tic][i])
      
      
      
      
      



      #     for i in [1,2,4,8,16,32,64, 128]:
      #         for col in ['feature_ffd_log_O','feature_ffd_log_H','feature_ffd_log_L','feature_ffd_log_C']:
      #             secDict[tic][f'{col}_lag{i}']=secDict[tic][f'{col}'].shift(i)

      #     DailyVol for dynamic tgts

      daily_vol = getDailyVol(secDict[tic]['C'])
      secDict[tic]=pd.merge(secDict[tic], daily_vol, left_index=True, right_index=True, how='left')
      secDict[tic].rename(columns={'dailyVol': 'feature_dailyVol'}, inplace=True)


      #     event sampling


      events=filters.cusum_filter(secDict[tic].C, secDict[tic].feature_dailyVol.median()*0.75)
      secDict[tic].loc[events, 'event']=1
      secDict[tic]['event']=secDict[tic]['event'].fillna(0)



      #     secDict[tic]['feature_sma5']=talib.SMA(secDict[tic]['C'], 5)
      #     secDict[tic]['feature_sma10']=talib.SMA(secDict[tic]['C'], 10)

      #     secDict[tic]['feature_sma_co']=secDict[tic]['feature_sma5']/secDict[tic]['feature_sma10']
      #     secDict[tic]['feature_sma_co_prev']=secDict[tic]['feature_sma_co'].shift(+1)

      #     secDict[tic]['feature_1_0_for_co']=secDict[tic][['feature_sma_co', 'feature_sma_co_prev']].apply(lambda x: co(*x), axis=1)


      #     get only tradeable bars

      if tic not in ['BTC', 'ETH']:
          secDict[tic]=secDict[tic].between_time('9:30', '15:59')



      cols=secDict[tic].columns
      cols=[s for s in cols if 'feature_' in s]

      secDict[tic]=secDict[tic].dropna(subset=cols)

      print(secDict[tic][cols].isna().sum().sum())



      #     ffd feats
      # secDict[tic]['feature_ffd_log_UB']=frac_diff_ffd(np.log(secDict[tic][['feature_UB']]), 0.6, thresh=1e-5)
      # secDict[tic]['feature_ffd_log_LB']=frac_diff_ffd(np.log(secDict[tic][['feature_LB']]), 0.6, thresh=1e-5)


      # secDict[tic]['feature_ffd_log_EMA10']=frac_diff_ffd(np.log(secDict[tic][['feature_EMA10']]), 0.6, thresh=1e-5)
      # secDict[tic]['feature_ffd_log_EMA21']=frac_diff_ffd(np.log(secDict[tic][['feature_EMA21']]), 0.6, thresh=1e-5)
      # secDict[tic]['feature_ffd_log_EMA55']=frac_diff_ffd(np.log(secDict[tic][['feature_EMA55']]), 0.6, thresh=1e-5)

      # secDict[tic]['feature_ffd_macd']=frac_diff_ffd(secDict[tic][['feature_macd']], 0.6, thresh=1e-5)
      # secDict[tic]['feature_ffd_macdsignal']=frac_diff_ffd(secDict[tic][['feature_macdsignal']], 0.6, thresh=1e-5)
      # secDict[tic]['feature_ffd_macdhist']=frac_diff_ffd(secDict[tic][['feature_macdhist']], 0.6, thresh=1e-5)

      # secDict[tic].drop(columns=['feature_UB', 'feature_LB', 'feature_EMA10', 'feature_EMA21', 'feature_EMA55', 'feature_macd', 'feature_macdsignal', 'feature_macdhist'], axis=1, inplace=True)


      cols=secDict[tic].columns
      cols=[s for s in cols if 'ffd_' in s]

      secDict[tic]=secDict[tic].dropna(subset=cols)

      print(secDict[tic][cols].isna().sum().sum())

      print(len(secDict[tic]))

  #     make tgts

      secDict[tic]['index_curr']=secDict[tic].index
      for i in np.arange(400):
          secDict[tic][f'high+{i+1}']=secDict[tic]['tradeableH'].shift(-(i+1))
          secDict[tic][f'low+{i+1}']=secDict[tic]['tradeableL'].shift(-(i+1))

          secDict[tic][f'index+{i+1}']=secDict[tic]['index_curr'].shift(-(i+1))


      secDict[tic]['close+400']=secDict[tic]['C'].shift(-(400))
      
      print(secDict[tic].isna().sum())
      secDict[tic].dropna(inplace=True)

      print(secDict[tic].isna().sum().sum())

  #     400_2_1

      t=secDict[tic].parallel_apply(lambda x: TradeSim(x, 400, 2, 1), axis=1)
      t=pd.DataFrame(list(t))
      secDict[tic]['pctAtBarrier_400_2sl_1tp']=list(t.iloc[:, 0])
      secDict[tic]['sideOfBarrier_400_2sl_1tp']=list(t.iloc[:, 1])
      secDict[tic]['nbars_400_2sl_1tp']=list(t.iloc[:, 2])
      secDict[tic]['barTimeOfExit_400_2sl_1tp']=list(t.iloc[:, 3])

      print(secDict[tic].isna().sum().sum())


      secDict[tic].loc[secDict[tic].sideOfBarrier_400_2sl_1tp==1, 'target_400_2sl_1tp_minRetPt5Pct']=1
      secDict[tic].loc[secDict[tic].sideOfBarrier_400_2sl_1tp==-1, 'target_400_2sl_1tp_minRetPt5Pct']=0
      secDict[tic].loc[((secDict[tic].sideOfBarrier_400_2sl_1tp==0) & (secDict[tic].pctAtBarrier_400_2sl_1tp>0.005)), 'target_400_2sl_1tp_minRetPt5Pct']=1
      secDict[tic].loc[((secDict[tic].sideOfBarrier_400_2sl_1tp==0) & (secDict[tic].pctAtBarrier_400_2sl_1tp<0.005)), 'target_400_2sl_1tp_minRetPt5Pct']=0


      #     400_1_1

      t=secDict[tic].parallel_apply(lambda x: TradeSim(x, 400, 1, 1), axis=1)
      t=pd.DataFrame(list(t))
      secDict[tic]['pctAtBarrier_400_1sl_1tp']=list(t.iloc[:, 0])
      secDict[tic]['sideOfBarrier_400_1sl_1tp']=list(t.iloc[:, 1])
      secDict[tic]['nbars_400_1sl_1tp']=list(t.iloc[:, 2])
      secDict[tic]['barTimeOfExit_400_1sl_1tp']=list(t.iloc[:, 3])

      print(secDict[tic].isna().sum().sum())


      secDict[tic].loc[secDict[tic].sideOfBarrier_400_1sl_1tp==1, 'target_400_1sl_1tp_minRetPt5Pct']=1
      secDict[tic].loc[secDict[tic].sideOfBarrier_400_1sl_1tp==-1, 'target_400_1sl_1tp_minRetPt5Pct']=0
      secDict[tic].loc[((secDict[tic].sideOfBarrier_400_1sl_1tp==0) & (secDict[tic].pctAtBarrier_400_1sl_1tp>0.005)), 'target_400_1sl_1tp_minRetPt5Pct']=1
      secDict[tic].loc[((secDict[tic].sideOfBarrier_400_1sl_1tp==0) & (secDict[tic].pctAtBarrier_400_1sl_1tp<0.005)), 'target_400_1sl_1tp_minRetPt5Pct']=0

      #     400_1_2


      t=secDict[tic].parallel_apply(lambda x: TradeSim(x, 400, 1, 2), axis=1)
      t=pd.DataFrame(list(t))
      secDict[tic]['pctAtBarrier_400_1sl_2tp']=list(t.iloc[:, 0])
      secDict[tic]['sideOfBarrier_400_1sl_2tp']=list(t.iloc[:, 1])
      secDict[tic]['nbars_400_1sl_2tp']=list(t.iloc[:, 2])
      secDict[tic]['barTimeOfExit_400_1sl_2tp']=list(t.iloc[:, 3])

      print(secDict[tic].isna().sum().sum())


      secDict[tic].loc[secDict[tic].sideOfBarrier_400_1sl_2tp==1, 'target_400_1sl_2tp_minRetPt5Pct']=1
      secDict[tic].loc[secDict[tic].sideOfBarrier_400_1sl_2tp==-1, 'target_400_1sl_2tp_minRetPt5Pct']=0
      secDict[tic].loc[((secDict[tic].sideOfBarrier_400_1sl_2tp==0) & (secDict[tic].pctAtBarrier_400_1sl_2tp>0.005)), 'target_400_1sl_2tp_minRetPt5Pct']=1
      secDict[tic].loc[((secDict[tic].sideOfBarrier_400_1sl_2tp==0) & (secDict[tic].pctAtBarrier_400_1sl_2tp<0.005)), 'target_400_1sl_2tp_minRetPt5Pct']=0



      #     400_0.5_2


      t=secDict[tic].parallel_apply(lambda x: TradeSim(x, 400, 0.5, 2), axis=1)
      t=pd.DataFrame(list(t))
      secDict[tic]['pctAtBarrier_400_0pt5sl_2tp']=list(t.iloc[:, 0])
      secDict[tic]['sideOfBarrier_400_0pt5sl_2tp']=list(t.iloc[:, 1])
      secDict[tic]['nbars_400_0pt5sl_2tp']=list(t.iloc[:, 2])
      secDict[tic]['barTimeOfExit_400_0pt5sl_2tp']=list(t.iloc[:, 3])

      print(secDict[tic].isna().sum().sum())


      secDict[tic].loc[secDict[tic].sideOfBarrier_400_0pt5sl_2tp==1, 'target_400_0pt5sl_2tp_minRetPt5Pct']=1
      secDict[tic].loc[secDict[tic].sideOfBarrier_400_0pt5sl_2tp==-1, 'target_400_0pt5sl_2tp_minRetPt5Pct']=0
      secDict[tic].loc[((secDict[tic].sideOfBarrier_400_0pt5sl_2tp==0) & (secDict[tic].pctAtBarrier_400_0pt5sl_2tp>0.005)), 'target_400_0pt5sl_2tp_minRetPt5Pct']=1
      secDict[tic].loc[((secDict[tic].sideOfBarrier_400_0pt5sl_2tp==0) & (secDict[tic].pctAtBarrier_400_0pt5sl_2tp<0.005)), 'target_400_0pt5sl_2tp_minRetPt5Pct']=0

      print(len(secDict[tic]))
      
      cols=secDict[tic].columns
      cols=[s for s in cols if 'feature_' in s]
      secDict[tic]=secDict[tic][['O', 'H', 'L', 'C', 'tradeableH', 'tradeableL',
                                'lempel_ziv_Pt1Pct', 'shanon_Pt1Pct', 'lempel_ziv_Pt5Pct',
                                  'shanon_Pt5Pct', 'lempel_ziv_1Pct', 'shanon_1Pct', 'tb_kyle_coef',
                                  'tb_kyle_t', 'tb_amihud_coef', 'tb_amihud_t', 'tb_hasbrouk_coef',
                                  'tb_hasbrouk_t']
                                +cols+
                                  ['event', 
                                    'nbars_400_2sl_1tp', 'pctAtBarrier_400_2sl_1tp', 'sideOfBarrier_400_2sl_1tp', 'barTimeOfExit_400_2sl_1tp', 'target_400_2sl_1tp_minRetPt5Pct',
                                'nbars_400_1sl_1tp', 'pctAtBarrier_400_1sl_1tp', 'sideOfBarrier_400_1sl_1tp', 'barTimeOfExit_400_1sl_1tp', 'target_400_1sl_1tp_minRetPt5Pct',
                                'nbars_400_1sl_2tp', 'pctAtBarrier_400_1sl_2tp', 'sideOfBarrier_400_1sl_2tp', 'barTimeOfExit_400_1sl_2tp', 'target_400_1sl_2tp_minRetPt5Pct',
                                'nbars_400_0pt5sl_2tp', 'pctAtBarrier_400_0pt5sl_2tp', 'sideOfBarrier_400_0pt5sl_2tp', 'barTimeOfExit_400_0pt5sl_2tp', 'target_400_0pt5sl_2tp_minRetPt5Pct']]

      print(secDict[tic].isna().sum().sum())

      print(len(secDict[tic]))

      secDict[tic].to_csv(f'/docker_storage/Tick_Data/Const_Resampled_2017/dolDF_const_10_BarsPerDay_wFeatsAndTgtsAndEvents/FullDataDict_400NbarLim_{tic}.csv')
      
  except Exception as e: 
      print('failed becuase -> ', e)



def update_transform_stage_three(tic_lst):
    for ticker in tic_lst:
        make_all_features(ticker)
