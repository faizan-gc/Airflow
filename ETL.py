import sys
sys.path.append('./dags/mlfinlab/')
import os
import pandas as pd
import sklearn
import mlfinlab
import json
import warnings
import datetime
import aiohttp
import pytz
import asyncio
import json
import pandas as pd
import numpy as np
import requests
import os
import glob
import nest_asyncio
import sys
import pickle
import glob
import os.path
from airflow import DAG
from airflow.decorators import dag, task
from datetime import datetime, timedelta
from mlfinlab.data_structures import standard_data_structures
from datetime import date as dt
from datetime import time as tm
from datetime import datetime, date, time, timedelta
warnings.simplefilter(action='ignore', category=FutureWarning)
nest_asyncio.apply()
from Extract import get_data
from Extract_Update import update_all
from Transform import step_one
from Transform_two import transform_stage_two, update_transform_stage_two
from Tranform_three import make_all_features, update_transform_stage_three
from Transform_update import update_step_one
import talib


DEFAULT_ARGS = {
    'owner' : 'faizan',
    'retries' : 5,
    'retry_delay' : timedelta(minutes=1)
}



@dag(
    dag_id='dag_test_new_4',
    default_args=DEFAULT_ARGS,
    start_date=datetime.now(),
    schedule_interval='@daily'
    )

def test_etl():

    @task
    def extract_tick_data():
        ticker_info = get_data()
        return ticker_info

    @task
    def transform_and_save(ticker_info):
        ticker_info_updated = step_one(ticker_info)
        return ticker_info_updated

    @task
    def transform_stage_two_and_save(ticker_info_updated):
        tic_name = transform_stage_two(ticker_info_updated)
        return tic_name

    @task
    def make_features(tic_name):
        make_all_features(tic_name)



    ticker_info = extract_tick_data()
    ticker_info_updated = transform_and_save(ticker_info)
    tic_name = transform_stage_two_and_save(ticker_info_updated)
    make_features(tic_name)


Dag_main = test_etl()

##########################################################################################################################################################

@dag(
    dag_id='dag_test_new_5',
    default_args=DEFAULT_ARGS,
    start_date=datetime.now(),
    schedule_interval='@daily'
    )

def update_etl():

    @task
    def update_tick_data():
        complete_ticker_info = update_all()
        return complete_ticker_info

    @task
    def update_transform_and_save(ticker_info):
        if len(ticker_info) != 0:
            ticker_info_updated = update_step_one(ticker_info)
            return ticker_info_updated
        else:
            print("Every ticker is up to date!")
            return None

    @task
    def update_transform_stage_two_and_save(ticker_info_updated):
        if ticker_info_updated is not None:
            tic_lst = update_transform_stage_two(ticker_info_updated)
            return tic_lst
        else:
            print("Every ticker is up to date!")
            return None

    @task
    def update_make_features(tic_lst):
        if tic_lst is not None:
            update_transform_stage_three(tic_lst)
        else:
            print("Every ticker is up to date!")



    ticker_info = update_tick_data()
    ticker_info_updated = update_transform_and_save(ticker_info)
    tic_lst = update_transform_stage_two_and_save(ticker_info_updated)
    update_make_features(tic_lst)
     

Dag_update = update_etl()
