import os
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pathlib import Path


def data_preprocess(years):
    # Data folder
    data = Path('data')
    fp = data/'CAISO_zone_1_.csv'

    # Ingest data
    df = pd.read_csv(fp)

    # Holidays
    holidays = USFederalHolidayCalendar().holidays()

    # Raw data is 2018, 2019, 2020
    for year in years:
        # Split by year and add index, place as first column
        #df_temp = df[df['time'].str.contains(year)].reset_index().drop(columns='index')
        df_temp = df[df['time'].str.contains(year)]
        df_temp['date_time'] = pd.to_datetime(df_temp['time'])
        df_temp['minute'] = df_temp['date_time'].dt.minute
        df_temp = df_temp.loc[df_temp['minute'] == 0].reset_index().drop(columns='index')
        df_temp['time_idx'] = df_temp.index
        first_column = df_temp.pop('time_idx')
        df_temp.insert(0, 'time_idx', first_column)
        
        # Add date related columns
        df_temp['date_time'] = pd.to_datetime(df_temp['time'])
        df_temp['month'] = df_temp['date_time'].dt.month
        df_temp['day'] = df_temp['date_time'].dt.day
        df_temp['month_day'] = df_temp['month'].astype(float) + df_temp['day'].astype(float)/31
        df_temp['day_of_week'] = df_temp['date_time'].dt.day_of_week
        df_temp['holiday'] = pd.to_datetime(df_temp['date_time'].dt.date).isin(holidays).astype(int)
        df_temp = df_temp.drop(columns='time')


        # Save
        name = 'CAISO_zone_1_' + year + '.csv'
        fp_temp = data/name
        df_temp.to_csv(fp_temp, index=False)

    return df_temp

years = ['2018', '2019', '2020']
data_preprocess(years)



