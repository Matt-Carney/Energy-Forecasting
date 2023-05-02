from pathlib import Path
import pandas as pd
import numpy as np
import os

perf = Path('results/performance')

cols = ['Parameters', 'LR', 'Val RMSE', 'Val MAE', 'Test RMSE', 'Val MAE.1']
df_2018 = pd.DataFrame(columns=cols)
df_2019 = pd.DataFrame(columns=cols)
df_2020 = pd.DataFrame(columns=cols)

for file in os.listdir(perf):
        fp = perf/file
        df = pd.read_csv(fp)
        if '2018' in file:
            df_2018 = pd.concat([df_2018, df])
            df_2018 = df_2018.drop_duplicates('Parameters')
        elif '2019' in file:
            df_2019 = pd.concat([df_2019, df])
            df_2019 = df_2019.drop_duplicates('Parameters')
        elif '2020' in file:
            df_2020 = pd.concat([df_2020, df])
            df_2020 = df_2020.drop_duplicates('Parameters')

df_2018 = df_2018.rename(columns= {'Val MAE.1': 'Test MAE'})
df_2019 = df_2019.rename(columns= {'Val MAE.1': 'Test MAE'})
df_2020 = df_2020.rename(columns= {'Val MAE.1': 'Test MAE'})


df_total = pd.concat([df_2018, df_2019, df_2020])

df_total = df_total.sort_values('Val RMSE')
df_total.iloc[:10,:]

df_2018.shape
df_2019.shape
df_2020.shape

