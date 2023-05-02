import pandas as pd
import numpy as np 
from pathlib import Path
import yaml
import os
import matplotlib.pyplot as plt


lc_fp = Path('results/learning_curves')
fig_fp = lc_fp/'figures'


df_deep = pd.read_csv(lc_fp/'deep_ar_lc_2018.csv')
df_tft = pd.read_csv(lc_fp/'learning_curves_2018.csv')


df_deep.head()
df_tft.head()



def plot(title, df):
    plt.figure(figsize=(6,4))
    #plt.margins(x=0.02)
    plt.plot(df['Epoch'], df['train_loss'], label='Training')
    plt.plot(df['Epoch'], df['val_loss'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    #x_axis = [x+1 for x in df['Epoch'].to_numpy()]
    plt.xticks(df['Epoch'])
    plt.ylim()
    plt.title(title)
    plt.legend()
    #plt.show()
    plt.savefig(fig_fp/f'{title}.jpg')
    plt.close()

df_st_2018 = pd.read_csv(lc_fp/'Pei_loss_metrics_2018.csv')
df_st_2019 = pd.read_csv(lc_fp/'Pei_loss_metrics_2019.csv')
df_st_2020 = pd.read_csv(lc_fp/'Pei_loss_metrics_2020.csv')
df_st_2018['Epoch'] = df_st_2018['Epoch']+1
df_st_2019['Epoch'] = df_st_2019['Epoch']+1
df_st_2020['Epoch'] = df_st_2020['Epoch']+1
plot('Spacetimeformer Learning Curve - 2018', df_st_2018)
plot('Spacetimeformer Learning Curve - 2019', df_st_2019)
plot('Spacetimeformer Learning Curve - 2020', df_st_2020)

df_st_2020