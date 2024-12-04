import pandas as pd
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from neuralforecast import NeuralForecast
from neuralforecast.models import GRU
from neuralforecast.losses.pytorch import DistributionLoss

from constants import tickers, metals
from random import seed
seed(42)

df = pd.read_csv('all.csv', index_col=0)
df.index = pd.to_datetime(df.index)
df_long = df.reset_index().melt(id_vars=['Date'], var_name='unique_id', value_name='y')
df_long.rename(columns={'Date': 'ds'}, inplace=True)
crit_date=df.index[int(0.2*len(df))]
train = df_long.loc[df_long['ds'] < crit_date]
valid = df_long.loc[df_long['ds'] >= crit_date]
h = valid['ds'].nunique()

models = [GRU(h=h,
               loss=DistributionLoss(distribution='Normal', level=[90]),
               max_steps=100,
               encoder_n_layers=2,
               encoder_hidden_size=200,
               context_size=10,
               encoder_dropout=0.5,
               decoder_hidden_size=200,
               decoder_layers=2,
               learning_rate=1e-3,
               scaler_type='standard')]

model = NeuralForecast(models=models, freq='D')
model.fit(train)

p = model.predict().reset_index()
p = p.merge(valid[['ds','unique_id', 'y']], on=['ds', 'unique_id'], how='left')

