import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.varmax import VARMAX
from random import random
from constants import tickers,metals

warnings.simplefilter('ignore')

df = pd.read_csv('all.csv', index_col=0)
#scaler = MinMaxScaler(feature_range=(1, 2))
#df_all = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
df_all = df
df_all.index = pd.to_datetime(df_all.index)
df_stocks = df_all[tickers]
df_metals = df_all[metals]

# Common code for display result
def show_graph(df1,df2,title):
    data = pd.concat([df1, df2])
    data.reset_index(inplace=True, drop=True)
    for col in data.columns:
        if col.lower().startswith('pred'):
            data[col].plot(label=col,linestyle="dotted")
        else:
            data[col].plot(label=col)
    plt.title(title)
    plt.legend()
    plt.show()

def VARMA_model(train, test):
    # fit model
    model = VARMAX(train, order=(1, 2))
    model_fit = model.fit(disp=True)
    # make prediction
    yhat = model_fit.forecast(steps=len(test))
    r={}
    for col in train.columns:
        r[f'Pred_{col}'] = yhat[col]
        r[col] = test[col].values
    res = pd.DataFrame(r)
    return res

train_sample = int(len(df_all)*0.8)
for m in metals:
    df_selected = df_stocks
    df_selected[m] = df_metals[m]
    df_train = df_selected[1:train_sample]
    df_test = df_selected[train_sample:]
    df_ret = VARMA_model(df_train, df_test)
    show_graph(df_train, df_ret, "Vector Autoregression Moving-Average (VARMA)")

def VARMAX_model(train,test):
    # fit model
    # ------------ PAKOREGUOTI EXOG KINTAMAJI ----------------------
    model = VARMAX(train.drop('Exog', axis=1), exog=train['Exog'], order=(1, 1))
    model_fit = model.fit(disp=True)
    # make prediction
    yhat = model_fit.forecast(steps=len(test),exog=test['Exog'])
    res=pd.DataFrame({"Pred1":yhat['Act1'], "Pred2":yhat['Act2'],
            "Act1":test["Act1"].values, "Act2":test["Act2"].values, "Exog":test["Exog"].values})
    return res

df_train = pd.DataFrame({'Act1':[x + random()*10 for x in range(0, 100)],
                         'Act2':[x*3 + random()*10 for x in range(0, 100)],
                         'Exog':50+np.sin(np.linspace(0, 2*np.pi, 100))*50})
df_test = pd.DataFrame({'Act1':[x + random()*10 for x in range(101, 201)],
                         'Act2':[x*3 + random()*10 for x in range(101, 201)],
                         'Exog':50+np.sin(np.linspace(0, 2*np.pi, 100))*50})
df_ret = VARMAX_model(df_train, df_test)
show_graph(df_train, df_ret,"Vector Autoregression Moving-Average with Exogenous Regressors (VARMAX)")