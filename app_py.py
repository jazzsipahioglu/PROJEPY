import warnings

import prophet as prophet
from prophet.plot import plot_cross_validation_metric

warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
from plotly import graph_objs as go
import yfinance as yf
import streamlit as st
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from datetime import date
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import base64
from price_predictor import price_predictor as pp
from sklearn.svm._libsvm import cross_validation
from xgboost import XGBClassifier
from prophet import Prophet
import math
import datetime as dt
from itertools import cycle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance, accuracy_score
from plotly.offline import plot, iplot, init_notebook_mode
init_notebook_mode(connected=True)
import pickle




st.title("Finansal Analiz Platformu")
st.write("Borsa, Kriptopara, Emtia Bilgi Platformu")
st.sidebar.title("Filtrele")

islemturu=st.sidebar.radio("Varlık Türü",["Kripto","Borsa","Döviz","Emtia"])
if islemturu=="Kripto":
    kriptosec=st.sidebar.selectbox("Kripto Para Cinsi",["BTC","ETH","XRP","DOT","DOGE","AVAX","BNB"])
    kriptosec=kriptosec+"-USD"
    sembol=kriptosec

elif islemturu=="Emtia":
    emtiasec=st.sidebar.selectbox("Emtia",["ALTIN","PETROL"])
    emtialar={
        "ALTIN":"GC=F",
        "PETROL":"CL=F"
    }
    commoditysec=emtialar[emtiasec]
    sembol=commoditysec

elif islemturu=="Döviz":
    dovizsec=st.sidebar.selectbox("Döviz",["DOLAR","AVRO"])
    dovizler={
        "DOLAR":"USDTRY=X",
        "AVRO":"EURTRY=X"
    }
    parasec=dovizler[dovizsec]
    sembol=parasec

else:
    borsasec=st.sidebar.selectbox("Hisse Senetleri",["ASELSAN","THY","GARANTİ","AKBANK","BJK"])
    senetler={
        "ASELSAN":"ASELS.IS",
        "THY":"THYAO.IS",
        "GARANTİ":"GARAN.IS",
        "AKBANK":"AKBNK.IS",
        "BJK":"BJKAS.IS"
    }

    hissesec=senetler[borsasec]
    sembol=hissesec

zaman_aralik=range(1,91)
slider=st.sidebar.select_slider("Zaman Aralığı",options=zaman_aralik,value=30)

bugun=datetime.today()
aralik=timedelta(days=slider)

baslangic=st.sidebar.date_input("Başlangıç Tarihi",value=bugun-aralik)
bitis=st.sidebar.date_input("Bitiş Tarihi",value=bugun)

st.sidebar.write("### Machine Learning Tahmin")

prophet=st.sidebar.checkbox("Tahmin")




if prophet:
    pparalik=range(1,1500)
    ppperiyot=st.sidebar.select_slider("Periyot", options=zaman_aralik, value=30)
    components=st.sidebar.checkbox("Components")

if prophet:
    cvsec=st.sidebar.checkbox("CV")
    if cvsec:
        st.sidebar.write("#### Metrik Seçiniz")
        metrics=st.sidebar.radio("Metrik", ["rmse","mse","mape","mdape"])

        st.sidebar.write("####Parametre Seçiniz")
        inaralik=range(1,1500)
        cvin=st.sidebar.select_slider("Initial",options=inaralik,value=120)
        peraralik=range(1,1500)
        cvper=st.sidebar.select_slider("CV Periyot",options=peraralik, value=30)
        horaralik=range(1,1500)
        cvhor=st.sidebar.select_slider("Horizon",options=horaralik, value=60)

    else:
        pass


    def plot_cross_validation_metric(cv, metric):
        pass


    def cvgrafik(model,initial,period,horizon,metric):
        initial=str(initial)+"days"
        period=str(period)+"days"
        horizon=str(horizon)+"days"
        cv=cross_validation(model,initial=initial,period=period,horizon=horizon)
        grap3=plot_cross_validation_metric(cv,metric=metric)
        st.write(grap3)





def grafikgetir(sembol,baslangic,bitis):
    data=yf.Ticker(sembol)
    global df
    df=data.history(period="1d",start=baslangic, end=bitis)
    st.line_chart(df["Close"])
    if prophet:
        prophet=df.reset_index()
        prophet=prophet[["Date","Close"]]
        prophet.columns=["ds","y"]
        global model
        model= Prophet()


        model.fit(df)
        future = model.make_future_dataframe(periods=ppperiyot)
        forecast = model.predict(future)
        forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
        grap = model.plot(forecast)
        st.write(grap)

        if components:
            grap2=model.plot_components(forecast)
            st.write(grap2)


def indir (df):
    csv=df.to_csv()
    b64=base64.b64encode(csv.encode()).decode()
    href=f'<a href="data:file/csv;base64,{b64}"CSV İndir</a>'
    return href
st.markdown(indir(df),unsafe_allow_html=True)

def SMA(data,period=30,column='Close'):
    return data[column].rolling(window=period).mean()
def EMA (data, period=21,column='Close'):
    return data[column].ewm (span=period, adjust=False).mean()
def MACD (data, period_long=26, period_short=12, period_signal=9, column='Close', signal=None):
    ShortEMA=EMA(data,period_short, column=column)
    LongEMA=EMA(data,period_long,column="MACD")
    data["MACD"]=ShortEMA-LongEMA
    from pandas.core.arrays import period
    data["Signal_Line"]=EMA(data, period,signal, column=column)
    return
def RSI(data, period=14, column="Close", dowm=None):
    delta=data[column].diff(1)
    delta=delta[1:]
    up=delta.copy()
    down=delta.copy()
    up[up<0]=0
    down[down>0]=0
    data["up"]=up
    data["down"]=dowm
    AVG_Gain=SMA(data,period,column="up")
    AVG_Loss=abs(SMA(data,period,column="down"))
    RS=AVG_Gain/AVG_Loss
    RS=100.0-(100.0/(1.0+RS))
    data["RSI"]=RSI
    return data



st.sidebar.write("### Finansal İndikatörler")
fi=st.sidebar.checkbox("Finansal İndikatörler")

def filer():

    if fi:
        fimacd=st.sidebar.checkbox("MACD")
        firsi=st.sidebar.checkbox("RSI")
        fisl=st.sidebar.checkbox("Signal Line")

        if fimacd:
            macd=MACD(df)
            st.line_chart(macd["MACD"])

        if firsi:
            rsi=RSI(df)
            st.line_chart(rsi["RSI"])

        if fisli:
            sl=MACD(df)
            st.line_chart(macd["Signal Line"])

filer()
