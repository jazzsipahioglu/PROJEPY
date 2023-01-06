import streamlit as st
import pandas as pd
import numpy as np
import chart_studio.plotly as plotly
import plotly.figure_factory as ff
from plotly import graph_objs as go
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
import yfinance as yf
import streamlit as st

from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from datetime import date
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential

from xgboost import XGBRegressor
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

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

def grafikgetir(sembol,baslangic,bitis):
    data=yf.Ticker(sembol)
    df=data.history(period="1",start=baslangic,end=bitis)
    st.line_chart(df["Open"])
    if prophet:
        fb=df.reset_index()
        fb=fb[["Date","Close"]]
        fb.columns=["ds","y"]
        model=Prophet()
        model.fit(fb)
        future=model.make_future_dataframe(periods=360)
        predict=model.predict(future)
        predict=predict[["ds","trend"]]
        predict=predict.set_index("ds")
        st.line_chart(predict["trend"]       
grafikgetir(sembol,baslangic,bitis)

