# To jest wersja 9 z Arima

import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime, timedelta, date
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import appdirs as ad
CACHE_DIR = ".cache"
# Force appdirs to say that the cache dir is .cache
ad.user_cache_dir = lambda *args: CACHE_DIR
# Create the cache dir if it doesn't exist
Path(CACHE_DIR).mkdir(exist_ok=True)
import yfinance as yf
from sklearn.linear_model import LinearRegression
from streamlit import set_page_config
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA

# Set page configuration for full width
set_page_config(layout="wide")

# start definicji strony
st.title('The main global economy indicators and my own EUR/PLN D5 prediction mode')

# Definicje
today = date.today()
comm_dict = {'EURUSD=X':'USD_EUR','CNY=X':'USD/CNY','CL=F':'Crude_Oil',
             '^DJI':'DJI30','GC=F':'Gold','^IXIC':'NASDAQ',
             '^GSPC':'SP_500','^TNX':'10_YB',
             'HG=F':'Copper','GBPUSD=X':'USD_GBP',
             'JPY=X':'USD_JPY','EURPLN=X':'EUR/PLN','PLN=X':'PLN/USD'
             ,'^FVX':'5_YB','RUB=X':'USD/RUB','PL=F':'Platinum',
             'SI=F':'Silver','NG=F':'Natural Gas','ZR=F':'Rice Futures',
             'ZS=F':'Soy Futures','KE=F':'KC HRW Wheat Futures'}

# Pobieranie danych
def comm_f(comm):
    global df_c1
    for label, name in comm_dict.items():
        if name == comm:
            df_c = pd.DataFrame(yf.download(f'{label}', start='2000-09-01', end = today,interval='1d'))
            df_c1 = df_c.reset_index()
           
    return df_c1   

# Dane historyczne                    
def comm_data(comm):
    global Tab_his1
    shape_test=[]
    sh = df_c1.shape[0]
    start_date = df_c1.Date.min()
    end_date = df_c1.Date.max()
    close_max = "{:.2f}".format(df_c1['Close'].max())
    close_min = "{:.2f}".format(df_c1['Close'].min())
    last_close = "{:.2f}".format(df_c1['Close'].iloc[-1])
    v = (comm, sh, start_date,end_date,close_max,close_min,last_close)
    shape_test.append(v)
    Tab_length = pd.DataFrame(shape_test, columns= ['Name','Rows', 'Start_Date', 'End_Date','Close_max','Close_min','Last_close'])   
    Tab_his = Tab_length[['Start_Date','End_Date','Close_max','Close_min','Last_close']]
    Tab_his['Start_Date'] = Tab_his['Start_Date'].dt.strftime('%Y-%m-%d')
    Tab_his['End_Date'] = Tab_his['End_Date'].dt.strftime('%Y-%m-%d')
    Tab_his1 = Tab_his.T
    Tab_his1.rename(columns={0: "Details"}, inplace=True)
    
    return Tab_his1

st.sidebar.title('Commodities, Indexies, Currencies & Bonds')
comm = st.sidebar.selectbox('What do you want to analyse today ?', list(comm_dict.values()))
comm_f(comm)
st.sidebar.write('You selected:', comm)
st.sidebar.dataframe(comm_data(comm))

# tu wstawimy wykresy 15 minutowe
def t1_f(char1):
    global tf_c1
    for label, name in comm_dict.items():
        if name == char1:
            box = yf.Ticker(label)
            tf_c = pd.DataFrame(box.history(period='1d', interval="1m"))
            tf_c1 = tf_c[-100:]
    return tf_c1 

def t2_f(char2):
    global tf_c2
    for label, name in comm_dict.items():
        if name == char2:        
            box = yf.Ticker(label)
            tf_c = pd.DataFrame(box.history(period='1d', interval="1m"))
            tf_c2 = tf_c[-100:]
    return tf_c2 


col1, col2 = st.columns([0.47, 0.53])
with col1:
    box = list(comm_dict.values())
    char1 = st.selectbox('Daily trading dynamics', box, index= box.index('Crude_Oil'),key = "<char1>")
    t1_f(char1)
    data_x1 = tf_c1.index
    fig_char1 = px.line(tf_c1, x=data_x1, y=['Open','High','Low','Close'],color_discrete_map={
                 'Open':'yellow','High':'red','Low':'blue','Close':'green'}, width=500, height=400) 
    fig_char1.update_layout(showlegend=False)
    fig_char1.update_layout(xaxis=None, yaxis=None)
    st.plotly_chart(fig_char1) #use_container_width=True
with col2:
    char2 = st.selectbox('Daily trading dynamics', box, index=box.index('PLN/USD'),key = "<char2>")
    t2_f(char2)
    data_x2 = tf_c2.index
    fig_char2 = px.line(tf_c2, x=data_x2, y=['Open','High','Low','Close'],color_discrete_map={
                 'Open':'yellow','High':'red','Low':'blue','Close':'green'}, width=500, height=400) 
    fig_char2.update_layout(showlegend=True)
    fig_char2.update_layout(xaxis=None, yaxis=None)
    st.plotly_chart(fig_char2)

# tutaj wprowadzamy kod do wykresów 
st.subheader(comm+' Prices in NYSE')
xy = (list(df_c1.index)[-1] + 1)  
col3, col4 = st.columns([0.7, 0.3])
with col3:
    oil_p = st.slider('How long prices history you need?', 0, xy, 100, key = "<commodities>")
with col4:
    numm = st.number_input('Enter the number of days for moving average',value=30, key = "<m30>")
    st.write(f'Custom mean of {numm} days ')

def roll_avr(numm):
    global df_c_XDays
    df_c1[(f'M{numm}')]= df_c1['Close'].rolling(window=numm).mean()
    df_c1['M90']= df_c1['Close'].rolling(window=90).mean()
    df_c_XDays = df_c1.iloc[xy - oil_p:xy]
    
    fig1 = px.line(df_c_XDays,x='Date', y=['High',(f'M{numm}'),'M90'],color_discrete_map={
                  'High':'#d62728',(f'M{numm}'): '#f0f921','M90':'#0d0887'}, width=900, height=400) 
    fig1.update_layout(xaxis=None, yaxis=None)
    st.plotly_chart(fig1, use_container_width=True)

roll_avr(numm)
    
# Arima - model - prognoza trendu
def Arima_f(comm):
    data = np.asarray(df_c1['Close'][-300:]).reshape(-1, 1)
    p = 10
    d = 0
    q = 5
    n = size_a

    model = ARIMA(data, order=(p, d, q))
    model_fit = model.fit(method_kwargs={'maxiter': 3000})
    model_fit = model.fit(method_kwargs={'xtol': 1e-6})
    fore_arima = model_fit.forecast(steps=n)  
    
    arima_dates = [datetime.today() + timedelta(days=i) for i in range(0, size_a)]
    arima_pred_df = pd.DataFrame({'Date': arima_dates, 'Predicted Close': fore_arima})
    arima_pred_df['Date'] = arima_pred_df['Date'].dt.strftime('%Y-%m-%d')
    arima_df = pd.DataFrame(df_c1[['Date','High','Close']][-500:])
    arima_df['Date'] = arima_df['Date'].dt.strftime('%Y-%m-%d')
    arima_chart_df = pd.concat([arima_df, arima_pred_df], ignore_index=True)
    x_ar = (list(arima_chart_df.index)[-1] + 1)
    arima_chart_dff = arima_chart_df.iloc[x_ar - 30:x_ar]
    
    fig_ar = px.line(arima_chart_dff, x='Date', y=['High', 'Close', 'Predicted Close'], color_discrete_map={
                  'High': 'yellow', 'Close': 'black', 'Predicted Close': 'red'}, width=900, height=500)
    fig_ar.add_vline(x = today,line_width=3, line_dash="dash", line_color="green")
    fig_ar.update_layout(xaxis=None, yaxis=None)
    st.plotly_chart(fig_ar, use_container_width=True)      
    
# definicja wykresu obortów
def vol_chart(comm):
    volc = ['Crude_Oil','Gold','Copper','Platinum','Silver','Natural Gas','Rice Futures','Soy Futures','KC HRW Wheat Futures']
    if comm in volc:

        Co_V = df_c1[['Date', 'Volume']]
        Co_V['Co_V_M']= Co_V['Volume'].rolling(window=90).mean().fillna(0)
        V_end = (list(Co_V.index)[-1] + 1)

        st.subheader(comm+' Volume in NYSE')
        Vol = st.slider('How long prices history you need?', 0, V_end, 100, key = "<volume>") 
        Co_V_XD = Co_V.iloc[V_end - Vol:V_end]

        fig3 = px.area(Co_V_XD, x='Date', y='Volume',color_discrete_map={'Volume':'#1f77b4'})
        fig3.add_traces(go.Scatter(x= Co_V_XD.Date, y= Co_V_XD.Co_V_M, mode = 'lines', line_color='red'))
        fig3.update_traces(name='90 Days Mean', showlegend = False)

        st.plotly_chart(fig3, use_container_width=True)
     
vol_chart(comm)

# tu poblikujemy komentarz z yf
@st.cache_data
def news_stream():
    comm_dict1 = {'EURUSD=X':'USD_EUR','CNY=X':'USD/CNY'}
    subs = 'finance.yahoo.com'
    news = []
    today = date.today()
    for label, name in comm_dict1.items():
        Comm_news = yf.Ticker(name)
        info = Comm_news.get_news()
        info_link = list(info[0].values())

        for i in info_link: 
            try:
                if subs in i:
                    v = (today,name,i)
                    news.append(v)
                    news_tab = pd.DataFrame(news, columns= ['Date','Topic','Link'])
            except TypeError:
                   pass
                                            
    st.sidebar.subheader('Yahoo Finance current news review')
    st.sidebar.write('News about '+news_tab.Topic[0])
    st.sidebar.markdown(news_tab.Link[0])
    st.sidebar.write('News about '+news_tab.Topic[1])
    st.sidebar.markdown(news_tab.Link[1])   

news_stream()

# Definicje do Korelacji
comm_dict2 = {k: v for k, v in comm_dict.items() if k != '^DJI'}
cor_history = [100,500,1000,1500,2000]

#Źródło danych do korelacji
@st.cache_data
def cor_tab(past):
    df_list = []
    start_date = (datetime.today() - timedelta(days=past)).strftime('%Y-%m-%d')
    col_n = {'Close': 'DJI30'}
    x = pd.DataFrame(yf.download('^DJI', start=start_date, end = today))
    x2 = x[['Close']][-past:]
    x2.rename(columns = col_n, inplace=True)
    x2 = pd.DataFrame(x2.reset_index(drop=True)) 
    for label, name in comm_dict2.items(): 
        col_name = {'Close': name}
        y1 = pd.DataFrame(yf.download(label, start=start_date, end = today)) 
        y2 = y1[['Close']][-past:]
        y2 = pd.DataFrame(y2.reset_index(drop=True))
        y2.rename(columns = col_name, inplace=True)
        m_tab = pd.concat([x2, y2], axis=1)
        df_list.append(m_tab)
        cor_df = pd.concat(df_list, axis=1)
        cor_df = cor_df.T.drop_duplicates().T 
        cor_rr = (cor_df - cor_df.shift(1)) / cor_df.shift(1)
        cor_rr.fillna(0)
        cor_data = cor_rr.to_pickle('cor_data.pkl')

@st.cache_data        
def cor_chart(cor1,cor2):
    cor_d2 = pd.read_pickle('cor_data.pkl')
    cor = cor_d2.corr()
    cor_below_76 = cor[(cor >= cor1) & (cor <= cor2) & (cor != 1.0)].stack()
    below_76 = pd.DataFrame(cor_below_76)
    below_76.reset_index(inplace=True)
    below_76 = below_76.rename(columns={'level_0': 'Com_1', 'level_1': 'Com_2', 0: 'Cor_v'})

    fig_76 = px.scatter(below_76, x='Com_1', y='Com_2', color='Cor_v', template='plotly_white')
    fig_76.update_traces(showlegend=False)
    fig_76.update_layout(width=1000, height=500)
    fig_76.update_layout(xaxis=None, yaxis=None)
    st.plotly_chart(fig_76)        
                               
col5, col6, col7 = st.columns(3)
with col5:
    checkbox_value3 = st.checkbox(f'Arima model trend prediction for x days',key = "<arima_m>")

if checkbox_value3:
    st.subheader(f'{comm} Arima model prediction')
    size_a = st.radio('Prediction for ... days ?: ', [5,4,3,2,1], horizontal=True, key = "<arima21>")
    Arima_f(comm)    

with col6:
    checkbox_value2 = st.checkbox(f'Own LSTM model EUR/PLN prediction for 5 days',key = "<lstm1>")

if checkbox_value2:
    st.subheader('Own LSTM EUR/PLN model prediction')
    val_oil = pd.read_excel('LSTM_mv.xlsx', sheet_name='D5_EUR')
    val_oil1 = val_oil[['Date','EUR/PLN','Day + 5 Prediction']] 
    fig_oil1 = px.line(val_oil1[-50:], x='Date', y=['EUR/PLN','Day + 5 Prediction'],color_discrete_map={
                 'EUR/PLN':'dodgerblue','Day + 5 Prediction':'red'}, width=1200, height=600, title=f'Day + 5 EUR/PLN prediction ') 
    fig_oil1.update_layout(plot_bgcolor='white',showlegend=True,xaxis=dict(showgrid=True, gridwidth=0.5, gridcolor='Lightgrey'),
                      yaxis=dict(showgrid=True, gridwidth=0.5, gridcolor='Lightgrey'))
    fig_oil1.add_vline(x = today,line_width=1, line_dash="dash", line_color="black")
    fig_oil1.add_annotation(x=today , y= ['Day + 5 Prediction'], text= f'Today - {today}', showarrow=False)
    st.plotly_chart(fig_oil1)
    
with col7:
    checkbox_value4 = st.checkbox('Correlation table for today',key = "<cor>")

if checkbox_value4:
    st.subheader('Correlation table for today')
    past = st.radio('Correlation based on x days: ', cor_history, horizontal=True, key = "<cor11>")
    col8, col9 = st.columns(2)
    with col8:
       cor1 = st.slider('Range from', -1.0, 1.0, 0.1, key = "<cor1>")
    with col9:
       cor2 = st.slider('Range to', -1.0, 1.0, 0.6, key = "<cor2>")   
    
    cor_tab(past)
    cor_chart(cor1,cor2)