import streamlit as st
import pandas as pd
from joblib import load
import numpy as np

with open('styles.css') as f:
    st.markdown(f'<style>{f.read()}</style>',unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    # st.subheader('AI Driven Deposition Fraction Calculator [%]')
    original_title = '<p style="font-family:Times New Roman;text-align: center; color:white; font-size: 25px;">AI Driven Deposition Fraction Calculator [%]</p>'
    st.markdown(original_title, unsafe_allow_html=True)
    for i in range(8):
        st.write("\n")
    den = st.number_input('Density [gram/cm^3]')
    dia = st.number_input('Diameter [nm]')
    MH = st.number_input('Medium Height [mm]')
    time = st.number_input('Time  [hour]')
    But1 = st.button('Calculate')
with col2:
    # st.subheader('Medium Height and Incubation Time for a given density and diameter to reach a desired deposition fraction of nano particle on the cell')
    original_title = '<p style="font-family:Times New Roman;text-align: left; color:white; font-size: 22px;height:205px;">Medium Height and Incubation Time calculator for a given density and diameter to reach a desired deposition fraction of nano particle on the cell</p>'
    st.markdown(original_title, unsafe_allow_html=True)
    density = st.number_input('Density[gram/cm^3]')
    diameter = st.number_input('Diameter[nm]')
    deposition = st.number_input('Deposition Fraction [%]')
    But2 = st.button('Calculate ')
if But1:
    scaler = load(open('scale.sav', 'rb'))
    model = load(open('randomForest.sav','rb'))
    x_test= np.array([[MH/1000,dia,den,time]])
    X_test_scaled = scaler.transform(x_test)
    DE = model.predict(X_test_scaled)

    d = {'Deposition Fraction %':[DE]}
    df = pd.DataFrame(data=d)
    st.table(df)
def df_combination(density,diameter,deposition):

    data=[]

    for d in [0.0001, 0.001, 0.005, 0.015, 0.01, 0.02]:
        for t in range(2,73,2):
            data.append([d,diameter,density,t])

    dfs=pd.DataFrame(data,columns=['Depth','Diameter','Density','Time'])

    scaler = load(open('scale.sav', 'rb'))
    model = load(open('randomForest.sav','rb'))

    dtest=scaler.transform(dfs.values)

    dfs['pred']=model.predict(dtest)

    res=[]
    for d in [0.0001, 0.001, 0.005, 0.01, 0.015, 0.02]:
        dfd=dfs.loc[(dfs['Depth']==d)&(dfs['pred']>=deposition)]
        dfd.sort_values(by=['Time'])

        if dfd.shape[0] >= 0:
            res.append([d,dfd['Time'].min()])
        else:
            res.append([d,np.nan])


    return pd.DataFrame(res,columns=['Medium Height[mm]','Time[h]'])
if But2:

    df1 = df_combination(density,diameter,deposition)
    st.table(df1)
