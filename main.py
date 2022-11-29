import streamlit as st
import pandas as pd
from joblib import load
import numpy as np
import sklearn
header = st.container()
header1 = st.container()
data_input = st.container()
Result = st.container()
But1 = st.button('Calculate')


with header:
    st.title('AI Driven Deposition Fraction Calculator [%]')
st.caption("The following calculator estimates desired Medium Height and Incubation Time for a given density and diameter to reach a desired deposition fraction of nano particle on the cell.")

with data_input:
    den=st.number_input('Density [1.0-15.0][gram/cm^3]')
    dia=st.number_input('Diameter [1.0-1000.0] [nm]')
    MH=st.number_input('Medium Height [0.1-20] [mm]')
    time=st.number_input('Time [0-72] [hour]')


with Result:
    if But1:
        scaler = load(open('scale.sav', 'rb'))
        model = load(open('randomForest.sav','rb'))
        x_test= np.array([[MH,dia,den,time]])
        X_test_scaled = scaler.transform(x_test)
        DE = model.predict(X_test_scaled)

        d = {'Deposition Fraction %':[DE]}
        df = pd.DataFrame(data=d)
        st.table(df)
data_input1 = st.container()
with data_input1:
    density=st.number_input('Density [1.0-15.0] [gram/cm^3]')

    diameter=st.number_input('Diameter [1.0-1000.0][nm]')
    deposition=st.number_input('Deposition Fraction [0-100] [%]')

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
Result1 = st.container()
But2 = st.button('Calculate ')
with Result1:
    if But2:

        df1 = df_combination(density,diameter,deposition)
        st.table(df1)
