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
    st.title('DE Calculator')

with data_input:
    den=st.number_input('Particle Density')

    dia=st.number_input('Diameter')
    MH=st.number_input('MH')
    time=st.number_input('Time')


with Result:
    if But1:
        
        scaler = load(open('scale.sav', 'rb'))
        model = load(open('randomForest.sav','rb'))
        x_test= np.array([[MH,dia,den,time]])
        X_test_scaled = scaler.transform(x_test)
        DE = model.predict(X_test_scaled)

        d = {'result':[DE]}
        df = pd.DataFrame(data=d)
        st.dataframe(df)
        
data_input1 = st.container()
with data_input1:
    density=st.number_input('Density1')

    diameter=st.number_input('Diameter1')
    deposition=st.number_input('Deposition')

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


    return pd.DataFrame(res,columns=['Dish(mm)','Time(h)'])
Result1 = st.container()
But2 = st.button('Calculate1')
with Result1:
    if But2:

        df1 = df_combination(density,diameter,deposition)
        st.table(df1)
