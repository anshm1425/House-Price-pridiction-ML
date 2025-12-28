import streamlit as st
import pickle
from sklearn.preprocessing import StandardScaler
import pandas as pd
import time
from sklearn.datasets import fetch_california_housing

st.title('🏘️ HOUSE PRICE PRIDICTION USING ML')
st.image('https://i.pinimg.com/originals/93/c7/44/93c744bcde1780c94bb1d3f03991f8a7.gif')

df = pd.read_csv('house_data.csv')
X = df.iloc[:,:-3]
y = df.iloc[:,-1]

st.sidebar.title('🏡 Select House feature')
st.sidebar.image('https://i.pinimg.com/originals/93/c7/44/93c744bcde1780c94bb1d3f03991f8a7.gif')
all_value = []
for i in X:
  ans = st.sidebar.slider(f'Select {i} value')
  all_value.append(ans)

st.write(all_value)


