import streamlit as st
import pickle
from sklearn.preprocessing import StandardScaler
import pandas as pd
import time
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor

st.title('🏘️ HOUSE PRICE PRIDICTION USING ML')
st.image('https://i.pinimg.com/originals/93/c7/44/93c744bcde1780c94bb1d3f03991f8a7.gif')

df = pd.read_csv('house_data.csv')
X = df.iloc[:,:-3]
y = df.iloc[:,-1]

st.sidebar.title('🏡 Select House feature')
st.sidebar.image('https://i.pinimg.com/originals/46/cb/a9/46cba9f93a3d437db6d42f4bcd1a5f5f.gif')
all_value = []
for i in X:
  min_value = int(X[i].min())
  max_value = int(X[i].max())
  ans = st.sidebar.slider(f'Select {i} value',min_value , max_value)
  all_value.append(ans)

# st.write(all_value)

scaler = StandardScaler()
scaled_X = scaler.fit_transform(X)

final_value = scaler.transform([all_value])

@st.cache_data
def model_run():
  model = st.cache(RandomForestRegressor)()
  model.fit(X,y)
  return model

model = model_run()


house_price = model.predict(final_value)[0]

with st.spinner('Predicting House price'):
  time.sleep(1) 
msg = f'''House Price is: ${round(house_price*100000,2)}'''  
st.success(msg)

st.markdown('''**Design & Developed by: ANSH MISHRA**''')















