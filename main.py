## https://www.youtube.com/watch?v=CSv2TBA9_2E

import streamlit as st 
import pandas as pd 

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from pyarrow.parquet import ParquetFile
import pyarrow as pa 


header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()

## do some customization with background and fonts
st.markdown(
    """
    <style>
    .main{
    background-color: #FSFSFS;
    }
    </style>
    """,
    unsafe_allow_html=True
)


## Caching use st.cache_data or st.cache_resource
@st.cache_data
def get_Data(filename):
#    taxi_data = pd.read_parquet('Data/yellow_tripdata_2024-09.parquet')
    pf = ParquetFile(filename)
    first_100_rows = next(pf.iter_batches(batch_size = 1000))
    taxi_data = pa.Table.from_batches([first_100_rows]).to_pandas()
    return taxi_data
##    st.write(taxi_data.head())



with header:
    st.title("Welcome to my awesome data science project!")
    st.text('In this project I look into transactions of taxis in NYC')

with dataset:
    st.header('NYC taxi dataset')
    st.text('I found this dataset on ....com')

    taxi_data = get_Data('Data/yellow_tripdata_2024-09.parquet')

    st.subheader('Puckup location ID disctibution NYC')
    pulocation_dist = pd.DataFrame(taxi_data['PULocationID'].value_counts().head(50))
    st.bar_chart(pulocation_dist)


with features:
    st.header('The features I created')

    st.markdown('* **first feature:** I created this feature because of I wanted to calculate')


with model_training:
    st.header('Time to train the model')
    st.text('Here you get to choose the hyperparameters of the model and see how the performance change')

## Create columns
sel_col, disp_col = st.columns(2)

max_depth = sel_col.slider('What should be the max depth of the model?', min_value=10, max_value=100, value=20, step=10)

n_estimators = sel_col.selectbox('How many should there be?', options=[100,200,300,'No limit'], index=0)

imput_feature = sel_col.text_input('Which feature should be used as the input feature?', 'PULocationID')

## Random forest regressor
if n_estimators == 'No limit':
    regr = RandomForestRegressor(max_depth=max_depth)
else:
    regr = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)

X = taxi_data[[imput_feature]]
y = taxi_data[['trip_distance']]

regr.fit(X, y)
prediction = regr.predict(y)

disp_col.subheader('Mean absolute error for the model is:')
disp_col.write(mean_absolute_error(y, prediction))

disp_col.subheader('Mean squared error of the model is:')
disp_col.write(mean_squared_error(y, prediction))

disp_col.subheader('R squared error of the model is:')
disp_col.write(mean_r2_error(y, prediction))
