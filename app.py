import streamlit as st
import os
import datetime
import requests
import pandas as pd



'''
# BioMassters DataSet App
'''

st.markdown(f'''
This is an app to download data for the BioMassters challenge.\n
Data will be downloaded in {os.getenv('LOCAL_DATA_PATH')}
''')

st.header("""
          Load Data
          """)

features = pd.read_csv ('../project-biomassters/data/features_metadata.csv')

col1, col2 = st.columns(2)

with col1:
    sat = st.selectbox(
        "Select satellite imagery: ",
        ("Both", "S1", "S2")
        )
    mode = st.selectbox(
        "Select data for: ",
        ("Train", "Test")
        )

    month = st.selectbox(
        "Select data month: ",
        ("January", "February", "March", "April", "May",
         "June", "July", "August", "September", "October",
         "November", "December", "All")
        )

    chip_id_size = st.select_slider("Select the chip_id size (number of chip_id's per chunk):",
                                    options=['1', '5', '10', '20', '50', '100', '1000'])

    chip_id_num = st.slider("Select the number of chip_id chunks:",
                            1, 15, 1)


os.environ['MODE'] = mode
os.environ['MONTH'] = month
os.environ['CHIP_ID_NUM'] = str(chip_id_num)
os.environ['CHIP_ID_SIZE'] = chip_id_size



if (month == 'All') and (sat == 'Both'):
    filtered_features = features[(features.split == mode.lower())]
elif sat == 'Both':
    filtered_features = features[(features.month == month) &
                                 (features.split == mode.lower())]
elif month == 'All':
    filtered_features = features[(features.sattelite == sat) &
                                 (features.split == mode.lower())]
else:
    filtered_features = features[(features.month == month) &
                                 (features.split == mode.lower()) &
                                 (features.satellite == sat)]


with col2:
    st.dataframe(filtered_features[['chip_id', 'file_downloaded']])

datasize = round (filtered_features['size'].sum() / 10**6, ndigits = 3)

st.markdown (f""" ## Total number of files: {len(filtered_features)}""")

st.markdown (f""" ## Total data to download: {datasize} Mb""")


# Calling API URL

# url = 'https://taxifare-wsbi2k6wha-ew.a.run.app/predict'

# if url == 'https://taxifare.lewagon.ai/predict':

#    st.markdown('Maybe you want to use your own API for the prediction, not the one provided by Le Wagon...')


#url_req = f"""https://taxifare-wsbi2k6wha-ew.a.run.app/predict?pickup_datetime={date} {time}&pickup_longitude={pickup_long}&pickup_latitude={pickup_lat}&dropoff_longitude={dropoff_long}&dropoff_latitude={dropoff_lat}&passenger_count={passengers}"""

# fare = requests.get(url_req)

# final_fare = round (float(fare.text.split(':')[1].replace('}', '').strip()), ndigits=2)
datasize = round (filtered_features['size'].sum() / 10**6, ndigits = 3)

st.subheader("""
             'features_metadata.csv'
              """)

st.dataframe(filtered_features)

# 2. Let's build a dictionary containing the parameters for our API...
# 3. Let's call our API using the `requests` package...
# 4. Let's retrieve the prediction from the **JSON** returned by the API...
## Finally, we can display the prediction to the user
