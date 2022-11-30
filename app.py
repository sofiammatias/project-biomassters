import streamlit as st
import os
import pandas as pd
from biomassters.interface.main import load_dataset



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
    # CODE TO CONSIDER SATELLITE IMAGE CHOICE
    #sat = st.selectbox(
    #    "Select satellite imagery: ",
    #    ("Both", "S1", "S2")
    #    )
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



os.environ['MODE'] = mode
os.environ['MONTH'] = month


if (month == 'All'): #and (sat == 'Both'):
    filtered_features = features[(features.split == mode.lower())]
else:
# elif    sat == 'Both':
    filtered_features = features[(features.month == month) &
                                 (features.split == mode.lower())]
# CODE TO CONSIDER SATELLITE IMAGE CHOICE
#elif month == 'All':
#    filtered_features = features[(features.sattelite == sat) &
#                                 (features.split == mode.lower())]
#else:
#    filtered_features = features[(features.month == month) &
#                                 (features.split == mode.lower()) &
#                                 (features.satellite == sat)]


    # USE A CHIP_ID TO START YOUR DOWNLOAD
    #chip_id_start = st.selectbox("Select the chip_id where download should start:",
    #                        filtered_features['chip_id'].sort_values())
    #st.dataframe(filtered_features[['chip_id', 'file_downloaded']])


os.environ['CHIP_ID_SIZE'] = chip_id_size

chip_id_list = filtered_features.chip_id.unique()[ : int(chip_id_size)]
chip_id_list = pd.Series(chip_id_list)
file_list_df = filtered_features[filtered_features['chip_id'].isin(chip_id_list)]

with col2:
    st.markdown("""
              List of files to download
              """)
    st.dataframe(file_list_df[['chip_id', 'filename']])

download_size = file_list_df['size'].sum() / 10**6

download_info = f""" \n Files to download: {len(file_list_df)} \n
Download size: {download_size} Mb \n
Download time @2Mb/s: {round(download_size / 120, ndigits = 2)} mins"""

if st.button('Download files'):
    os.system ("python -c 'from biomassters.interface.main import load_dataset; load_dataset()'")

st.subheader("""Downloading Info""")
st.info(download_info)

# Calling API URL - TO USE AN API

# url = 'https://taxifare-wsbi2k6wha-ew.a.run.app/predict'

# if url == 'https://taxifare.lewagon.ai/predict':

#    st.markdown('Maybe you want to use your own API for the prediction, not the one provided by Le Wagon...')


#url_req = f"""https://taxifare-wsbi2k6wha-ew.a.run.app/predict?pickup_datetime={date} {time}&pickup_longitude={pickup_long}&pickup_latitude={pickup_lat}&dropoff_longitude={dropoff_long}&dropoff_latitude={dropoff_lat}&passenger_count={passengers}"""

# fare = requests.get(url_req)

# final_fare = round (float(fare.text.split(':')[1].replace('}', '').strip()), ndigits=2)




datasize = round (filtered_features['size'].sum() / 10**6, ndigits = 3)

st.header("""
          Features Metadata
          """)

st.subheader(f"""
             'features_metadata.csv': {mode} data for month {month}
              """)

st.dataframe(filtered_features)

st.markdown (f""" ## Total number of files: {len(filtered_features)}""")
st.markdown (f""" ## Total data to download: {datasize} Mb""")


# 2. Let's build a dictionary containing the parameters for our API...
# 3. Let's call our API using the `requests` package...
# 4. Let's retrieve the prediction from the **JSON** returned by the API...
## Finally, we can display the prediction to the user
