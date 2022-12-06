import streamlit as st
import os
import pandas as pd
from biomassters.interface.main import load_dataset
from contextlib import contextmanager, redirect_stdout
from io import StringIO



'''
# BioMassters DataSet App
'''

st.markdown(f'''
Here you can download data for the BioMassters challenge.\n
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

@contextmanager
def st_capture(output_func):
    with StringIO() as stdout, redirect_stdout(stdout):
        old_write = stdout.write

        def new_write(string):
            ret = old_write(string)
            output_func(stdout.getvalue())
            return ret

        stdout.write = new_write
        yield


output = st.empty()
if st.button('Download files'):
    with st_capture(output.code):
        load_dataset()

st.subheader("""Downloading Info""")
st.info(download_info)

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