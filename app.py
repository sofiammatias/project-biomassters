import streamlit as st
import os
import pandas as pd
from biomassters.interface.main import load_dataset
from contextlib import contextmanager, redirect_stdout
from io import StringIO


st.image('https://images.unsplash.com/photo-1604009506606-fd4989d50e6d?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=870&q=80')
#         caption='Photo by <a href="https://unsplash.com/@chelseabock?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Chelsea Bock</a> on <a href="https://unsplash.com/s/photos/forest?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>')



'''
# BioMassters DataSet App
'''

st.markdown(f'''
Here you can download data for the BioMassters challenge.\n
Data will be downloaded in {os.getenv('LOCAL_DATA_PATH')}
''')

st.markdown('''
            See the BioMassters challenge in this link: [https://www.drivendata.org/competitions/99/biomass-estimation/page/534/](https://www.drivendata.org/competitions/99/biomass-estimation/page/534/)
            ''')


st.header("""
          Load Data
          """)

features = pd.read_csv ('../project-biomassters/data/features_metadata.csv')


col1, col2 = st.columns(2)

with col1:
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


if (month == 'All'):
    filtered_features = features[(features.split == mode.lower())]
else:
    filtered_features = features[(features.month == month) &
                                 (features.split == mode.lower())]

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
