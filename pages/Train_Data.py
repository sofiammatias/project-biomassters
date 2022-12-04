import streamlit as st
import os
import pandas as pd
from biomassters.data_sources.utils import features_mode, features_per_month
from biomassters.ml_logic.params import FEATURES_FILE



'''
# BioMassters DataSet App
'''

st.markdown(f'''
Here you can sort data to train the model for the BioMassters challenge.\n
''')

st.header("""
          Train Data
          """)

features = FEATURES_FILE

col1, col2 = st.columns(2)

with col1:
    # CODE TO CONSIDER SATELLITE IMAGE CHOICE
    #sat = st.selectbox(
    #    "Select satellite imagery: ",
    #    ("Both", "S1", "S2")
    #    )
    st.write ("Dataset: Train")

    st.write ("Months selected for training: from May to September")
#    month = st.selectbox(
#        "Select data month: ",
#        ("January", "February", "March", "April", "May",
#         "June", "July", "August", "September", "October",
#         "November", "December", "All")
#        )

    chip_id_size = st.select_slider("Select the chip_id size (number of chip_id's per chunk):",
                                    options=['1', '5', '10', '20', '50', '100', '1000'])


os.environ['MODE'] = 'train'
from biomassters.ml_logic.params import MODE

filtered_features = features_mode(features, MODE)
filtered_features = pd.concat (features_per_month(filtered_features, 'April'),
                    features_per_month(filtered_features, 'May'),
                    features_per_month(filtered_features, 'June'),
                    features_per_month(filtered_features, 'July'),
                    features_per_month(filtered_features, 'August'),
                    features_per_month(filtered_features, 'September'))
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

download_info = f""" \n Files to train: {len(file_list_df)} \n
Training time (estimate): {int(len(file_list_df) * 15 / (25 * 60))} mins"""

if st.button('Train model'):
    os.system ("make run_train")
st.header("""
          Features Metadata
          """)

st.subheader(f"""
             'features_metadata.csv': Train data for months April to September
              """)

st.dataframe(filtered_features)

st.markdown (f""" ## Total number of files: {len(filtered_features)}""")
