import streamlit as st
import os
import pandas as pd
from biomassters.data_sources.utils import image_to_np
import matplotlib.pyplot as plt



'''
# BioMassters Predictions
'''

st.markdown(f'''
See an image being predicted by our model in real time!!
''')

st.subheader('''
             Dataset To Train
             ''')

XS1 = image_to_np('../project-biomassters/data/train data/' , '0060c0a5_S1_00.tif')
XS2 = image_to_np('../project-biomassters/data/train data/' , '0060c0a5_S2_06.tif')

col1, col2, col3 = st.columns(3)

with col1:
    st.image('../project-biomassters/data/train data/0060c0a5_S1_00.tif')
    st.image(XS1[:,:,3])
    st.image(XS2[:,:,2])
    st.image(XS2[:,:,5])
    st.image(XS2[:,:,8])
    st.image(XS2[:,:11])


with col2:
    st.image(XS1[:,:,1])
    st.image(XS2[:,:,0])
    st.image(XS2[:,:,3])
    st.image(XS2[:,:,6])
    st.image(XS2[:,:,9])

with col3:
    st.image(XS1[:,:,2])
    st.image(XS2[:,:,1])
    st.image(XS2[:,:,4])
    st.image(XS2[:,:,7])
    st.image(XS2[:,:,10])
