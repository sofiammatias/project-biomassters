import streamlit as st
import pandas as pd
from biomassters.interface.main import load_dataset

st.image('https://images.unsplash.com/photo-1604009506606-fd4989d50e6d?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=870&q=80')


#         caption='Photo by <a href="https://unsplash.com/@chelseabock?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Chelsea Bock</a> on <a href="https://unsplash.com/s/photos/forest?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>')


'''
# BioMassters App
'''

st.markdown(f'''
This is an app to show the work done for BioMassters challenge.
''')

'''
# Overview
'''

st.markdown('''
**Do you ever fail to see the forest for the trees?** How about the aboveground
biomass for the forest? When it comes to understanding how forests change over
time, scientists and conservationists across the world look to this exact metric.
Aboveground biomass (ABGM) is a widespread measure in the study of carbon
release and sequestration by forests. Forests can act as carbon sinks by
removing carbon dioxide from the air through photosynthesis, but they can also
release carbon dioxide through wildfires, respiration, and decomposition.
**In order to understand how much carbon a forest contains (its carbon stock) and
how it changes (carbon flux), it is important to have an accurate measure of
AGBM. In turn, such information allows landowners and policy makers to make
better decisions for the conservation of forests.**
            ''')

st.markdown('''
            Link: [https://www.drivendata.org/competitions/99/biomass-estimation/page/534/](https://www.drivendata.org/competitions/99/biomass-estimation/page/534/)
            ''')
