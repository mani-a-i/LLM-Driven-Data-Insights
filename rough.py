import streamlit as st
import pandas as pd
import os


with st.sidebar:
    data = st.file_uploader(label="upload")
    
if data is not None:
    df = pd.read_csv(data) 
    st.sidebar.dataframe(df.head(5))

if data is not None: 
    file_path = os.path.join(os.getcwd(), data.name)
    with open(file_path, "wb") as f:
        f.write(data.getbuffer())
    st.sidebar.write("File {} uploaded successfully".format(data.name))
