import streamlit as st
import pandas as pd
import os


with st.sidebar:
    data = st.file_uploader(label="upload")
    
if data is not None:
    # st.write(data.name)
    df = pd.read_csv(data) 
    st.sidebar.dataframe(df.head(5))

if data is not None:
    # Save the uploaded CSV file to the current working directory
    file_path = os.path.join(os.getcwd(), data.name)
    with open(file_path, "wb") as f:
        f.write(data.getbuffer())

    st.sidebar.write("file {} uploaded successfully".format(data.name))
