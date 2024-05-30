import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm.local_llm import LocalLLM
import matplotlib.pyplot as plt
import streamlit as st

model = LocalLLM(
    api_base='http://localhost:11434/v1',
    model="llama3:latest"
)

st.title('Data Analysis with PandasAI')

# upload file
uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    sdf = SmartDataframe(df, config={"llm":model})
    st.write(df.head())

    prompt = st.text_area('Enter your prompt:')
    # button
    if st.button('Generate'):
        if prompt:
            with st.spinner('Generating response...'):
                st.write(sdf.chat(prompt))