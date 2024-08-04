from pandasai.llm.local_llm import LocalLLM
import pandas as pd
from pandasai import SmartDataframe
import streamlit as st


model = LocalLLM(
    api_base='http://127.0.0.1:11434/v1',
    model="mistral:instruct"
)

print(model)

st.title("Data analysis with pandasai")
uploaded_file = st.file_uploader("Choose a file", type=["csv", "txt"])

if uploaded_file is not None:
    if uploaded_file.type == "text":
        df = pd.read_csv(uploaded_file, sep="\t")
    else:
        df = pd.read_csv(uploaded_file)
    st.write(df)
    df = SmartDataframe(df, config={'llm': model})
    prompt = st.text_input("Enter your prompt")
    if st.button("Analyze"):
        if prompt:
            with st.spinner("Analyzing..."):
                st.write(df.chat(prompt))
