import streamlit as st
from transformers import pipeline

@st.cache(allow_output_mutation=True)

def load_model():
    model=pipeline("sentiment-analysis",   
                      "blanchefort/rubert-base-cased-sentiment")
    return model

model = load_model()
st.header ("Определение тональности текстов")
st.subheader ("Введите текст для анализа")
text = st.text_area(" ",height=100)
result = st.button("Определить тональность текста")


if result:
    res = model(text)
    sent = res[0]['label'] 
    st.write(model(text)[0]["label"])


