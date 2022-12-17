import streamlit as st
from transformers import pipeline


@st.cache(allow_output_mutation=True)

st.header("Определение тональности текстов")
st.subheader("Введите текст для анализа")
def load_model():
	  return torch.load("path/to/model.pt")

model = load_model()

text = st.text_area(" ",height=100)

classifier = pipeline("sentiment-analysis",   
                      "blanchefort/rubert-base-cased-sentiment")  

result = st.button("Определить тональность текста")
if result:
    x=pipeline("sentiment-analysis",   
                      "blanchefort/rubert-base-cased-sentiment")  
    st.write(classifier(text)[0]["label"])


