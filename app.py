import numpy as np
import streamlit as st
from transformers import pipeline

# Title of the application 
st.title('Анализ тональности текста\n', )
st.subheader("by Group 32")

display = Image.open('images/display.jpg')
display = np.array(display)
st.image(display)

# Sidebar options
option = st.sidebar.selectbox('выбрать из списка', 
["Home",
 "Определение тональности текста", 
  "Word Cloud", 
 ])

st.set_option('deprecation.showfileUploaderEncoding', False)

if option == 'Home':
	st.write(
			"""
				## Project Description
				This is a complete text analysis tool developed by Group 32. It's built in with multiple features which can be accessed
				from the left side bar.
			"""
		)
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



