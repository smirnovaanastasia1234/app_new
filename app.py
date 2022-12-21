import numpy as np
import streamlit as st
from transformers import pipeline
from PIL import  Image
import text_analysis as nlp

# Title of the application 
st.title('Анализ тональности текста\n', )
st.subheader("Группа 32: Смирнова А., Кожедуб Н., Багаудинов Э., Петраков В.")

display = Image.open('images/display.jpg')
display = np.array(display)
st.image(display)

# Sidebar options
option = st.sidebar.selectbox('выбрать из списка', 
["Главная",
 "Определение тональности текста", 
  "Word Cloud", 
 ])

st.set_option('deprecation.showfileUploaderEncoding', False)

if option == 'Главная':
	st.write(
			"""
				## Описание проекта
				Это инструмент анализа текста, разработанный группой 32. Доступ к инстументам можно получить в левой боковой панели.
			"""
		)
elif option == "Определение тональности текста":



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

elif option == "Word Cloud":

	st.header("Generate Word Cloud")
	st.subheader("Generate a word cloud from text containing the most popular words in the text.")

	# Ask for text or text file
	st.header('Enter text or upload file')
	text = st.text_area('Type Something', height=400)

	# Upload mask image 
	mask = st.file_uploader('Use Image Mask', type = ['jpg'])

	# Add a button feature
	if st.button("Generate Wordcloud"):

		# Generate word cloud 
		st.write(len(text))
		nlp.create_wordcloud(text, mask)
		st.pyplot()