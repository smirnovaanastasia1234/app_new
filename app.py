import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from transformers import pipeline
import tensorflow as tf



st.header("Определение тональности текстов")
st.subheader("Введите текст для анализа")

text = st.text_area(" ",height=100)

classifier = pipeline("sentiment-analysis",   
                      "blanchefort/rubert-base-cased-sentiment")
st.write ("Тональность текста:")
st.write(classifier(text)[0]["label"])