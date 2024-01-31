import io
import streamlit as st
from transformers import pipeline


st.title('RU/ENG translation')
text_from_st = st.text_input('Текст для перевода') #ввод пользователя
    
translator = pipeline("translation_ru_to_en", "Helsinki-NLP/opus-mt-ru-en")

def translate_model(ruText):
    st.write('Переведенный текст :',translator(ruText)[0]['translation_text']) #вывод ответа

translate_model(text_from_st)