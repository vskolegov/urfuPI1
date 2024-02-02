import torch
from transformers import AutoModelForSequenceClassification
from transformers import BertTokenizerFast
import streamlit as st

st.title('Определение сентиментальности текста')

text_from_st = st.text_input('Текст')

tokenizer = BertTokenizerFast.from_pretrained('blanchefort/rubert-base-cased-sentiment-rusentiment')
model = AutoModelForSequenceClassification.from_pretrained('blanchefort/rubert-base-cased-sentiment-rusentiment', return_dict=True)
if torch.cuda.is_available():
    model.cuda()  
    
@torch.no_grad()
def predict(text):
    inputs = tokenizer(text, max_length=512, padding=True, truncation=True, return_tensors='pt')
    outputs = model(**inputs)
    predicted_probabilities = torch.nn.functional.softmax(outputs.logits, dim=1).squeeze()
    predicted_label = torch.argmax(predicted_probabilities).item()  # Convert to a single int value
    return predicted_label, predicted_probabilities

def to_model(Text):
    try:
        predicted_label, predicted_probabilities = predict(Text)
        LABELS = ['без эмоций', 'радость', 'грусть', 'сюрприз', 'страх', 'злость']
        st.write('Распознанные емоции:')
        for i, emotion in enumerate(LABELS):
            percentage = predicted_probabilities[i].item() * 100
            st.write(f"{emotion}: {percentage:.2f}%")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if text_from_st.strip() != "":
    to_model(text_from_st)
else:
    st.warning("Пожалуйста, введите текст")