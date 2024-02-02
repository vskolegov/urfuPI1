import torch
from transformers import AutoModelForSequenceClassification
from transformers import BertTokenizerFast
import streamlit as st

tokenizer = BertTokenizerFast.from_pretrained('blanchefort/rubert-base-cased-sentiment-rusentiment')
model = AutoModelForSequenceClassification.from_pretrained('blanchefort/rubert-base-cased-sentiment-rusentiment', return_dict=True)


def predict(text):
    inputs = tokenizer(text, max_length=512, padding=True, truncation=True, return_tensors='pt')
    outputs = model(**inputs)
    predicted = torch.nn.functional.softmax(outputs.logits, dim=1).squeeze()
    predicted_class = torch.argmax(predicted).item()
    probabilities = predicted.tolist()
    return predicted_class, probabilities

def main():
    st.title("Анализ эмоций в тексте")
    user_input = st.text_input("Введите текст для анализа:", "")
    if st.button("Predict"):
        if user_input.strip() != "":
            predicted_class, probabilities = predict(user_input)
            st.write(f"Найденные эмоции: {predicted_class}")
            st.write("вероятность:")
            for i, prob in enumerate(probabilities):
                st.write(f"{i}: {prob * 100:.2f}%")


if __name__ == "__main__":
    main()