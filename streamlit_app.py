import torch
from transformers import AutoModelForSequenceClassification, BertTokenizerFast
import streamlit as st
from typing import Tuple, List

# Инициализация токенизатора и модели
tokenizer = BertTokenizerFast.from_pretrained(
    'blanchefort/rubert-base-cased-sentiment-rusentiment'
)
model = AutoModelForSequenceClassification.from_pretrained(
    'blanchefort/rubert-base-cased-sentiment-rusentiment', return_dict=True
)

# Словарь меток эмоций
sentiment_labels = {
    0: "без эмоций",
    1: "радость",
    2: "грусть",
    3: "сюрприз",
    4: "страх",
    5: "злость"
}


@torch.no_grad()
def predict(text: str) -> Tuple[int, List[float]]:
    """
    Предсказывает метку и вероятности для заданного текста.

    :param text: Текст для анализа.
    :return: Предсказанная метка и вероятности для каждой метки.
    """
    inputs = tokenizer(text, max_length=512, padding=True, truncation=True, return_tensors='pt')
    outputs = model(**inputs)
    predicted = torch.nn.functional.softmax(outputs.logits, dim=1).squeeze()
    predicted_class = torch.argmax(predicted).item()
    probabilities = predicted.tolist()
    return predicted_class, probabilities


def display_results(predicted_class: int, probabilities: List[float]) -> None:
    """
    Отображает результаты анализа эмоций.

    :param predicted_class: Предсказанная метка.
    :param probabilities: Вероятности для каждой метки.
    """
    st.write(f"Найденные эмоции: {sentiment_labels[predicted_class]}")
    st.write("Вероятности:")
    for i, prob in enumerate(probabilities):
        sentiment_label = sentiment_labels[i]
        st.write(f"{sentiment_label}: {prob * 100:.2f}%")


def main() -> None:
    """
    Главная функция, отображающая интерфейс Streamlit и выполняющая анализ текста.
    """
    st.title("Анализ эмоций в тексте")
    user_input = st.text_input("Введите текст для анализа:", "")
    if st.button("Predict"):
        if user_input.strip():
            predicted_class, probabilities = predict(user_input)
            display_results(predicted_class, probabilities)


if __name__ == "__main__":
    main()
