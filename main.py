import torch
from transformers import AutoModelForSequenceClassification, BertTokenizerFast
from typing import Tuple, List

# Инициализация токенизатора и модели
tokenizer = BertTokenizerFast.from_pretrained(
    'blanchefort/rubert-base-cased-sentiment-rusentiment'
)
model = AutoModelForSequenceClassification.from_pretrained(
    'blanchefort/rubert-base-cased-sentiment-rusentiment', return_dict=True
)

# Перемещение модели на GPU, если доступно
if torch.cuda.is_available():
    model.cuda()

LABELS = ['без эмоций', 'радость', 'грусть', 'сюрприз', 'страх', 'злость']
THRESHOLDS = [0.90, 0.75, 0.50, 0.25]


@torch.no_grad()
def predict(text: str) -> Tuple[int, torch.Tensor]:
    """
    Предсказывает метку и вероятности для заданного текста.

    :param text: Текст для анализа.
    :return: Предсказанная метка и вероятности для каждой метки.
    """
    inputs = tokenizer(text, max_length=512, padding=True, truncation=True, return_tensors='pt')
    outputs = model(**inputs)
    predicted_probabilities = torch.nn.functional.softmax(outputs.logits, dim=1).squeeze()
    predicted_label = torch.argmax(predicted_probabilities).item()
    return predicted_label, predicted_probabilities


def get_top_emotions(
    predicted_probabilities: torch.Tensor,
    thresholds: List[float],
    labels: List[str]
) -> List[Tuple[str, float]]:
    """
    Получает топовые эмоции с процентами на основе вероятностей и порогов.

    :param predicted_probabilities: Вероятности для каждой метки.
    :param thresholds: Пороги для отбора топовых эмоций.
    :param labels: Список меток.
    :return: Список кортежей (эмоция, процент).
    """
    top_emotions = torch.argsort(predicted_probabilities, descending=True)
    top_emotions_with_percentage = []
    for i in top_emotions:
        emotion = labels[i]
        percentage = predicted_probabilities[i].item() * 100
        if percentage >= thresholds[0]:
            top_emotions_with_percentage.append((emotion, percentage))
        elif percentage >= thresholds[1] and len(top_emotions_with_percentage) < 2:
            top_emotions_with_percentage.append((emotion, percentage))
        elif percentage >= thresholds[2] and len(top_emotions_with_percentage) < 3:
            top_emotions_with_percentage.append((emotion, percentage))
        elif percentage >= thresholds[3] and len(top_emotions_with_percentage) < 4:
            top_emotions_with_percentage.append((emotion, percentage))
    return top_emotions_with_percentage


def main(text: str) -> None:
    """
    Главная функция, выполняющая анализ текста и выводящая результаты.

    :param text: Текст для анализа.
    """
    predicted_label, predicted_probabilities = predict(text)
    top_emotions_with_percentage = get_top_emotions(predicted_probabilities, THRESHOLDS, LABELS)
    print("Top emotions with percentages:")
    for emotion, percentage in top_emotions_with_percentage:
        print(f"{emotion}: {percentage:.2f}%")


if __name__ == "__main__":
    text_input = 'Нифига себе, неужели так тоже бывает!'
    main(text_input)
