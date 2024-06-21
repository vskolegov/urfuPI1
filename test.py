from transformers import AutoModelForSequenceClassification, BertTokenizerFast
import torch
import streamlit_app

tokenizer = BertTokenizerFast.from_pretrained(
    'blanchefort/rubert-base-cased-sentiment-rusentiment'
)
model = AutoModelForSequenceClassification.from_pretrained(
    'blanchefort/rubert-base-cased-sentiment-rusentiment', return_dict=True
)


def predict_sentiment(text):
    inputs = tokenizer(text, max_length=512, padding=True, truncation=True, return_tensors='pt')
    outputs = model(**inputs)
    predicted = torch.nn.functional.softmax(outputs.logits, dim=1).squeeze()
    predicted_class = torch.argmax(predicted).item()
    return predicted_class


def test_predict_sentiment():
    text = "This is a test text"
    predicted_class = streamlit_app.predict_sentiment(text)
    assert predicted_class in [0, 1, 2], f"Invalid predicted class: {predicted_class}"


if __name__ == "__main__":
    test_predict_sentiment()
