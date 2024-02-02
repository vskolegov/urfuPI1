import torch
from transformers import AutoModelForSequenceClassification
from transformers import BertTokenizerFast

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


text_input = 'Ты мне не нравишься. Я тебя не люблю'
LABELS = ['без эмоций', 'радость', 'грусть', 'сюрприз', 'страх', 'злость']

predicted_label, predicted_probabilities = predict(text_input)

top_3_emotions = torch.argsort(predicted_probabilities, descending=True)[:3]
top_3_emotions_with_percentage = [(LABELS[i], predicted_probabilities[i].item() * 100) for i in top_3_emotions]

print("Распознанные емоции:")
for emotion, percentage in top_3_emotions_with_percentage:
    print(f"{emotion}: {percentage:.2f}%")