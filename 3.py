from fastapi import FastAPI
from transformers import pipeline
from pydantic import BaseModel

class Item(BaseModel):
    text:str

app = FastAPI()
translator = pipeline("translation_ru_to_en", "Helsinki-NLP/opus-mt-ru-en")
@app.get('/')
def root():
    return {'message':'This model translation text'}

@app.post('/translate/') #Перевод
def translate(item:Item):
    return translator(item.text)[0]